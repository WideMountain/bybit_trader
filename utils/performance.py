import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

from utils.database import DatabaseManager
import config

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """Performance analysis for trading strategies"""
    
    def __init__(self, db_url=None):
        """
        Initialize PerformanceAnalyzer
        
        Parameters:
        db_url (str): Database connection URL
        """
        self.db = DatabaseManager(db_url)
    
    def analyze_backtest(self, trades: List[Dict], positions: List[Dict], 
                       initial_balance: float = 10000.0) -> Dict:
        """
        Analyze backtest performance
        
        Parameters:
        trades (List[Dict]): List of trades
        positions (List[Dict]): List of positions
        initial_balance (float): Initial account balance
        
        Returns:
        Dict: Performance metrics
        """
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_return': 0,
                'annualized_return': 0,
                'avg_trade_duration': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0
            }
        
        # Convert trades to DataFrame for analysis
        trades_df = pd.DataFrame(trades)
        
        # Calculate basic metrics
        winning_trades = trades_df[trades_df['realized_pnl'] > 0]
        losing_trades = trades_df[trades_df['realized_pnl'] < 0]
        
        total_trades = len(trades_df)
        winning_trades_count = len(winning_trades)
        losing_trades_count = len(losing_trades)
        
        # Win rate
        win_rate = winning_trades_count / total_trades if total_trades > 0 else 0
        
        # Profit metrics
        total_profit = winning_trades['realized_pnl'].sum() if not winning_trades.empty else 0
        total_loss = abs(losing_trades['realized_pnl'].sum()) if not losing_trades.empty else 0
        net_profit = total_profit - total_loss
        
        # Profit factor
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Print the last 10 trades for debugging
        print(trades_df[['timestamp', 'realized_pnl', 'fee']].tail(10))

        # Calculate equity curve
        equity_curve = self._calculate_equity_curve(trades_df, initial_balance)

        # --- ADD THESE DEBUG PRINTS ---
        print("Equity curve tail:\n", equity_curve.tail())
        print("Initial equity:", initial_balance)
        if not equity_curve.empty:
            print("Final equity:", equity_curve['equity'].iloc[-1])
        else:
            print("Equity curve is empty!")
        # --- END DEBUG PRINTS ---

        # Drawdown analysis
        max_drawdown, max_drawdown_duration = self._calculate_max_drawdown(equity_curve)

        # Return metrics
        if initial_balance > 0 and not equity_curve.empty:
            last_equity = equity_curve['equity'].iloc[-1]
            if pd.isna(last_equity):
                total_return = 0.0
            else:
                total_return = (last_equity / initial_balance - 1) * 100
        else:
            total_return = 0.0

        # Annualized return and Sharpe ratio
        annualized_return = 0
        sharpe_ratio = 0.0
        if isinstance(equity_curve.index, pd.DatetimeIndex):
            daily_equity = equity_curve['equity'].resample('1D').last().ffill()
            daily_returns = daily_equity.pct_change().dropna()
            if not daily_returns.empty and daily_returns.std() > 0:
                sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
            if len(daily_equity) > 1:
                start_date = daily_equity.index[0]
                end_date = daily_equity.index[-1]
                years = (end_date - start_date).days / 365
                annualized_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
            else:
                annualized_return = 0
        else:
            daily_returns = equity_curve['equity'].pct_change().dropna()
            if not daily_returns.empty and daily_returns.std() > 0:
                sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(len(daily_returns))
            else:
                sharpe_ratio = 0.0
            annualized_return = 0
        
        # Trade duration
        avg_trade_duration = 0
        if 'timestamp' in trades_df.columns and 'closed_at' in trades_df.columns:
            trades_df['duration'] = (trades_df['closed_at'] - trades_df['timestamp']).dt.total_seconds() / 3600  # hours
            avg_trade_duration = trades_df['duration'].mean()
        
        # Best and worst trades
        best_trade = winning_trades['realized_pnl'].max() if not winning_trades.empty else 0
        worst_trade = losing_trades['realized_pnl'].min() if not losing_trades.empty else 0
        
        # Average profit and loss
        avg_profit = winning_trades['realized_pnl'].mean() if not winning_trades.empty else 0
        avg_loss = losing_trades['realized_pnl'].mean() if not losing_trades.empty else 0
        
        # Consecutive wins and losses
        consecutive_wins, consecutive_losses = self._calculate_consecutive_trades(trades_df)
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades_count,
            'losing_trades': losing_trades_count,
            'win_rate': win_rate * 100,
            'profit_factor': profit_factor,
            'net_profit': net_profit,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'max_drawdown_duration': max_drawdown_duration,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'avg_trade_duration': avg_trade_duration,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses
        }
    
    def _calculate_equity_curve(self, trades_df: pd.DataFrame, initial_balance: float) -> pd.DataFrame:
        """
        Calculate equity curve from trades

        Parameters:
        trades_df (pd.DataFrame): Trades DataFrame
        initial_balance (float): Initial account balance

        Returns:
        pd.DataFrame: Equity curve
        """
        if trades_df.empty:
            # Return empty DataFrame with initial balance
            return pd.DataFrame({'equity': [initial_balance]}, index=[datetime.now()])

        # Sort trades by timestamp
        trades_df = trades_df.sort_values('timestamp')

        # Ensure timestamps are datetime
        if not np.issubdtype(trades_df['timestamp'].dtype, np.datetime64):
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])

        # Create equity curve
        equity = initial_balance
        equity_data = []

        for _, trade in trades_df.iterrows():
            fee = trade['fee'] if pd.notna(trade.get('fee', 0)) else 0
            realized_pnl = trade['realized_pnl'] if pd.notna(trade.get('realized_pnl', 0)) else 0
            equity += realized_pnl - fee
            equity_data.append({'timestamp': trade['timestamp'], 'equity': equity})

        # Convert to DataFrame
        equity_df = pd.DataFrame(equity_data)
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        equity_df.set_index('timestamp', inplace=True)
        equity_df.sort_index(inplace=True)

        # Ensure index is a DatetimeIndex
        if not isinstance(equity_df.index, pd.DatetimeIndex):
            equity_df.index = pd.to_datetime(equity_df.index)

        return equity_df
    
    def _calculate_max_drawdown(self, equity_curve: pd.DataFrame) -> Tuple[float, int]:
        """
        Calculate maximum drawdown and duration
        
        Parameters:
        equity_curve (pd.DataFrame): Equity curve
        
        Returns:
        Tuple[float, int]: (Max drawdown as fraction, Max drawdown duration in days)
        """
        if equity_curve.empty or 'equity' not in equity_curve.columns:
            return 0.0, 0
        
        # Calculate drawdown
        equity = equity_curve['equity']
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        
        # Calculate max drawdown
        max_drawdown = drawdown.min()
        
        # Calculate max drawdown duration
        in_drawdown = drawdown < 0
        if not any(in_drawdown):
            return 0.0, 0
            
        # Find the longest sequence of consecutive drawdown days
        is_dd = in_drawdown.astype(int)
        is_dd_change = is_dd.diff().fillna(0).abs()
        dd_start_indices = is_dd_change[is_dd_change == 1].index.tolist()
        
        if not dd_start_indices:
            return 0.0, 0
            
        # Add the start of the series if we begin in a drawdown
        if is_dd.iloc[0] == 1:
            dd_start_indices = [equity_curve.index[0]] + dd_start_indices
            
        # Add the end of the series if we end in a drawdown
        dd_end_indices = is_dd_change.shift(-1)
        dd_end_indices = dd_end_indices[dd_end_indices == 1].index.tolist()
        
        if is_dd.iloc[-1] == 1:
            dd_end_indices = dd_end_indices + [equity_curve.index[-1]]
            
        # Calculate durations
        if len(dd_start_indices) > len(dd_end_indices):
            dd_start_indices = dd_start_indices[:len(dd_end_indices)]
        elif len(dd_end_indices) > len(dd_start_indices):
            dd_end_indices = dd_end_indices[:len(dd_start_indices)]
            
        durations = [(end - start).days for start, end in zip(dd_start_indices, dd_end_indices)]
        max_dd_duration = max(durations) if durations else 0
        
        return abs(max_drawdown), max_dd_duration
    
    def _calculate_consecutive_trades(self, trades_df: pd.DataFrame) -> Tuple[int, int]:
        """
        Calculate maximum consecutive winning and losing trades
        
        Parameters:
        trades_df (pd.DataFrame): Trades DataFrame
        
        Returns:
        Tuple[int, int]: (Max consecutive wins, Max consecutive losses)
        """
        if trades_df.empty or 'realized_pnl' not in trades_df.columns:
            return 0, 0
        
        # Create win/loss series
        trades_df = trades_df.sort_values('timestamp')
        is_win = (trades_df['realized_pnl'] > 0).astype(int)
        
        # Count consecutive wins and losses
        win_streak = 0
        loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        for is_winning in is_win:
            if is_winning:
                win_streak += 1
                loss_streak = 0
                max_win_streak = max(max_win_streak, win_streak)
            else:
                loss_streak += 1
                win_streak = 0
                max_loss_streak = max(max_loss_streak, loss_streak)
        
        return max_win_streak, max_loss_streak
    
    def plot_equity_curve(self, trades: List[Dict], initial_balance: float = 10000.0, 
                        save_path: Optional[str] = None) -> None:
        """
        Plot equity curve
        
        Parameters:
        trades (List[Dict]): List of trades
        initial_balance (float): Initial account balance
        save_path (str, optional): Path to save the plot
        """
        if not trades:
            logger.warning("No trades to plot equity curve")
            return
        
        # Calculate equity curve
        trades_df = pd.DataFrame(trades)
        equity_curve = self._calculate_equity_curve(trades_df, initial_balance)
        
        if equity_curve.empty:
            logger.warning("Empty equity curve, cannot plot")
            return
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot equity curve
        plt.plot(equity_curve.index, equity_curve['equity'], label='Equity', color='blue')
        
        # Calculate drawdown
        peak = equity_curve['equity'].expanding().max()
        drawdown = (equity_curve['equity'] - peak) / peak
        
        # Plot drawdown
        ax2 = plt.gca().twinx()
        ax2.fill_between(equity_curve.index, 0, drawdown * 100, color='red', alpha=0.3, label='Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_ylim(drawdown.min() * 100 * 1.5, 5)  # Set y-axis limit with some padding
        
        # Add labels and title
        plt.title('Equity Curve and Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.grid(True, alpha=0.3)
        
        # Add legend
        lines1, labels1 = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
    
    def plot_trade_distribution(self, trades: List[Dict], save_path: Optional[str] = None) -> None:
        """
        Plot trade profit distribution
        
        Parameters:
        trades (List[Dict]): List of trades
        save_path (str, optional): Path to save the plot
        """
        if not trades:
            logger.warning("No trades to plot distribution")
            return
        
        # Convert to DataFrame
        trades_df = pd.DataFrame(trades)
        
        if 'realized_pnl' not in trades_df.columns or trades_df.empty:
            logger.warning("No realized PnL data to plot distribution")
            return
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot histogram
        plt.hist(trades_df['realized_pnl'], bins=50, alpha=0.7, color='blue')
        
        # Add labels and title
        plt.title('Trade Profit Distribution')
        plt.xlabel('Profit/Loss')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Add vertical line at zero
        plt.axvline(x=0, color='red', linestyle='--')
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
    
    def generate_performance_report(self, trades: List[Dict], positions: List[Dict], 
                                  initial_balance: float = 10000.0) -> str:
        """
        Generate performance report as a formatted string
        
        Parameters:
        trades (List[Dict]): List of trades
        positions (List[Dict]): List of positions
        initial_balance (float): Initial account balance
        
        Returns:
        str: Formatted performance report
        """
        # Analyze performance
        metrics = self.analyze_backtest(trades, positions, initial_balance)
        
        # Format report
        report = "===== PERFORMANCE REPORT =====\n\n"
        
        report += f"Total Trades: {metrics['total_trades']}\n"
        report += f"Winning Trades: {metrics['winning_trades']} ({metrics['win_rate']:.2f}%)\n"
        report += f"Losing Trades: {metrics['losing_trades']}\n\n"
        
        report += f"Net Profit: ${metrics['net_profit']:.2f}\n"
        report += f"Total Return: {metrics['total_return']:.2f}%\n"
        report += f"Annualized Return: {metrics['annualized_return']:.2f}%\n"
        report += f"Profit Factor: {metrics['profit_factor']:.2f}\n\n"
        
        report += f"Max Drawdown: {metrics['max_drawdown']:.2f}%\n"
        report += f"Max Drawdown Duration: {metrics['max_drawdown_duration']} days\n"
        report += f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n\n"
        
        report += f"Avg Trade Duration: {metrics['avg_trade_duration']:.2f} hours\n"
        report += f"Best Trade: ${metrics['best_trade']:.2f}\n"
        report += f"Worst Trade: ${metrics['worst_trade']:.2f}\n"
        report += f"Avg Profit: ${metrics['avg_profit']:.2f}\n"
        report += f"Avg Loss: ${metrics['avg_loss']:.2f}\n\n"
        
        report += f"Max Consecutive Wins: {metrics['max_consecutive_wins']}\n"
        report += f"Max Consecutive Losses: {metrics['max_consecutive_losses']}\n"
        
        return report