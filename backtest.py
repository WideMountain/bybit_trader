import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.database import DatabaseManager
from utils.data_loaders import DataLoader
from utils.performance import PerformanceAnalyzer
import config

logger = logging.getLogger(__name__)

class Backtester:
    """Backtesting engine for trading strategies"""
    
    def __init__(self, symbol: str, timeframe: str = '1h', 
               strategy_params: Optional[Dict] = None, 
               risk_per_trade: float = 1.0, leverage: int = 1,
               initial_balance: float = 10000.0, db_url: Optional[str] = None,
               api_key: Optional[str] = None, api_secret: Optional[str] = None, 
               testnet: bool = True,
               strategy_class: Optional[type] = None):
        """
        Initialize Backtester
        
        Parameters:
        symbol (str): Trading pair symbol
        timeframe (str): Timeframe (1m, 5m, 1h, 1D)
        strategy_params (Dict): Strategy parameters
        risk_per_trade (float): Percentage of account to risk per trade
        leverage (int): Leverage to use
        initial_balance (float): Initial account balance
        db_url (str): Database connection URL
        api_key (str): Bybit API key (for fetching historical data)
        api_secret (str): Bybit API secret (for fetching historical data)
        testnet (bool): Use testnet if True, mainnet if False (for fetching historical data)

    Note:
        Slippage of 0.02% is applied to both entry and exit prices in backtests.
        """
        
        self.symbol = symbol
        self.timeframe = timeframe
        self.strategy_params = strategy_params or config.STRATEGY_PARAMS.get(
            symbol, config.DEFAULT_STRATEGY_PARAMS
        )
        # Enforce risk_per_trade between 1 and 10
        self.risk_per_trade = min(max(risk_per_trade, 1.0), 10.0)
        self.leverage = min(max(leverage, 1), 50)
        self.initial_balance = initial_balance
        
        # Database and data loader
        self.db = DatabaseManager(db_url)
        self.data_loader = DataLoader(api_key, api_secret, testnet, db_url)
        self.performance = PerformanceAnalyzer(db_url)
        
        # Internal state
        self.data = pd.DataFrame()
        self.trades = []
        self.positions = []
        self.equity_curve = pd.DataFrame()
        self.balance = initial_balance
        self.slippage = 0.0002  # 0.02% slippage applied to entry and exit prices
        
        self.strategy_class = strategy_class  # <-- Add this
        
        logger.info(f"Backtester initialized for {symbol} on {timeframe} timeframe")
    
    def close(self):
        """Close connections"""
        self.db.close()
        self.data_loader.close()
    
    def load_data(self, start_date: str, end_date: Optional[str] = None, 
                force_download: bool = False) -> pd.DataFrame:
        """
        Load historical data for backtesting
        
        Parameters:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        force_download (bool): Force download from API even if data exists
        
        Returns:
        pd.DataFrame: Historical data with indicators
        """
        # Convert string dates to datetime
        start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
        if end_date:
            end_datetime = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            end_datetime = datetime.now()
        
        # Fetch historical data
        self.data = self.data_loader.fetch_historical_data(
            self.symbol, self.timeframe, start_datetime, end_datetime, force_download
        )
        
        # Calculate indicators
        self.data = self.data_loader.prepare_data_for_strategy(self.data, self.strategy_params)
        
        return self.data
    
    def run_backtest(self, start_date: str, end_date: Optional[str] = None, 
                   force_download: bool = False) -> Dict:
        """
        Run backtest

        Parameters:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        force_download (bool): Force download from API even if data exists

        Returns:
        Dict: Backtest results
        """
        # Clear previous backtest data
        self.db.clear_backtest_data(self.symbol)
        self.trades = []
        self.positions = []
        self.balance = self.initial_balance

        # Load data
        self.load_data(start_date, end_date, force_download)

        if self.data.empty:
            logger.error("No data available for backtesting")
            return {
                'success': False,
                'error': "No data available"
            }

        # --- Generate signals using the selected strategy ---
        if self.strategy_class is not None:
            strategy = self.strategy_class(self.strategy_params)
            def signal_func(idx, df):
                return strategy.generate_signal(df.iloc[:idx+1], None)
            self.data['signal'] = [signal_func(i, self.data) for i in range(len(self.data))]
        elif 'signal' not in self.data.columns:
            logger.error("No 'signal' column in data and no strategy_class provided.")
            return {
                'success': False,
                'error': "No 'signal' column in data and no strategy_class provided."
            }

        # --- Prevent lookahead bias: shift signal by 1 ---
        self.data['signal'] = self.data['signal'].shift(1)

        # Reset index to iterate through rows
        data = self.data.reset_index()
        
        # Initialize variables
        current_position = None
        
        # Simulate trading
        logger.info(f"Running backtest for {self.symbol} from {start_date} to {end_date or 'now'}")
        
        for i, row in tqdm(data.iterrows(), total=len(data), desc="Backtesting"):
            # Skip the first window to allow indicators to initialize
            if i < max(self.strategy_params.get('short_window', 20), self.strategy_params.get('long_window', 50)):
                continue
            
            # Get signal from data
            signal = row['signal']
            
            # Get current price and timestamp
            current_price = row['close']
            timestamp = row['timestamp'] if isinstance(row['timestamp'], datetime) else pd.to_datetime(row['timestamp'])

            # Get next bar's open price and timestamp if available, else use current
            if i + 1 < len(data):
                next_row = data.iloc[i + 1]
                execution_price = next_row['open'] if 'open' in next_row else next_row['close']
                execution_timestamp = next_row['timestamp'] if isinstance(next_row['timestamp'], datetime) else pd.to_datetime(next_row['timestamp'])
            else:
                execution_price = current_price
                execution_timestamp = timestamp

            # Process signal
            if signal != 0:
                if current_position:
                    # Check if signal is opposite to current position
                    if (signal == 1 and current_position['side'] == 'short') or (signal == -1 and current_position['side'] == 'long'):
                        # Close existing position at next bar's open
                        self._close_position(current_position, execution_price, execution_timestamp)
                        current_position = None

                        # Open new position at next bar's open
                        current_position = self._open_position(signal, execution_price, execution_timestamp)
                else:
                    # Open new position at next bar's open
                    current_position = self._open_position(signal, execution_price, execution_timestamp)
            
            # Update unrealized PnL for open position
            if current_position:
                if current_position['side'] == 'long':
                    unrealized_pnl = (current_price - current_position['entry_price']) * current_position['amount']
                else:
                    unrealized_pnl = (current_position['entry_price'] - current_price) * current_position['amount']
                current_position['unrealized_pnl'] = unrealized_pnl

            # --- Liquidation check: stop if balance is zero or less ---
            if self.balance <= 0:
                logger.warning("Account liquidated! Stopping backtest.")
                break
        
        # Close any remaining position at the end of the backtest
        if current_position:
            last_price = data['close'].iloc[-1]
            last_timestamp = data['timestamp'].iloc[-1]
            self._close_position(current_position, last_price, last_timestamp)
        
        # Calculate performance metrics
        metrics = self.performance.analyze_backtest(self.trades, self.positions, self.initial_balance)
        
        # Generate equity curve
        self.equity_curve = self.performance._calculate_equity_curve(pd.DataFrame(self.trades), self.initial_balance)
        
        logger.info(f"Backtest completed with {len(self.trades)} trades")
        
        return {
            'success': True,
            'metrics': metrics,
            'trades': self.trades,
            'positions': self.positions,
            'equity_curve': self.equity_curve
        }
    
    def _calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """
        Calculate position size based on risk parameters

        Parameters:
        entry_price (float): Entry price
        stop_loss (float): Stop loss price

        Returns:
        float: Position size in contracts
        """
        risk_per_contract = abs(entry_price - stop_loss) / entry_price
        min_risk = 0.001
        if risk_per_contract < min_risk:
            risk_per_contract = min_risk

        risk_amount = self.initial_balance * (self.risk_per_trade / 100)
        position_size = risk_amount / risk_per_contract

        # Cap position size so notional value does not exceed balance * leverage
        max_notional = self.initial_balance * self.leverage
        max_position_size = max_notional / entry_price
        position_size = min(position_size, max_position_size)

        # Remove hard cap (let max_position_size logic handle it for all assets)
        # position_size = min(position_size, 1.0)

        position_size = round(abs(position_size), 6)  # Always positive

        logger.debug(
            f"Calculated position size: {position_size} | "
            f"risk_per_contract: {risk_per_contract} | "
            f"risk_amount: {risk_amount} | "
            f"entry_price: {entry_price} | stop_loss: {stop_loss}"
        )

        return position_size
    
    def _calculate_stop_loss(self, signal: int, price: float) -> float:
        """
        Calculate stop loss price based on signal and current price
        
        Parameters:
        signal (int): Trading signal (1: buy, -1: sell)
        price (float): Current price
        
        Returns:
        float: Stop loss price
        """
        # Default ATR multiplier for stop loss
        atr_multiplier = 2.0
        
        # If we have ATR in the data, use it for stop loss
        if 'atr' in self.data.columns:
            atr = self.data['atr'].iloc[-1]
        else:
            # Calculate a simple volatility measure (high - low)
            recent_data = self.data.tail(14)
            volatility = (recent_data['high'] - recent_data['low']).mean()
            atr = volatility
        
        # Calculate stop loss based on signal
        if signal == 1:  # Long position
            stop_loss = price - (atr * atr_multiplier)
        elif signal == -1:  # Short position
            stop_loss = price + (atr * atr_multiplier)
        else:
            stop_loss = price
        
        # Round to appropriate precision
        stop_loss = round(stop_loss, 2)
        
        return stop_loss
    
    def _calculate_take_profit(self, signal: int, entry_price: float, stop_loss: float) -> float:
        """
        Calculate take profit price based on signal, entry price, and stop loss
        
        Parameters:
        signal (int): Trading signal (1: buy, -1: sell)
        entry_price (float): Entry price
        stop_loss (float): Stop loss price
        
        Returns:
        float: Take profit price
        """
        # Risk to reward ratio
        risk_reward_ratio = 2.0
        
        # Calculate risk
        risk = abs(entry_price - stop_loss)
        
        # Calculate take profit
        if signal == 1:  # Long position
            take_profit = entry_price + (risk * risk_reward_ratio)
        elif signal == -1:  # Short position
            take_profit = entry_price - (risk * risk_reward_ratio)
        else:
            take_profit = entry_price
        
        # Round to appropriate precision
        take_profit = round(take_profit, 2)
        
        return take_profit
    
    def _open_position(self, signal: int, price: float, timestamp: datetime) -> Dict:
        """
        Open a position in backtest
        
        Parameters:
        signal (int): Trading signal (1: buy, -1: sell)
        price (float): Entry price
        timestamp (datetime): Entry timestamp
        
        Returns:
        Dict: Position information
        """
        # Determine position side
        side = 'long' if signal == 1 else 'short'
        
        # Calculate stop loss
        stop_loss = self._calculate_stop_loss(signal, price)
        
        # Calculate take profit
        take_profit = self._calculate_take_profit(signal, price, stop_loss)
        
        # Calculate position size
        position_size = self._calculate_position_size(price, stop_loss)
        
        # Apply slippage to entry price
        if signal == 1:  # Long entry
            price = price * (1 + self.slippage)
        elif signal == -1:  # Short entry
            price = price * (1 - self.slippage)
        entry_fee = price * position_size * 0.0006  # 0.06% fee
        self.balance -= entry_fee  # Deduct entry fee from balance
        
        # Create position
        position = {
            'symbol': self.symbol,
            'timestamp': timestamp,
            'side': side,
            'entry_price': price,
            'amount': position_size,
            'leverage': self.leverage,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'unrealized_pnl': 0,
            'is_open': True,
            'is_backtest': True
        }
        
        # Create trade record
        trade = {
            'symbol': self.symbol,
            'timestamp': timestamp,
            'side': 'buy' if side == 'long' else 'sell',
            'price': price,
            'amount': position_size,
            'cost': price * position_size,
            'fee': price * position_size * 0.0006,  # Assume 0.06% fee
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'is_backtest': True,
            'strategy': self.strategy_class.__name__ if self.strategy_class else "Unknown"
        }
        
        # Save to database and local lists
        self.db.save_position(position, is_backtest=True)
        self.db.save_trade(trade, is_backtest=True)
        
        self.positions.append(position)
        self.trades.append(trade)
        
        return position
    
    def _close_position(self, position: Dict, price: float, timestamp: datetime) -> Dict:
        """
        Close a position in backtest
        
        Parameters:
        position (Dict): Position to close
        price (float): Exit price
        timestamp (datetime): Exit timestamp
        
        Returns:
        Dict: Updated position information
        """
        # Apply slippage to exit price
        if position['side'] == 'long':
            price = price * (1 - self.slippage)  # Sell, so worse price
            realized_pnl = (price - position['entry_price']) * position['amount']
        else:
            price = price * (1 + self.slippage)  # Buy to cover, so worse price
            realized_pnl = (position['entry_price'] - price) * position['amount']
        exit_fee = price * position['amount'] * 0.0006  # 0.06% fee
        self.balance += realized_pnl - exit_fee  # Add PnL, subtract exit fee
        
        
        # Determine trade side (opposite of position)
        side = 'sell' if position['side'] == 'long' else 'buy'
        
        # Create trade record
        trade = {
            'symbol': self.symbol,
            'timestamp': timestamp,
            'side': side,
            'price': price,
            'amount': position['amount'],
            'cost': price * position['amount'],
            'fee': price * position['amount'] * 0.0006,  # Assume 0.06% fee
            'realized_pnl': realized_pnl,
            'is_backtest': True,
            'strategy': self.strategy_class.__name__ if self.strategy_class else "Unknown"
        }
        
        # Update position
        position['is_open'] = False
        position['closed_at'] = timestamp
        position['exit_price'] = price
        position['realized_pnl'] = realized_pnl
        
        # Save to database and local lists
        self.db.save_position(position, is_backtest=True)
        self.db.save_trade(trade, is_backtest=True)
        
        self.trades.append(trade)
        
        return position
    
    def plot_results(self, save_path: Optional[str] = None) -> None:
        """
        Plot backtest results
        
        Parameters:
        save_path (str, optional): Path to save the plots
        """
        if not self.trades:
            logger.warning("No trades to plot")
            return
        
        # Plot equity curve
        self.performance.plot_equity_curve(self.trades, self.initial_balance, 
                                        save_path=(save_path + '_equity.png') if save_path else None)
        
        # Plot trade distribution
        self.performance.plot_trade_distribution(self.trades, 
                                              save_path=(save_path + '_distribution.png') if save_path else None)
        
        # Plot price chart with entries and exits
        self._plot_trades_on_chart(save_path=(save_path + '_trades.png') if save_path else None)
    
    def _plot_trades_on_chart(self, save_path: Optional[str] = None) -> None:
        """
        Plot price chart with trade entries and exits
        
        Parameters:
        save_path (str, optional): Path to save the plot
        """
        if self.data.empty or not self.trades:
            logger.warning("No data or trades to plot")
            return
        
        # Create figure
        plt.figure(figsize=(14, 8))
        
        # Plot price
        plt.plot(self.data.index, self.data['close'], label='Price', color='blue', alpha=0.6)
        
        # Plot moving averages
        if 'sma_short' in self.data.columns:
            plt.plot(self.data.index, self.data['sma_short'], label=f"SMA {self.strategy_params.get('short_window', 20)}", color='orange', alpha=0.7)
        if 'sma_long' in self.data.columns:
            plt.plot(self.data.index, self.data['sma_long'], label=f"SMA {self.strategy_params.get('long_window', 50)}", color='green', alpha=0.7)
        
        # Plot trade entries and exits
        for trade in self.trades:
            timestamp = trade['timestamp']
            price = trade['price']
            
            if trade['side'] == 'buy':
                plt.scatter(timestamp, price, color='green', marker='^', s=100, label='_Buy')
            else:
                plt.scatter(timestamp, price, color='red', marker='v', s=100, label='_Sell')
        
        # Add labels and title
        plt.title(f'Backtest Results for {self.symbol} ({self.timeframe})')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        
        # Add legend with unique labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper left')
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
    
    def get_performance_report(self) -> str:
        """
        Generate performance report
        
        Returns:
        str: Formatted performance report
        """
        if not self.trades:
            return "No trades to analyze"
        
        return self.performance.generate_performance_report(self.trades, self.positions, self.initial_balance)
