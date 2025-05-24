import logging
import argparse
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading
import pandas as pd
import matplotlib.pyplot as plt
from strategies.ma_crossover import MACrossoverStrategy
from strategies.rsi_strategy import RSIStrategy

STRATEGY_MAP = {
    "MA Crossover": MACrossoverStrategy,
    "RSI": RSIStrategy,
}

from tabulate import tabulate
import colorama
from colorama import Fore, Style

from trading_bot import TradingBot
from backtest import Backtester
from live_trading import LiveTrader
import config

# Initialize colorama
colorama.init()

# Set up logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class TradingBotApp:
    """Command-line interface for the trading bot"""
    
    def __init__(self):
        """Initialize the application"""
        self.api_key = config.BYBIT_API_KEY
        self.api_secret = config.BYBIT_API_SECRET
        self.testnet = config.BYBIT_TESTNET
        
        self.symbols = config.TRADING_SYMBOLS
        self.timeframe = config.DEFAULT_TIMEFRAME
        self.risk_per_trade = config.RISK_PER_TRADE_PCT
        self.leverage = config.DEFAULT_LEVERAGE
        
        self.trader = None
        self.backtesters = {}
        self.bots = {}
        
        self.running = True
        self.stop_event = threading.Event()
        
        # Check API credentials
        if not self.api_key or not self.api_secret:
            logger.error("API key and secret not found. Please set them in .env file.")
            print(f"{Fore.RED}ERROR: API key and secret not found. Please set them in .env file.{Style.RESET_ALL}")
            sys.exit(1)
        
         # === ADD STRATEGY SELECTION PROMPT HERE ===
        print("\nAvailable strategies:")
        for idx, name in enumerate(STRATEGY_MAP.keys(), 1):
            print(f"{idx}. {name}")
        try:
            strategy_idx = int(input("Select strategy [1]: ") or "1") - 1
            self.strategy_name = list(STRATEGY_MAP.keys())[strategy_idx]
        except (ValueError, IndexError):
            print(f"{Fore.YELLOW}Invalid selection, defaulting to MA Crossover.{Style.RESET_ALL}")
            self.strategy_name = "MA Crossover"
        self.strategy_class = STRATEGY_MAP[self.strategy_name]
        # === END STRATEGY SELECTION PROMPT === 

    def print_header(self):
        """Print application header"""
        print("\n" + "=" * 80)
        print(f"{Fore.CYAN}Bybit Crypto Trading Bot{Style.RESET_ALL}")
        print(f"Running on {'TESTNET' if self.testnet else 'MAINNET'}")
        print("=" * 80 + "\n")
    
    def print_menu(self):
        """Print main menu"""
        print("\n" + "-" * 40)
        print(f"{Fore.YELLOW}Main Menu:{Style.RESET_ALL}")
        print("-" * 40)
        print(f"1. {Fore.GREEN}Live Trading{Style.RESET_ALL}")
        print(f"2. {Fore.BLUE}Backtesting{Style.RESET_ALL}")
        print(f"3. {Fore.CYAN}Configuration{Style.RESET_ALL}")
        print(f"4. {Fore.MAGENTA}Account Info{Style.RESET_ALL}")
        print(f"0. {Fore.RED}Exit{Style.RESET_ALL}")
        print("-" * 40)
    
    def run(self):
        """Run the application"""
        self.print_header()
        
        while self.running:
            self.print_menu()
            choice = input("Enter your choice [0-4]: ")
            
            if choice == '0':
                self.exit_app()
                break
            elif choice == '1':
                self.live_trading_menu()
            elif choice == '2':
                self.backtesting_menu()
            elif choice == '3':
                self.configuration_menu()
            elif choice == '4':
                self.show_account_info()
            else:
                print(f"{Fore.RED}Invalid choice. Please try again.{Style.RESET_ALL}")
    
    def live_trading_menu(self):
        """Live trading menu"""
        while True:
            print("\n" + "-" * 40)
            print(f"{Fore.GREEN}Live Trading Menu:{Style.RESET_ALL}")
            print("-" * 40)
            print(f"1. Start Trading (All Symbols)")
            print(f"2. Start Trading (Single Symbol)")
            print(f"3. Stop Trading")
            print(f"4. Show Trading Status")
            print(f"0. Back to Main Menu")
            print("-" * 40)
            
            choice = input("Enter your choice [0-4]: ")
            
            if choice == '0':
                break
            elif choice == '1':
                self.start_live_trading()
            elif choice == '2':
                self.start_single_symbol_trading()
            elif choice == '3':
                self.stop_live_trading()
            elif choice == '4':
                self.show_trading_status()
            else:
                print(f"{Fore.RED}Invalid choice. Please try again.{Style.RESET_ALL}")
    
    def backtesting_menu(self):
        """Backtesting menu"""
        while True:
            print("\n" + "-" * 40)
            print(f"{Fore.BLUE}Backtesting Menu:{Style.RESET_ALL}")
            print("-" * 40)
            print(f"1. Run Backtest")
            print(f"2. Show Backtest Results")
            print(f"3. Plot Backtest Results")
            print(f"0. Back to Main Menu")
            print("-" * 40)
            
            choice = input("Enter your choice [0-3]: ")
            
            if choice == '0':
                break
            elif choice == '1':
                self.run_backtest()
            elif choice == '2':
                self.show_backtest_results()
            elif choice == '3':
                self.plot_backtest_results()
            else:
                print(f"{Fore.RED}Invalid choice. Please try again.{Style.RESET_ALL}")
    
    def configuration_menu(self):
        """Configuration menu"""
        while True:
            print("\n" + "-" * 40)
            print(f"{Fore.CYAN}Configuration Menu:{Style.RESET_ALL}")
            print("-" * 40)
            print(f"1. Change Trading Symbols")
            print(f"2. Change Timeframe")
            print(f"3. Change Risk Per Trade")
            print(f"4. Change Leverage")
            print(f"5. Toggle Testnet/Mainnet")
            print(f"6. Show Current Configuration")
            print(f"0. Back to Main Menu")
            print("-" * 40)
            
            choice = input("Enter your choice [0-6]: ")
            
            if choice == '0':
                break
            elif choice == '1':
                self.change_trading_symbols()
            elif choice == '2':
                self.change_timeframe()
            elif choice == '3':
                self.change_risk_per_trade()
            elif choice == '4':
                self.change_leverage()
            elif choice == '5':
                self.toggle_testnet()
            elif choice == '6':
                self.show_configuration()
            else:
                print(f"{Fore.RED}Invalid choice. Please try again.{Style.RESET_ALL}")
    
    def start_live_trading(self):
        """Start live trading on all symbols"""
        if self.trader and self.trader.is_running:
            print(f"{Fore.YELLOW}Trading is already running.{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.GREEN}Starting live trading on all symbols...{Style.RESET_ALL}")
        
        # Get check interval
        try:
            interval = int(input("Enter check interval in seconds [60]: ") or "60")
        except ValueError:
            interval = 60
            
        # Create trader if not exists
        if not self.trader:
            self.trader = LiveTrader(
                symbols=self.symbols,
                timeframe=self.timeframe,
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet,
                check_interval=interval
            )
        
        # Start trading
        self.trader.start()
        
        print(f"{Fore.GREEN}Live trading started on: {', '.join(self.symbols)}{Style.RESET_ALL}")
    
    def start_single_symbol_trading(self):
        """Start trading on a single symbol"""
        if self.trader and self.trader.is_running:
            print(f"{Fore.YELLOW}Trading is already running. Stop it first.{Style.RESET_ALL}")
            return
        
        # Select symbol
        print(f"\nAvailable symbols: {', '.join(self.symbols)}")
        symbol = input("Enter symbol to trade: ").upper()
        
        if symbol not in self.symbols:
            print(f"{Fore.RED}Invalid symbol. Please choose from the available symbols.{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.GREEN}Starting live trading on {symbol}...{Style.RESET_ALL}")
        
        # Get check interval
        try:
            interval = int(input("Enter check interval in seconds [60]: ") or "60")
        except ValueError:
            interval = 60
            
        # Create individual bot if not exists
        if symbol not in self.bots:
            self.bots[symbol] = TradingBot(
                symbol=symbol,
                timeframe=self.timeframe,
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet,
                risk_per_trade=self.risk_per_trade,
                leverage=self.leverage,
                strategy_class=self.strategy_class  # <-- Add this line
            )
        
        # Start trading
        self.bots[symbol].start(interval)
        
        print(f"{Fore.GREEN}Live trading started on {symbol}{Style.RESET_ALL}")
    
    def stop_live_trading(self):
        """Stop all live trading"""
        stopped = False
        
        # Stop trader if running
        if self.trader and self.trader.is_running:
            self.trader.stop()
            stopped = True
        
        # Stop individual bots if running
        for symbol, bot in self.bots.items():
            if bot.is_running:
                bot.stop()
                stopped = True
        
        if stopped:
            print(f"{Fore.YELLOW}Live trading stopped.{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}No active trading to stop.{Style.RESET_ALL}")
    
    def show_trading_status(self):
        """Show current trading status"""
        has_status = False
        
        # Show trader status
        if self.trader:
            status = self.trader.get_status()
            
            print("\n" + "=" * 80)
            print(f"{Fore.GREEN}Trading Status:{Style.RESET_ALL}")
            print("=" * 80)
            print(f"Running: {status['is_running']}")
            print(f"Timeframe: {status['timeframe']}")
            print(f"Environment: {'Testnet' if status['testnet'] else 'Mainnet'}")
            print(f"Check Interval: {status['check_interval']} seconds")
            print("-" * 80)
            
            # Print status for each bot
            for symbol, bot_status in status['bots'].items():
                print(f"\n{Fore.CYAN}Symbol: {symbol}{Style.RESET_ALL}")
                print(f"  Running: {bot_status['is_running']}")
                print(f"  Last Signal: {bot_status['last_signal']}")
                if 'current_price' in bot_status:
                    print(f"  Current Price: {bot_status['current_price']}")
                print(f"  Risk Per Trade: {bot_status['risk_per_trade']}%")
                print(f"  Leverage: {bot_status['leverage']}x")
                
                # Print position if any
                if bot_status['current_position']:
                    pos = bot_status['current_position']
                    print(f"  {Fore.YELLOW}Current Position:{Style.RESET_ALL}")
                    print(f"    Side: {pos['side']}")
                    print(f"    Entry Price: {pos['entry_price']}")
                    print(f"    Amount: {pos['amount']}")
                    if 'unrealized_pnl' in pos:
                        pnl = pos['unrealized_pnl']
                        color = Fore.GREEN if pnl >= 0 else Fore.RED
                        print(f"    Unrealized PnL: {color}{pnl:.2f}{Style.RESET_ALL}")
            
            has_status = True
        
        # Show individual bot status
        for symbol, bot in self.bots.items():
            if not self.trader or symbol not in self.trader.bots:
                status = bot.get_status()
                
                print(f"\n{Fore.CYAN}Individual Bot - Symbol: {symbol}{Style.RESET_ALL}")
                print(f"  Running: {status['is_running']}")
                print(f"  Last Signal: {status['last_signal']}")
                if 'current_price' in status:
                    print(f"  Current Price: {status['current_price']}")
                print(f"  Risk Per Trade: {status['risk_per_trade']}%")
                print(f"  Leverage: {status['leverage']}x")
                
                # Print position if any
                if status['current_position']:
                    pos = status['current_position']
                    print(f"  {Fore.YELLOW}Current Position:{Style.RESET_ALL}")
                    print(f"    Side: {pos['side']}")
                    print(f"    Entry Price: {pos['entry_price']}")
                    print(f"    Amount: {pos['amount']}")
                    if 'unrealized_pnl' in pos:
                        pnl = pos['unrealized_pnl']
                        color = Fore.GREEN if pnl >= 0 else Fore.RED
                        print(f"    Unrealized PnL: {color}{pnl:.2f}{Style.RESET_ALL}")
                
                has_status = True
        
        if not has_status:
            print(f"{Fore.YELLOW}No active trading bots.{Style.RESET_ALL}")
    
    def run_backtest(self):
        """Run backtest"""
        # Select symbol
        print(f"\nAvailable symbols: {', '.join(self.symbols)}")
        symbol = input("Enter symbol to backtest: ").upper()
        
        if symbol not in self.symbols:
            print(f"{Fore.RED}Invalid symbol. Please choose from the available symbols.{Style.RESET_ALL}")
            return
        
        # Select timeframe
        print(f"\nAvailable timeframes: {', '.join(config.AVAILABLE_TIMEFRAMES.keys())}")
        timeframe = input(f"Enter timeframe [{self.timeframe}]: ") or self.timeframe
        
        if timeframe not in config.AVAILABLE_TIMEFRAMES:
            print(f"{Fore.RED}Invalid timeframe. Please choose from the available timeframes.{Style.RESET_ALL}")
            return
        
        # Get date range
        default_start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        start_date = input(f"Enter start date [{default_start}]: ") or default_start
        end_date = input("Enter end date [today]: ") or datetime.now().strftime('%Y-%m-%d')
        
        # Get risk per trade and leverage
        try:
            risk = float(input(f"Enter risk per trade % [{self.risk_per_trade}]: ") or str(self.risk_per_trade))
        except ValueError:
            risk = self.risk_per_trade
            
        try:
            leverage = int(input(f"Enter leverage [{self.leverage}]: ") or str(self.leverage))
        except ValueError:
            leverage = self.leverage
        
        # Get initial balance
        try:
            initial_balance = float(input("Enter initial balance [10000]: ") or "10000")
        except ValueError:
            initial_balance = 10000.0
        
        # Force download option
        force_download = input("Force download historical data? (y/n) [n]: ").lower() == 'y'
        
        print(f"\n{Fore.BLUE}Running backtest for {symbol} on {timeframe} timeframe...{Style.RESET_ALL}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Risk: {risk}%, Leverage: {leverage}x, Initial Balance: ${initial_balance}")
        
        # Create backtester
        backtester = Backtester(
            symbol=symbol,
            timeframe=timeframe,
            strategy_params=config.STRATEGY_PARAMS.get(symbol, config.DEFAULT_STRATEGY_PARAMS),
            risk_per_trade=risk,
            leverage=leverage,
            initial_balance=initial_balance,
            api_key=self.api_key,
            api_secret=self.api_secret,
            testnet=self.testnet,
            strategy_class=self.strategy_class  # <-- Add this line if supported
        )
        
        # Run backtest
        results = backtester.run_backtest(start_date, end_date, force_download)
        
        if not results['success']:
            print(f"{Fore.RED}Backtest failed: {results.get('error', 'Unknown error')}{Style.RESET_ALL}")
            return
        
        # Store backtester for later use
        self.backtesters[symbol] = backtester
        
        # Show results summary
        metrics = results['metrics']
        
        print("\n" + "=" * 80)
        print(f"{Fore.BLUE}Backtest Results Summary:{Style.RESET_ALL}")
        print("=" * 80)
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Net Profit: ${metrics['net_profit']:.2f}")
        print(f"Total Return: {metrics['total_return']:.2f}%")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print("=" * 80)
        
        # Ask if user wants to plot results
        if input("\nPlot results? (y/n) [y]: ").lower() != 'n':
            backtester.plot_results()
    
    def show_backtest_results(self):
        """Show detailed backtest results"""
        if not self.backtesters:
            print(f"{Fore.YELLOW}No backtest results available. Run a backtest first.{Style.RESET_ALL}")
            return
        
        # Select symbol
        symbols = list(self.backtesters.keys())
        print(f"\nAvailable backtest results: {', '.join(symbols)}")
        symbol = input("Enter symbol to show results: ").upper()
        
        if symbol not in self.backtesters:
            print(f"{Fore.RED}No backtest results for {symbol}.{Style.RESET_ALL}")
            return
        
        # Get backtester
        backtester = self.backtesters[symbol]
        
        # Show detailed report
        report = backtester.get_performance_report()
        
        print("\n" + "=" * 80)
        print(f"{Fore.BLUE}Detailed Backtest Results for {symbol}:{Style.RESET_ALL}")
        print("=" * 80)
        print(report)
        print("=" * 80)
        
        # Show trades
        show_trades = input("\nShow all trades? (y/n) [n]: ").lower() == 'y'
        
        if show_trades and backtester.trades:
            print("\n" + "=" * 80)
            print(f"{Fore.BLUE}Trades:{Style.RESET_ALL}")
            print("=" * 80)
            
            # Prepare trade data for tabulate
            trade_data = []
            for trade in backtester.trades[:20]:  # Show first 20 trades
                trade_data.append([
                    trade['timestamp'].strftime('%Y-%m-%d %H:%M'),
                    trade['side'],
                    f"{trade['price']:.2f}",
                    f"{trade['amount']:.4f}",
                    f"{trade.get('realized_pnl', 0):.2f}" if 'realized_pnl' in trade else 'N/A'
                ])
            
            headers = ["Timestamp", "Side", "Price", "Amount", "PnL"]
            print(tabulate(trade_data, headers=headers, tablefmt="grid"))
            
            if len(backtester.trades) > 20:
                print(f"\nShowing first 20 of {len(backtester.trades)} trades.")
    
    def plot_backtest_results(self):
        """Plot backtest results"""
        if not self.backtesters:
            print(f"{Fore.YELLOW}No backtest results available. Run a backtest first.{Style.RESET_ALL}")
            return
        
        # Select symbol
        symbols = list(self.backtesters.keys())
        print(f"\nAvailable backtest results: {', '.join(symbols)}")
        symbol = input("Enter symbol to plot results: ").upper()
        
        if symbol not in self.backtesters:
            print(f"{Fore.RED}No backtest results for {symbol}.{Style.RESET_ALL}")
            return
        
        # Get backtester
        backtester = self.backtesters[symbol]
        
        # Plot results
        backtester.plot_results()
    
    def change_trading_symbols(self):
        """Change trading symbols"""
        print(f"\nCurrent trading symbols: {', '.join(self.symbols)}")
        
        # Show available symbols
        print("\nEnter new trading symbols (comma-separated):")
        new_symbols_input = input("> ")
        
        if not new_symbols_input:
            return
        
        new_symbols = [s.strip().upper() for s in new_symbols_input.split(',')]
        
        if not new_symbols:
            print(f"{Fore.RED}No valid symbols entered.{Style.RESET_ALL}")
            return
        
        self.symbols = new_symbols
        print(f"{Fore.GREEN}Trading symbols updated to: {', '.join(self.symbols)}{Style.RESET_ALL}")
        
        # Reset trader to apply new symbols
        if self.trader and self.trader.is_running:
            print(f"{Fore.YELLOW}Please restart trading to apply new symbols.{Style.RESET_ALL}")
    
    def change_timeframe(self):
        """Change trading timeframe"""
        print(f"\nCurrent timeframe: {self.timeframe}")
        print(f"Available timeframes: {', '.join(config.AVAILABLE_TIMEFRAMES.keys())}")
        
        new_timeframe = input("Enter new timeframe: ")
        
        if not new_timeframe:
            return
        
        if new_timeframe not in config.AVAILABLE_TIMEFRAMES:
            print(f"{Fore.RED}Invalid timeframe. Please choose from the available timeframes.{Style.RESET_ALL}")
            return
        
        self.timeframe = new_timeframe
        print(f"{Fore.GREEN}Timeframe updated to: {self.timeframe}{Style.RESET_ALL}")
        
        # Reset trader to apply new timeframe
        if self.trader and self.trader.is_running:
            print(f"{Fore.YELLOW}Please restart trading to apply new timeframe.{Style.RESET_ALL}")
    
    def change_risk_per_trade(self):
        """Change risk per trade percentage"""
        print(f"\nCurrent risk per trade: {self.risk_per_trade}%")
        
        try:
            new_risk = float(input("Enter new risk per trade percentage: "))
        except ValueError:
            print(f"{Fore.RED}Invalid input. Please enter a number.{Style.RESET_ALL}")
            return
        
        if new_risk <= 0 or new_risk > 100:
            print(f"{Fore.RED}Invalid risk percentage. Must be between 0 and 100.{Style.RESET_ALL}")
            return
        
        self.risk_per_trade = new_risk
        print(f"{Fore.GREEN}Risk per trade updated to: {self.risk_per_trade}%{Style.RESET_ALL}")
        
        # Update trader and bots
        if self.trader:
            for symbol, bot in self.trader.bots.items():
                bot.risk_per_trade = new_risk
        
        for symbol, bot in self.bots.items():
            bot.risk_per_trade = new_risk
    
    def change_leverage(self):
        """Change leverage"""
        print(f"\nCurrent leverage: {self.leverage}x")
        
        try:
            new_leverage = int(input("Enter new leverage: "))
        except ValueError:
            print(f"{Fore.RED}Invalid input. Please enter a number.{Style.RESET_ALL}")
            return
        
        if new_leverage <= 0 or new_leverage > config.MAX_LEVERAGE:
            print(f"{Fore.RED}Invalid leverage. Must be between 1 and {config.MAX_LEVERAGE}.{Style.RESET_ALL}")
            return
        
        self.leverage = new_leverage
        print(f"{Fore.GREEN}Leverage updated to: {self.leverage}x{Style.RESET_ALL}")
        
        # Update trader and bots
        if self.trader:
            for symbol, bot in self.trader.bots.items():
                bot.leverage = new_leverage
        
        for symbol, bot in self.bots.items():
            bot.leverage = new_leverage
    
    def toggle_testnet(self):
        """Toggle between testnet and mainnet"""
        current = "TESTNET" if self.testnet else "MAINNET"
        new_mode = "MAINNET" if self.testnet else "TESTNET"
        
        confirm = input(f"\nYou are about to switch from {current} to {new_mode}. Are you sure? (y/n): ").lower()
        
        if confirm != 'y':
            return
        
        self.testnet = not self.testnet
        print(f"{Fore.GREEN}Switched to {new_mode}{Style.RESET_ALL}")
        
        # Reset trader to apply new mode
        if self.trader and self.trader.is_running:
            print(f"{Fore.YELLOW}Please restart trading to apply changes.{Style.RESET_ALL}")
    
    def show_configuration(self):
        """Show current configuration"""
        print("\n" + "=" * 80)
        print(f"{Fore.CYAN}Current Configuration:{Style.RESET_ALL}")
        print("=" * 80)
        print(f"Trading Symbols: {', '.join(self.symbols)}")
        print(f"Timeframe: {self.timeframe}")
        print(f"Risk Per Trade: {self.risk_per_trade}%")
        print(f"Leverage: {self.leverage}x")
        print(f"Environment: {'TESTNET' if self.testnet else 'MAINNET'}")
        print("-" * 80)
        
        # Show strategy parameters for each symbol
        print(f"{Fore.CYAN}Strategy Parameters:{Style.RESET_ALL}")
        for symbol in self.symbols:
            params = config.STRATEGY_PARAMS.get(symbol, config.DEFAULT_STRATEGY_PARAMS)
            print(f"  {symbol}:")
            print(f"    Short Window: {params.get('short_window', 20)}")
            print(f"    Long Window: {params.get('long_window', 50)}")
        
        print("=" * 80)
    
    def show_account_info(self):
        """Show account information"""
        try:
            # Create API instance if needed
            from utils.bybit_api import BybitAPI
            api = BybitAPI(self.api_key, self.api_secret, self.testnet)
            
            # Get account balance
            balance = api.get_account_balance()
            
            if not balance:
                print(f"{Fore.RED}Failed to get account information.{Style.RESET_ALL}")
                return
            
            # Get positions
            positions = api.get_positions()
            
            print("\n" + "=" * 80)
            print(f"{Fore.MAGENTA}Account Information:{Style.RESET_ALL}")
            print("=" * 80)
            print(f"Environment: {'TESTNET' if self.testnet else 'MAINNET'}")
            print(f"Balance (USDT): {balance['total']}")
            print(f"Available: {balance['free']}")
            print(f"In Use: {balance['used']}")
            
            # Show positions
            if positions:
                print("\n" + "-" * 80)
                print(f"{Fore.YELLOW}Open Positions:{Style.RESET_ALL}")
                print("-" * 80)
                
                for pos in positions:
                    side_color = Fore.GREEN if pos['side'] == 'long' else Fore.RED
                    print(f"Symbol: {pos['symbol']}")
                    print(f"Side: {side_color}{pos['side']}{Style.RESET_ALL}")
                    print(f"Size: {pos['size']}")
                    print(f"Entry Price: {pos['entry_price']}")
                    print(f"Leverage: {pos['leverage']}x")
                    
                    if 'unrealized_pnl' in pos:
                        pnl = pos['unrealized_pnl']
                        pnl_color = Fore.GREEN if pnl >= 0 else Fore.RED
                        print(f"Unrealized PnL: {pnl_color}{pnl:.2f}{Style.RESET_ALL}")
                    
                    print(f"Liquidation Price: {pos.get('liquidation_price', 'N/A')}")
                    print("-" * 40)
            else:
                print("\nNo open positions.")
            
            print("=" * 80)
            
        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
            print(f"{Fore.RED}Error fetching account information: {e}{Style.RESET_ALL}")
    
    def exit_app(self):
        """Exit the application"""
        print(f"\n{Fore.YELLOW}Shutting down...{Style.RESET_ALL}")
        
        # Stop trading
        if self.trader and self.trader.is_running:
            self.trader.stop()
        
        for symbol, bot in self.bots.items():
            if bot.is_running:
                bot.stop()
        
        # Close connections
        if self.trader:
            self.trader.close()
        
        for symbol, bot in self.bots.items():
            bot.close()
        
        for symbol, backtester in self.backtesters.items():
            backtester.close()
        
        self.running = False
        print(f"{Fore.GREEN}Goodbye!{Style.RESET_ALL}")

def main():
    """Main entry point"""
    # Set up logging
    os.makedirs('logs', exist_ok=True)
    
    # Register signal handlers
    def signal_handler(sig, frame):
        print(f"\n{Fore.YELLOW}Interrupt received, shutting down...{Style.RESET_ALL}")
        if 'app' in globals():
            app.exit_app()
        sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and run application
    app = TradingBotApp()
    app.run()
    # Haj här testar vi
    # här testar vi igen

if __name__ == "__main__":
    main()