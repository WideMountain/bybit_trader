import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import threading
import time
import traceback

from utils.bybit_api import BybitAPI
from utils.database import DatabaseManager
from utils.data_loaders import DataLoader
import config

logger = logging.getLogger(__name__)

class TradingBot:
    """Main trading bot class for strategy implementation"""
    
    def __init__(self, symbol: str, timeframe: str = '1h', api_key: Optional[str] = None, 
               api_secret: Optional[str] = None, testnet: Optional[bool] = None, 
               strategy_params: Optional[Dict] = None, risk_per_trade: Optional[float] = None,
               leverage: Optional[int] = None, db_url: Optional[str] = None):
        """
        Initialize TradingBot
        
        Parameters:
        symbol (str): Trading pair symbol
        timeframe (str): Timeframe (1m, 5m, 1h, 1D)
        api_key (str): Bybit API key
        api_secret (str): Bybit API secret
        testnet (bool): Use testnet if True, mainnet if False
        strategy_params (Dict): Strategy parameters
        risk_per_trade (float): Percentage of account to risk per trade
        leverage (int): Leverage to use
        db_url (str): Database connection URL
        """
        self.symbol = symbol
        self.timeframe = timeframe
        
        # API and database
        self.api = BybitAPI(api_key, api_secret, testnet)
        self.db = DatabaseManager(db_url)
        self.data_loader = DataLoader(api_key, api_secret, testnet, db_url)
        
        # Strategy and risk parameters
        self.strategy_params = strategy_params or config.STRATEGY_PARAMS.get(
            symbol, config.DEFAULT_STRATEGY_PARAMS
        )
        self.risk_per_trade = risk_per_trade or config.RISK_PER_TRADE_PCT
        self.leverage = leverage or config.DEFAULT_LEVERAGE
        
        # Runtime flags
        self.is_running = False
        self.stop_event = threading.Event()
        
        # Internal state
        self.data = pd.DataFrame()
        self.current_position = None
        self.last_signal = 0
        
        logger.info(f"TradingBot initialized for {symbol} on {timeframe} timeframe")
    
    def close(self):
        """Close connections"""
        self.db.close()
        self.data_loader.close()
    
    def fetch_data(self, limit: int = 500, update_latest: bool = True) -> pd.DataFrame:
        """
        Fetch data for the trading pair
        
        Parameters:
        limit (int): Number of candles to fetch
        update_latest (bool): Update latest data from API
        
        Returns:
        pd.DataFrame: Price data with indicators
        """
        # Determine start time based on limit and timeframe
        now = datetime.now()
        if self.timeframe == '1m':
            start_time = now - timedelta(minutes=limit)
        elif self.timeframe == '5m':
            start_time = now - timedelta(minutes=5 * limit)
        elif self.timeframe == '1h':
            start_time = now - timedelta(hours=limit)
        elif self.timeframe == '1D':
            start_time = now - timedelta(days=limit)
        else:
            start_time = now - timedelta(days=30)
        
        # Fetch historical data
        if update_latest:
            self.data = self.data_loader.update_latest_data(self.symbol, self.timeframe, limit)
        else:
            self.data = self.data_loader.fetch_historical_data(
                self.symbol, self.timeframe, start_time, now
            )
        
        # Calculate indicators
        self.data = self.data_loader.prepare_data_for_strategy(self.data, self.strategy_params)
        
        return self.data
    
    def generate_signal(self, data: Optional[pd.DataFrame] = None) -> int:
        """
        Generate trading signal
        
        Parameters:
        data (pd.DataFrame): Price data with indicators
        
        Returns:
        int: Signal (1: buy, -1: sell, 0: hold)
        """
        # Use provided data or the latest fetched data
        df = data if data is not None else self.data
        
        if df.empty:
            logger.warning("No data available to generate signal")
            return 0
        
        # Get the latest row for signal
        latest = df.iloc[-1]
        
        # Get current position if any
        current_position = self.get_current_position()
        
        # Default signal is hold
        signal = 0
        
        # Moving Average Crossover strategy
        if latest['sma_short'] > latest['sma_long']:
            # Bullish signal
            if current_position is None or current_position['side'] == 'short':
                signal = 1  # Buy/Long signal
        elif latest['sma_short'] < latest['sma_long']:
            # Bearish signal
            if current_position is None or current_position['side'] == 'long':
                signal = -1  # Sell/Short signal
        
        # Update last signal
        self.last_signal = signal
        
        return signal
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """
        Calculate position size based on risk parameters
        
        Parameters:
        entry_price (float): Entry price
        stop_loss (float): Stop loss price
        
        Returns:
        float: Position size in contracts
        """
        # Get account balance
        balance = self.get_account_balance()
        
        # Calculate risk amount
        risk_amount = balance * (self.risk_per_trade / 100)
        
        # Calculate position size
        position_size = self.api.calculate_position_size(
            symbol=self.symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            risk_amount=risk_amount,
            leverage=self.leverage
        )
        
        return position_size
    
    def calculate_stop_loss(self, signal: int, price: float) -> float:
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
    
    def calculate_take_profit(self, signal: int, entry_price: float, stop_loss: float) -> float:
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
    
    def execute_trade(self, signal: int) -> Dict:
        """
        Execute a trade based on signal
        
        Parameters:
        signal (int): Trading signal (1: buy, -1: sell)
        
        Returns:
        Dict: Trade information
        """
        # Get current price
        ticker = self.api.get_ticker(self.symbol)
        if not ticker:
            logger.error(f"Failed to get ticker for {self.symbol}")
            return {}
        
        current_price = ticker['last']
        
        # Calculate stop loss and take profit
        stop_loss = self.calculate_stop_loss(signal, current_price)
        take_profit = self.calculate_take_profit(signal, current_price, stop_loss)
        
        # Set leverage
        self.api.set_leverage(self.symbol, self.leverage)
        
        # Calculate position size
        position_size = self.calculate_position_size(current_price, stop_loss)
        
        # Determine order side
        side = 'buy' if signal == 1 else 'sell'
        
        # Execute order
        order = self.api.create_order(
            symbol=self.symbol,
            order_type='Market',
            side=side,
            amount=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        if not order:
            logger.error(f"Failed to create order for {self.symbol}")
            return {}
        
        # Save trade to database
        trade_data = {
            **order,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'strategy': 'MA_Crossover'
        }
        
        self.db.save_trade(trade_data)
        
        # Update position
        position_data = {
            'symbol': self.symbol,
            'timestamp': datetime.now(),
            'side': 'long' if side == 'buy' else 'short',
            'entry_price': current_price,
            'amount': position_size,
            'leverage': self.leverage,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'is_open': True
        }
        
        self.db.save_position(position_data)
        self.current_position = position_data
        
        logger.info(f"Executed {side} order for {position_size} {self.symbol} at {current_price}")
        
        return trade_data
    
    def close_position(self) -> Dict:
        """
        Close current position
        
        Returns:
        Dict: Trade information
        """
        # Get current position
        position = self.get_current_position()
        
        if not position:
            logger.warning("No position to close")
            return {}
        
        # Determine order side (opposite of position)
        side = 'sell' if position['side'] == 'long' else 'buy'
        
        # Get current price
        ticker = self.api.get_ticker(self.symbol)
        if not ticker:
            logger.error(f"Failed to get ticker for {self.symbol}")
            return {}
        
        current_price = ticker['last']
        
        # Execute order to close position
        order = self.api.create_order(
            symbol=self.symbol,
            order_type='Market',
            side=side,
            amount=position['amount'],
            reduce_only=True
        )
        
        if not order:
            logger.error(f"Failed to close position for {self.symbol}")
            return {}
        
        # Calculate realized PnL
        if position['side'] == 'long':
            realized_pnl = (current_price - position['entry_price']) * position['amount'] * self.leverage
        else:
            realized_pnl = (position['entry_price'] - current_price) * position['amount'] * self.leverage
        
        # Save trade to database
        trade_data = {
            **order,
            'realized_pnl': realized_pnl,
            'strategy': 'MA_Crossover'
        }
        
        self.db.save_trade(trade_data)
        
        # Update position
        position['is_open'] = False
        position['closed_at'] = datetime.now()
        position['exit_price'] = current_price
        position['realized_pnl'] = realized_pnl
        
        self.db.save_position(position)
        self.current_position = None
        
        logger.info(f"Closed {position['side']} position for {position['amount']} {self.symbol} at {current_price}")
        
        return trade_data
    
    def get_account_balance(self) -> float:
        """
        Get account balance
        
        Returns:
        float: Account balance
        """
        balance = self.api.get_account_balance()
        
        if not balance:
            logger.error("Failed to get account balance")
            return 10000.0  # Default balance
        
        return balance['total']
    
    def get_current_position(self) -> Optional[Dict]:
        """
        Get current position
        
        Returns:
        Dict: Position information, or None if no position
        """
        if self.current_position is not None:
            return self.current_position
        
        # Check database for open positions
        positions = self.db.get_positions(self.symbol, is_open=True, is_backtest=False)
        
        if positions:
            self.current_position = positions[0]
            return self.current_position
        
        # Check API for open positions
        api_positions = self.api.get_positions(self.symbol)
        
        if api_positions:
            # Convert from API format to internal format
            position = api_positions[0]
            position_data = {
                'symbol': position['symbol'],
                'timestamp': datetime.now() - timedelta(days=1),  # Approximate start
                'side': position['side'],
                'entry_price': position['entry_price'],
                'amount': position['size'],
                'leverage': position['leverage'],
                'liquidation_price': position['liquidation_price'],
                'unrealized_pnl': position['unrealized_pnl'],
                'is_open': True
            }
            
            # Save to database
            self.db.save_position(position_data)
            self.current_position = position_data
            
            return self.current_position
        
        return None
    
    def update_position_status(self) -> None:
        """Update current position status from API"""
        # Get current position from API
        position = self.get_current_position()
        
        if not position:
            return
        
        # Get current price
        ticker = self.api.get_ticker(self.symbol)
        if not ticker:
            logger.error(f"Failed to get ticker for {self.symbol}")
            return
        
        current_price = ticker['last']
        
        # Update unrealized PnL
        if position['side'] == 'long':
            unrealized_pnl = (current_price - position['entry_price']) * position['amount'] * self.leverage
        else:
            unrealized_pnl = (position['entry_price'] - current_price) * position['amount'] * self.leverage
        
        position['unrealized_pnl'] = unrealized_pnl
        
        # Save updated position
        self.db.save_position(position)
        self.current_position = position
    
    def run_once(self) -> Dict:
        """
        Execute one iteration of the trading strategy
        
        Returns:
        Dict: Status information
        """
        status = {
            'timestamp': datetime.now(),
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'signal': 0,
            'position': None,
            'action': 'none',
            'price': 0,
            'error': None
        }
        
        try:
            # Fetch latest data
            self.fetch_data(update_latest=True)
            
            if self.data.empty:
                status['error'] = "No data available"
                return status
            
            # Get current price
            ticker = self.api.get_ticker(self.symbol)
            if ticker:
                current_price = ticker['last']
                status['price'] = current_price
            
            # Generate signal
            signal = self.generate_signal()
            status['signal'] = signal
            
            # Get current position
            position = self.get_current_position()
            status['position'] = position
            
            # Update position status if we have one
            if position:
                self.update_position_status()
            
            # Execute trade based on signal
            if signal != 0:
                if position:
                    if (signal == 1 and position['side'] == 'short') or (signal == -1 and position['side'] == 'long'):
                        # Close existing position
                        self.close_position()
                        status['action'] = 'close'
                        
                        # Open new position in opposite direction
                        self.execute_trade(signal)
                        status['action'] = 'reverse'
                else:
                    # Open new position
                    self.execute_trade(signal)
                    status['action'] = 'open'
            
            logger.info(f"Run completed for {self.symbol} - Signal: {signal}, Action: {status['action']}")
            
        except Exception as e:
            logger.error(f"Error in run_once: {e}")
            logger.error(traceback.format_exc())
            status['error'] = str(e)
        
        return status
    
    def run_continuously(self, interval: int = 60) -> None:
        """
        Run the trading strategy continuously
        
        Parameters:
        interval (int): Interval in seconds between runs
        """
        if self.is_running:
            logger.warning("Trading bot is already running")
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        logger.info(f"Starting continuous trading for {self.symbol} on {self.timeframe} timeframe")
        
        while not self.stop_event.is_set():
            try:
                # Execute one iteration
                self.run_once()
                
                # Sleep until next interval
                for _ in range(interval):
                    if self.stop_event.is_set():
                        break
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Error in continuous trading: {e}")
                logger.error(traceback.format_exc())
                time.sleep(10)  # Sleep on error to avoid rapid retries
        
        self.is_running = False
        logger.info(f"Stopped continuous trading for {self.symbol}")
    
    def start(self, interval: int = 60) -> None:
        """
        Start trading in a separate thread
        
        Parameters:
        interval (int): Interval in seconds between runs
        """
        if self.is_running:
            logger.warning("Trading bot is already running")
            return
        
        # Start in a separate thread
        thread = threading.Thread(target=self.run_continuously, args=(interval,))
        thread.daemon = True
        thread.start()
        
        logger.info(f"Trading bot started for {self.symbol}")
    
    def stop(self) -> None:
        """Stop trading"""
        if not self.is_running:
            logger.warning("Trading bot is not running")
            return
        
        self.stop_event.set()
        logger.info(f"Stopping trading bot for {self.symbol}")
    
    def get_status(self) -> Dict:
        """
        Get current trading status
        
        Returns:
        Dict: Status information
        """
        status = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'is_running': self.is_running,
            'last_signal': self.last_signal,
            'current_position': self.get_current_position(),
            'account_balance': self.get_account_balance(),
            'leverage': self.leverage,
            'risk_per_trade': self.risk_per_trade
        }
        
        # Add latest price if available
        ticker = self.api.get_ticker(self.symbol)
        if ticker:
            status['current_price'] = ticker['last']
        
        return status
    
#hejhej