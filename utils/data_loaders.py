import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

from utils.bybit_api import BybitAPI
from utils.database import DatabaseManager
import config

logger = logging.getLogger(__name__)

class DataLoader:
    """Data loading and management for backtesting and live trading"""
    
    def __init__(self, api_key=None, api_secret=None, testnet=None, db_url=None):
        """
        Initialize DataLoader
        
        Parameters:
        api_key (str): Bybit API key
        api_secret (str): Bybit API secret
        testnet (bool): Use testnet if True, mainnet if False
        db_url (str): Database connection URL
        """
        self.api = BybitAPI(api_key, api_secret, testnet)
        self.db = DatabaseManager(db_url)
        
    def close(self):
        """Close connections"""
        self.db.close()
    
    def fetch_historical_data(self, symbol: str, timeframe: str, start_time: datetime, 
                            end_time: Optional[datetime] = None, force_download: bool = False) -> pd.DataFrame:
        """
        Fetch historical data for a symbol and timeframe
        
        Parameters:
        symbol (str): Trading pair symbol
        timeframe (str): Timeframe (1m, 5m, 1h, 1D)
        start_time (datetime): Start time
        end_time (datetime): End time (default: now)
        force_download (bool): Force download from API even if data exists
        
        Returns:
        pd.DataFrame: Historical data
        """
        if end_time is None:
            end_time = datetime.now()
        
        # Check if we already have the data
        if not force_download:
            # Get missing ranges
            missing_ranges = self.db.get_missing_candle_ranges(symbol, timeframe, start_time, end_time)
            
            if not missing_ranges:
                # All data available, just return it
                return self.db.get_candles(symbol, timeframe, start_time=start_time, end_time=end_time)
        else:
            # Force download the entire range
            missing_ranges = [(start_time, end_time)]
        
        # Download missing data
        all_candles = []
        for start, end in missing_ranges:
            candles = self._download_candles(symbol, timeframe, start, end)
            all_candles.extend(candles)
        
        # Save to database
        self.db.save_candles(symbol, timeframe, all_candles)
        
        # Return complete dataset
        return self.db.get_candles(symbol, timeframe, start_time=start_time, end_time=end_time)
    
    def update_latest_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """
        Update the latest data for a symbol and timeframe
        
        Parameters:
        symbol (str): Trading pair symbol
        timeframe (str): Timeframe (1m, 5m, 1h, 1D)
        limit (int): Number of candles to fetch
        
        Returns:
        pd.DataFrame: Updated data
        """
        # Get latest candle from database
        latest_candle = self.db.get_latest_candle(symbol, timeframe)
        
        # Determine start time for update
        if latest_candle:
            # Start from the last candle's timestamp
            start_time = latest_candle['timestamp']
            
            # Add one timeframe interval
            if timeframe == '1m':
                start_time += timedelta(minutes=1)
            elif timeframe == '5m':
                start_time += timedelta(minutes=5)
            elif timeframe == '1h':
                start_time += timedelta(hours=1)
            elif timeframe == '1D':
                start_time += timedelta(days=1)
        else:
            # No existing data, get data for the past day
            if timeframe == '1m':
                start_time = datetime.now() - timedelta(hours=24)
            elif timeframe == '5m':
                start_time = datetime.now() - timedelta(days=5)
            elif timeframe == '1h':
                start_time = datetime.now() - timedelta(days=30)
            elif timeframe == '1D':
                start_time = datetime.now() - timedelta(days=365)
        
        # Fetch new data
        candles = self._download_candles(symbol, timeframe, start_time, datetime.now())
        
        # Save to database
        if candles:
            self.db.save_candles(symbol, timeframe, candles)
        
        # Return updated data
        return self.db.get_candles(symbol, timeframe, limit=limit)
    
    def _download_candles(self, symbol: str, timeframe: str, start_time: datetime, 
                       end_time: datetime, batch_size: int = 1000) -> List[Dict]:
        """
        Download candlestick data from Bybit API
        
        Parameters:
        symbol (str): Trading pair symbol
        timeframe (str): Timeframe (1m, 5m, 1h, 1D)
        start_time (datetime): Start time
        end_time (datetime): End time
        batch_size (int): Batch size for each request
        
        Returns:
        List[Dict]: List of candlestick data
        """
        logger.info(f"Downloading {timeframe} data for {symbol} from {start_time} to {end_time}")
        
        # Convert datetime to milliseconds timestamp
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        
        all_candles = []
        current_start = start_ts
        
        while current_start < end_ts:
            try:
                # Calculate batch end time
                batch_end = min(current_start + (batch_size * self._get_timeframe_ms(timeframe)), end_ts)
                
                # Fetch candles
                candles = self.api.get_klines(
                    symbol=symbol,
                    interval=timeframe,
                    limit=batch_size,
                    start_time=current_start,
                    end_time=batch_end
                )
                
                if not candles:
                    # No data returned, move to next batch
                    current_start = batch_end
                    continue
                
                all_candles.extend(candles)
                
                # Update start time for next batch
                current_start = batch_end
                
                logger.info(f"Downloaded {len(candles)} candles")
                
            except Exception as e:
                logger.error(f"Error downloading candles: {e}")
                # Retry with a smaller batch
                batch_size = max(100, batch_size // 2)
                continue
        
        logger.info(f"Downloaded {len(all_candles)} candles in total")
        return all_candles
    
    def _get_timeframe_ms(self, timeframe: str) -> int:
        """
        Get timeframe interval in milliseconds
        
        Parameters:
        timeframe (str): Timeframe (1m, 5m, 1h, 1D)
        
        Returns:
        int: Interval in milliseconds
        """
        if timeframe == '1m':
            return 60 * 1000
        elif timeframe == '5m':
            return 5 * 60 * 1000
        elif timeframe == '1h':
            return 60 * 60 * 1000
        elif timeframe == "4h":
         return 4 * 60 * 60 * 1000
        elif timeframe == '1D':
            return 24 * 60 * 60 * 1000
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    def prepare_data_for_strategy(self, df: pd.DataFrame, strategy_params: Dict) -> pd.DataFrame:
        """
        Prepare data for strategy by calculating indicators
        
        Parameters:
        df (pd.DataFrame): Raw price data
        strategy_params (Dict): Strategy parameters
        
        Returns:
        pd.DataFrame: Data with indicators
        """
        if df.empty:
            return df
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Calculate SMA indicators for MA crossover strategy
        short_window = strategy_params.get('short_window', 20)
        long_window = strategy_params.get('long_window', 50)
        
        df['sma_short'] = df['close'].rolling(window=short_window).mean()
        df['sma_long'] = df['close'].rolling(window=long_window).mean()
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['sma_short'] > df['sma_long'], 'signal'] = 1  # Buy signal
        df.loc[df['sma_short'] < df['sma_long'], 'signal'] = -1  # Sell signal
        
        # Remove NaN values
        df.dropna(inplace=True)
        
        return df