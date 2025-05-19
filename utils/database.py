import os
import logging
import pandas as pd
import sqlite3
from typing import Dict, List, Optional, Tuple, Union
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Table, MetaData, select, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta

import config

logger = logging.getLogger(__name__)
Base = declarative_base()

class OHLCV(Base):
    """OHLCV candlestick data table"""
    __tablename__ = 'ohlcv'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    timeframe = Column(String, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    def __repr__(self):
        return f"<OHLCV(symbol='{self.symbol}', timestamp='{self.timestamp}', close={self.close})>"


class Trade(Base):
    """Trade execution records"""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    order_id = Column(String, nullable=True)  # May be null for backtest trades
    symbol = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    side = Column(String, nullable=False)  # 'buy' or 'sell'
    price = Column(Float, nullable=False)
    amount = Column(Float, nullable=False)
    cost = Column(Float, nullable=False)  # price * amount
    fee = Column(Float, nullable=True)
    fee_coin = Column(String, nullable=True)
    position_size = Column(Float, nullable=True)  # After this trade
    realized_pnl = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    is_backtest = Column(Boolean, default=False)
    strategy = Column(String, nullable=True)
    
    def __repr__(self):
        return f"<Trade(symbol='{self.symbol}', side='{self.side}', price={self.price}, amount={self.amount})>"


class Position(Base):
    """Position tracking table"""
    __tablename__ = 'positions'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    side = Column(String, nullable=False)  # 'long' or 'short'
    entry_price = Column(Float, nullable=False)
    amount = Column(Float, nullable=False)
    leverage = Column(Float, nullable=False, default=1.0)
    liquidation_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    unrealized_pnl = Column(Float, nullable=True)
    is_open = Column(Boolean, default=True, index=True)
    closed_at = Column(DateTime, nullable=True)
    exit_price = Column(Float, nullable=True)
    realized_pnl = Column(Float, nullable=True)
    is_backtest = Column(Boolean, default=False)
    
    def __repr__(self):
        return f"<Position(symbol='{self.symbol}', side='{self.side}', entry_price={self.entry_price}, amount={self.amount})>"


class DatabaseManager:
    """Database management for trading data"""
    
    def __init__(self, db_url=None):
        """
        Initialize DatabaseManager
        
        Parameters:
        db_url (str): Database connection URL
        """
        self.db_url = db_url or config.DATABASE_URL
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        logger.info(f"Database initialized at {self.db_url}")
    
    def close(self):
        """Close database session"""
        if self.session:
            self.session.close()
    
    def save_candles(self, symbol: str, timeframe: str, candles: List[Dict]) -> int:
        """
        Save candlestick data to database
        
        Parameters:
        symbol (str): Trading pair symbol
        timeframe (str): Timeframe (1m, 5m, 1h, 1D)
        candles (List[Dict]): List of candlestick data
        
        Returns:
        int: Number of candles saved
        """
        if not candles:
            return 0
            
        try:
            count = 0
            for candle in candles:
                # Convert timestamp to datetime
                if isinstance(candle['timestamp'], pd.Timestamp):
                    timestamp = candle['timestamp'].to_pydatetime()
                elif isinstance(candle['timestamp'], int):
                    timestamp = datetime.fromtimestamp(candle['timestamp'] / 1000)
                else:
                    timestamp = candle['timestamp']
                
                # Check if candle already exists
                existing = self.session.query(OHLCV).filter(
                    OHLCV.symbol == symbol,
                    OHLCV.timestamp == timestamp,
                    OHLCV.timeframe == timeframe
                ).first()
                
                if existing:
                    # Update existing candle
                    existing.open = candle['open']
                    existing.high = candle['high']
                    existing.low = candle['low']
                    existing.close = candle['close']
                    existing.volume = candle['volume']
                else:
                    # Create new candle
                    new_candle = OHLCV(
                        symbol=symbol,
                        timestamp=timestamp,
                        timeframe=timeframe,
                        open=candle['open'],
                        high=candle['high'],
                        low=candle['low'],
                        close=candle['close'],
                        volume=candle['volume']
                    )
                    self.session.add(new_candle)
                    count += 1
            
            self.session.commit()
            return count
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error saving candles: {e}")
            return 0
    
    def get_candles(self, symbol: str, timeframe: str, limit: int = 500,
                  start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get candlestick data from database
        
        Parameters:
        symbol (str): Trading pair symbol
        timeframe (str): Timeframe (1m, 5m, 1h, 1D)
        limit (int): Number of candles to return
        start_time (datetime): Start time
        end_time (datetime): End time
        
        Returns:
        pd.DataFrame: Candlestick data
        """
        try:
            query = self.session.query(OHLCV).filter(
                OHLCV.symbol == symbol,
                OHLCV.timeframe == timeframe
            )
            
            if start_time:
                query = query.filter(OHLCV.timestamp >= start_time)
            if end_time:
                query = query.filter(OHLCV.timestamp <= end_time)
                
            query = query.order_by(OHLCV.timestamp.desc()).limit(limit)
            
            result = query.all()
            
            if not result:
                return pd.DataFrame()
                
            # Convert to DataFrame
            data = []
            for candle in result:
                data.append({
                    'timestamp': candle.timestamp,
                    'open': candle.open,
                    'high': candle.high,
                    'low': candle.low,
                    'close': candle.close,
                    'volume': candle.volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)  # Sort by timestamp
            
            return df
        except Exception as e:
            logger.error(f"Error getting candles: {e}")
            return pd.DataFrame()
    
    def save_trade(self, trade_data: Dict, is_backtest: bool = False) -> Optional[Trade]:
        """
        Save trade record to database
        
        Parameters:
        trade_data (Dict): Trade information
        is_backtest (bool): Whether this is a backtest trade
        
        Returns:
        Trade: Saved trade object, or None if error
        """
        try:
            # Extract trade data fields
            trade = Trade(
                order_id=trade_data.get('id'),
                symbol=trade_data['symbol'],
                timestamp=(
                    trade_data['timestamp'].to_pydatetime() if isinstance(trade_data.get('timestamp'), pd.Timestamp)
                    else datetime.fromtimestamp(trade_data['timestamp'] / 1000) if isinstance(trade_data.get('timestamp'), int)
                    else trade_data.get('timestamp', datetime.now())
                ),
                side=trade_data['side'],
                price=trade_data['price'],
                amount=trade_data['amount'],
                cost=trade_data.get('cost', trade_data['price'] * trade_data['amount']),
                fee=trade_data.get('fee', {}).get('cost') if isinstance(trade_data.get('fee'), dict) else trade_data.get('fee'),
                fee_coin=trade_data.get('fee', {}).get('currency') if isinstance(trade_data.get('fee'), dict) else None,
                position_size=trade_data.get('position_size'),
                realized_pnl=trade_data.get('realized_pnl'),
                stop_loss=trade_data.get('stop_loss'),
                take_profit=trade_data.get('take_profit'),
                is_backtest=is_backtest,
                strategy=trade_data.get('strategy')
            )
            
            self.session.add(trade)
            self.session.commit()
            
            logger.info(f"Trade saved: {trade}")
            return trade
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error saving trade: {e}")
            return None
    
    def get_trades(self, symbol: Optional[str] = None, limit: int = 100, 
                 is_backtest: bool = False) -> List[Dict]:
        """
        Get trade records from database
        
        Parameters:
        symbol (str, optional): Trading pair symbol, None for all symbols
        limit (int): Number of trades to return
        is_backtest (bool): Whether to get backtest trades
        
        Returns:
        List[Dict]: List of trade records
        """
        try:
            query = self.session.query(Trade).filter(Trade.is_backtest == is_backtest)
            
            if symbol:
                query = query.filter(Trade.symbol == symbol)
                
            query = query.order_by(Trade.timestamp.desc()).limit(limit)
            
            result = query.all()
            
            trades = []
            for trade in result:
                trades.append({
                    'id': trade.id,
                    'order_id': trade.order_id,
                    'symbol': trade.symbol,
                    'timestamp': trade.timestamp,
                    'side': trade.side,
                    'price': trade.price,
                    'amount': trade.amount,
                    'cost': trade.cost,
                    'fee': trade.fee,
                    'fee_coin': trade.fee_coin,
                    'position_size': trade.position_size,
                    'realized_pnl': trade.realized_pnl,
                    'stop_loss': trade.stop_loss,
                    'take_profit': trade.take_profit,
                    'strategy': trade.strategy
                })
            
            return trades
        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            return []
    
    def save_position(self, position_data: Dict, is_backtest: bool = False) -> Optional[Position]:
        """
        Save position to database
        
        Parameters:
        position_data (Dict): Position information
        is_backtest (bool): Whether this is a backtest position
        
        Returns:
        Position: Saved position object, or None if error
        """
        try:
            # Check if position already exists and is open
            existing = None
            if not is_backtest:
                existing = self.session.query(Position).filter(
                    Position.symbol == position_data['symbol'],
                    Position.side == position_data['side'],
                    Position.is_open == True,
                    Position.is_backtest == is_backtest
                ).first()
            
            if existing:
                # Update existing position
                existing.amount = position_data['amount']
                existing.entry_price = position_data['entry_price']
                existing.leverage = position_data.get('leverage', 1.0)
                existing.liquidation_price = position_data.get('liquidation_price')
                existing.stop_loss = position_data.get('stop_loss')
                existing.take_profit = position_data.get('take_profit')
                existing.unrealized_pnl = position_data.get('unrealized_pnl')
                
                if not position_data.get('is_open', True):
                    existing.is_open = False
                    closed_at = position_data.get('closed_at')
                    if isinstance(closed_at, pd.Timestamp):
                        existing.closed_at = closed_at.to_pydatetime()
                    elif isinstance(closed_at, int):
                        existing.closed_at = datetime.fromtimestamp(closed_at / 1000)
                    elif closed_at:
                        existing.closed_at = closed_at
                    else:
                        existing.closed_at = datetime.now()
                    existing.exit_price = position_data.get('exit_price')
                    existing.realized_pnl = position_data.get('realized_pnl')
                
                self.session.commit()
                return existing
            else:        
                # Convert timestamp safely
                ts = position_data.get('timestamp', int(datetime.now().timestamp() * 1000))
                if isinstance(ts, pd.Timestamp):
                    safe_timestamp = ts.to_pydatetime()
                elif isinstance(ts, int):
                    safe_timestamp = datetime.fromtimestamp(ts / 1000)
                else:
                    safe_timestamp = ts

                position = Position(
                    symbol=position_data['symbol'],
                    timestamp=safe_timestamp,
                    side=position_data['side'],
                    entry_price=position_data['entry_price'],
                    amount=position_data['amount'],
                    leverage=position_data.get('leverage', 1.0),
                    liquidation_price=position_data.get('liquidation_price'),
                    stop_loss=position_data.get('stop_loss'),
                    take_profit=position_data.get('take_profit'),
                    unrealized_pnl=position_data.get('unrealized_pnl'),
                    is_open=position_data.get('is_open', True),
                    is_backtest=is_backtest
                )
                
                if not position.is_open:
                    closed_at = position_data.get('closed_at')
                    if isinstance(closed_at, pd.Timestamp):
                        position.closed_at = closed_at.to_pydatetime()
                    elif isinstance(closed_at, int):
                        position.closed_at = datetime.fromtimestamp(closed_at / 1000)
                    elif closed_at:
                        position.closed_at = closed_at
                    else:
                        position.closed_at = datetime.now()
                    position.exit_price = position_data.get('exit_price')
                    position.realized_pnl = position_data.get('realized_pnl')
                
                self.session.add(position)
                self.session.commit()
                
                logger.info(f"Position saved: {position}")
                return position
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error saving position: {e}")
            return None
    
    def get_positions(self, symbol: Optional[str] = None, is_open: bool = True, 
                    is_backtest: bool = False) -> List[Dict]:
        """
        Get positions from database
        
        Parameters:
        symbol (str, optional): Trading pair symbol, None for all symbols
        is_open (bool): Whether to get open positions
        is_backtest (bool): Whether to get backtest positions
        
        Returns:
        List[Dict]: List of positions
        """
        try:
            query = self.session.query(Position).filter(
                Position.is_open == is_open,
                Position.is_backtest == is_backtest
            )
            
            if symbol:
                query = query.filter(Position.symbol == symbol)
                
            query = query.order_by(Position.timestamp.desc())
            
            result = query.all()
            
            positions = []
            for pos in result:
                positions.append({
                    'id': pos.id,
                    'symbol': pos.symbol,
                    'timestamp': pos.timestamp,
                    'side': pos.side,
                    'entry_price': pos.entry_price,
                    'amount': pos.amount,
                    'leverage': pos.leverage,
                    'liquidation_price': pos.liquidation_price,
                    'stop_loss': pos.stop_loss,
                    'take_profit': pos.take_profit,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'is_open': pos.is_open,
                    'closed_at': pos.closed_at,
                    'exit_price': pos.exit_price,
                    'realized_pnl': pos.realized_pnl
                })
            
            return positions
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def clear_backtest_data(self, symbol: Optional[str] = None):
        """
        Clear backtest data from database
        
        Parameters:
        symbol (str, optional): Trading pair symbol, None for all symbols
        """
        try:
            query = self.session.query(Trade).filter(Trade.is_backtest == True)
            if symbol:
                query = query.filter(Trade.symbol == symbol)
            query.delete()
            
            query = self.session.query(Position).filter(Position.is_backtest == True)
            if symbol:
                query = query.filter(Position.symbol == symbol)
            query.delete()
            
            self.session.commit()
            logger.info(f"Cleared backtest data for {symbol or 'all symbols'}")
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error clearing backtest data: {e}")
    
    def get_latest_candle(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """
        Get the latest candlestick for a symbol and timeframe
        
        Parameters:
        symbol (str): Trading pair symbol
        timeframe (str): Timeframe (1m, 5m, 1h, 1D)
        
        Returns:
        Dict: Latest candlestick data, or None if not found
        """
        try:
            candle = self.session.query(OHLCV).filter(
                OHLCV.symbol == symbol,
                OHLCV.timeframe == timeframe
            ).order_by(OHLCV.timestamp.desc()).first()
            
            if not candle:
                return None
                
            return {
                'timestamp': candle.timestamp,
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume
            }
        except Exception as e:
            logger.error(f"Error getting latest candle: {e}")
            return None
    
    def get_candle_count(self, symbol: str, timeframe: str) -> int:
        """
        Get the number of candles for a symbol and timeframe
        
        Parameters:
        symbol (str): Trading pair symbol
        timeframe (str): Timeframe (1m, 5m, 1h, 1D)
        
        Returns:
        int: Number of candles
        """
        try:
            count = self.session.query(OHLCV).filter(
                OHLCV.symbol == symbol,
                OHLCV.timeframe == timeframe
            ).count()
            
            return count
        except Exception as e:
            logger.error(f"Error getting candle count: {e}")
            return 0
    
    def get_missing_candle_ranges(self, symbol: str, timeframe: str, 
                                start_time: datetime, end_time: datetime) -> List[Tuple[datetime, datetime]]:
        """
        Get date ranges where candles are missing
        
        Parameters:
        symbol (str): Trading pair symbol
        timeframe (str): Timeframe (1m, 5m, 1h, 1D)
        start_time (datetime): Start time
        end_time (datetime): End time
        
        Returns:
        List[Tuple[datetime, datetime]]: List of (start, end) ranges with missing data
        """
        try:
            # Convert timeframe to timedelta
            if timeframe == '1m':
                delta = timedelta(minutes=1)
            elif timeframe == '5m':
                delta = timedelta(minutes=5)
            elif timeframe == '1h':
                delta = timedelta(hours=1)
            elif timeframe == '1D':
                delta = timedelta(days=1)
            else:
                logger.error(f"Unsupported timeframe: {timeframe}")
                return []
            
            # Get all timestamps in the range
            timestamps = self.session.query(OHLCV.timestamp).filter(
                OHLCV.symbol == symbol,
                OHLCV.timeframe == timeframe,
                OHLCV.timestamp >= start_time,
                OHLCV.timestamp <= end_time
            ).order_by(OHLCV.timestamp).all()
            
            timestamps = [t[0] for t in timestamps]
            
            if not timestamps:
                # No data at all, return the whole range
                return [(start_time, end_time)]
            
            # Find missing ranges
            missing_ranges = []
            current_time = start_time
            
            for timestamp in timestamps:
                if current_time + delta <= timestamp:
                    # Gap found
                    missing_ranges.append((current_time, timestamp - delta))
                current_time = timestamp + delta
            
            # Check if there's a gap at the end
            if current_time <= end_time:
                missing_ranges.append((current_time, end_time))
            
            return missing_ranges
        except Exception as e:
            logger.error(f"Error getting missing candle ranges: {e}")
            return []
        