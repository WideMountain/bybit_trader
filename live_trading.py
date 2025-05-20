import logging
import time
import threading
import schedule
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import argparse
import signal
import sys
import os

from trading_bot import TradingBot
import config

logger = logging.getLogger(__name__)

class LiveTrader:
    """Manager for live trading bots"""
    
    def __init__(self, symbols: Optional[List[str]] = None, timeframe: str = '1h',
               api_key: Optional[str] = None, api_secret: Optional[str] = None,
               testnet: Optional[bool] = None, check_interval: int = 60,
               strategy_class: Optional[type] = None):  
        """
        Initialize LiveTrader
        
        Parameters:
        symbols (List[str]): List of trading pair symbols
        timeframe (str): Timeframe (1m, 5m, 1h, 1D)
        api_key (str): Bybit API key
        api_secret (str): Bybit API secret
        testnet (bool): Use testnet if True, mainnet if False
        check_interval (int): Interval in seconds between checking for signals
        """
        self.symbols = symbols or config.TRADING_SYMBOLS
        self.timeframe = timeframe
        self.api_key = api_key or config.BYBIT_API_KEY
        self.api_secret = api_secret or config.BYBIT_API_SECRET
        self.testnet = testnet if testnet is not None else config.BYBIT_TESTNET
        self.check_interval = check_interval or config.CHECK_INTERVAL
        self.strategy_class = strategy_class
        
        # Initialize trading bots
        self.bots = {}
        for symbol in self.symbols:
            self.bots[symbol] = TradingBot(
                symbol=symbol,
                timeframe=self.timeframe,
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet,
                strategy_class=self.strategy_class  
            )
        
        # Internal state
        self.is_running = False
        self.stop_event = threading.Event()
        
        logger.info(f"LiveTrader initialized for {len(self.symbols)} symbols on {timeframe} timeframe")
    
    def start(self) -> None:
        """Start all trading bots"""
        if self.is_running:
            logger.warning("LiveTrader is already running")
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        logger.info("Starting all trading bots...")
        
        # Start each bot
        for symbol, bot in self.bots.items():
            bot.start(self.check_interval)
        
        logger.info(f"All bots started. Trading on: {', '.join(self.symbols)}")
    
    def stop(self) -> None:
        """Stop all trading bots"""
        if not self.is_running:
            logger.warning("LiveTrader is not running")
            return
        
        logger.info("Stopping all trading bots...")
        
        # Stop each bot
        for symbol, bot in self.bots.items():
            bot.stop()
        
        self.is_running = False
        self.stop_event.set()
        
        logger.info("All bots stopped")
    
    def get_status(self) -> Dict:
        """
        Get current status of all bots
        
        Returns:
        Dict: Status information for all bots
        """
        status = {
            'is_running': self.is_running,
            'timeframe': self.timeframe,
            'testnet': self.testnet,
            'check_interval': self.check_interval,
            'bots': {}
        }
        
        for symbol, bot in self.bots.items():
            status['bots'][symbol] = bot.get_status()
        
        return status
    
    def close(self) -> None:
        """Close all connections"""
        for symbol, bot in self.bots.items():
            bot.close()

# Signal handling for clean shutdown
def signal_handler(sig, frame):
    """Handle interrupt signals for clean shutdown"""
    logger.info("Interrupt received, shutting down...")
    
    if 'trader' in globals():
        trader.stop()
        trader.close()
    
    sys.exit(0)

def setup_logging():
    """Set up logging configuration"""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'live_trading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def main():
    """Main entry point for live trading"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Bybit Crypto Trading Bot')
    parser.add_argument('--symbols', nargs='+', default=config.TRADING_SYMBOLS,
                      help='Trading symbols (e.g., BTCUSDT ETHUSDT)')
    parser.add_argument('--timeframe', default=config.DEFAULT_TIMEFRAME,
                      help='Trading timeframe (1m, 5m, 1h, 1D)')
    parser.add_argument('--testnet', action='store_true', default=config.BYBIT_TESTNET,
                      help='Use testnet instead of mainnet')
    parser.add_argument('--interval', type=int, default=config.CHECK_INTERVAL,
                      help='Check interval in seconds')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    logger.info("Starting Bybit Crypto Trading Bot")
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start trader
    global trader
    trader = LiveTrader(
        symbols=args.symbols,
        timeframe=args.timeframe,
        testnet=args.testnet,
        check_interval=args.interval
    )
    
    trader.start()
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        trader.stop()
        trader.close()
        logger.info("Trading bot shut down")

if __name__ == "__main__":
    main()
