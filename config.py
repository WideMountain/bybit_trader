import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API credentials
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
BYBIT_TESTNET = os.getenv("BYBIT_TESTNET", "True").lower() in ("true", "1", "t")


# Trading parameters
TRADING_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DEFAULT_TIMEFRAME = "1h"
CHECK_INTERVAL = 60  # seconds
RISK_PER_TRADE_PCT = 1.0  # Percentage of account to risk per trade
DEFAULT_LEVERAGE = 3  # Default leverage
MAX_LEVERAGE = 20  # Maximum allowed leverage
MAX_REQUESTS_PER_MINUTE = 60  # Or set to the correct value for Bybit's API limits
REQUEST_WEIGHT_RESET = 60 # Time in seconds to reset request weight

# Available timeframes
AVAILABLE_TIMEFRAMES = {
    "1m": "1 minute",
    "5m": "5 minutes",
    "15m": "15 minutes",
    "1h": "1 hour",
    "4h": "4 hours",
    "1D": "1 day"
}

# Strategy parameters
DEFAULT_STRATEGY_PARAMS = {
    "short_window": 20,
    "long_window": 50,
    "atr_period": 14,
    "atr_multiplier": 2.0
}

# Symbol-specific strategy parameters
STRATEGY_PARAMS = {
    "BTCUSDT": {
        "short_window": 20,
        "long_window": 50,
        "atr_period": 14,
        "atr_multiplier": 2.0
    },
    "ETHUSDT": {
        "short_window": 15,
        "long_window": 40,
        "atr_period": 14,
        "atr_multiplier": 2.0
    },
    "SOLUSDT": {
        "short_window": 10,
        "long_window": 30,
        "atr_period": 14,
        "atr_multiplier": 2.5
    }
}

# Database settings
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///trading_bot.db")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create logger
logger = logging.getLogger(__name__)
