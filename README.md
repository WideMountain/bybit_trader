# Bybit-Trader

A modular Python trading bot and backtesting framework for Bybit.  
Easily customize strategies, analyze performance, and run live or paper trading.

---

## Features

- Automated trading on Bybit (testnet or live)
- Backtesting with detailed performance metrics
- Modular strategy and risk management
- SQLite database for trades and positions
- Performance reports and equity curve plotting

---

## Quick Start

### 1. Clone the repository

```sh
git clone https://github.com/yourusername/bybit-trader.git
cd bybit-trader
```

### 2. Install dependencies

```sh
pip install -r requirements.txt
```

### 3. Configure your environment

- Copy `.env.example` into a `.env` file of the 'bybit-trader' folder.
- Fill in your Bybit API credentials and settings:


`env.example`
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret
BYBIT_TESTNET=True  # Set to False for live trading
DATABASE_URL=sqlite:///data/trading_database.db


### 4. Start the Dashboard

To launch the interactive dashboard for managing and monitoring your bot:

```sh
python run.py
```

This will start the dashboard interface where you can run backtests, start/stop the trading bot, and view performance metrics.

---

## Notes

- **Never share your real API keys publicly.**
- For live trading, set `BYBIT_TESTNET=False` in your `.env` file.
- Review and customize strategies in the `strategies/` folder.

