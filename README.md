# Bybit Crypto Trading Bot

A modular, extensible crypto trading bot for Bybit, supporting live trading and backtesting with plug-and-play strategies.

---

## Features

- **Live Trading**: Trade crypto pairs on Bybit using your own API keys (testnet or mainnet).
- **Backtesting**: Simulate strategies on historical data to evaluate performance before going live.
- **Plug-and-Play Strategies**: Easily add or switch between strategies (e.g., MA Crossover, RSI) via a simple menu.
- **Risk Management**: Configure risk per trade, leverage, and initial balance.
- **Performance Metrics**: Detailed backtest results including net profit, win rate, drawdown, Sharpe ratio, and more.
- **Trade Logging**: All trades and positions are logged to a local SQLite database.
- **Extensible**: Add new strategies by creating a new file in the `strategies/` folder and updating the `STRATEGY_MAP` in `run.py`.
- **Configurable**: All settings (API keys, trading symbols, database, etc.) are managed via the `.env` file and `config.py`.

---

## Getting Started

### 1. Clone the Repository

```sh
git clone https://github.com/yourusername/bybit-trader.git
cd bybit-trader
```

### 2. Install Dependencies

```sh
pip install -r requirements.txt
```

### 3. Configure Environment

- Copy `.env.example` to `.env` and fill in your Bybit API credentials and settings:
  ```
  BYBIT_API_KEY=your_api_key
  BYBIT_API_SECRET=your_api_secret
  BYBIT_TESTNET=True  # Set to False for live trading
  DATABASE_URL=sqlite:///data/trading_database.db
  ```

### 4. Run the Bot

```sh
python run.py
```

---

## Usage

- **Select a strategy** at startup (e.g., MA Crossover, RSI).
- **Choose live trading or backtesting** from the main menu.
- **Configure trading parameters** (symbols, timeframe, risk, leverage, etc.).
- **View detailed results** after backtesting, including trade logs and performance metrics.
- **Plot equity curves** and analyze your strategy's performance.

---

## Adding New Strategies

1. **Create a new file** in the `strategies/` directory (e.g., `my_strategy.py`).
2. **Inherit from the base `Strategy` class** in `base_strategy.py`.
3. **Implement the `generate_signal` method**.
4. **Add your strategy to the `STRATEGY_MAP`** in `run.py`:
   ```python
   from strategies.my_strategy import MyStrategy
   STRATEGY_MAP = {
       "MA Crossover": MACrossoverStrategy,
       "RSI": RSIStrategy,
       "My Strategy": MyStrategy,
   }
   ```

---

## Example Strategies

- **MA Crossover**: Buys when short MA crosses above long MA, sells when it crosses below.
- **RSI**: Buys when RSI < 30, sells when RSI > 70.

---

## Security

- **Keep your `.env` file private** and never share your API keys.
- **Restrict your Bybit API key** to your IP address for added security.

---

## Disclaimer

This bot is for educational and research purposes only.  
**Trading cryptocurrencies involves significant risk. Use at your own risk.**

---

## License

MIT License
