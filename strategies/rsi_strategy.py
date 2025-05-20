from .base_strategy import Strategy

class RSIStrategy(Strategy):
    def generate_signal(self, data, current_position=None):
        if data.empty or 'rsi' not in data.columns:
            return 0
        latest = data.iloc[-1]
        if latest['rsi'] < 30:
            return 1  # Buy
        elif latest['rsi'] > 70:
            return -1  # Sell
        return 0