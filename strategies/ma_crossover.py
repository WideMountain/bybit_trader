from .base_strategy import Strategy

class MACrossoverStrategy(Strategy):
    def generate_signal(self, data, current_position=None):
        if data.empty:
            return 0
        latest = data.iloc[-1]
        if latest['sma_short'] > latest['sma_long']:
            if not current_position or current_position['side'] == 'short':
                return 1
        elif latest['sma_short'] < latest['sma_long']:
            if not current_position or current_position['side'] == 'long':
                return -1
        return 0