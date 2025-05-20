from typing import Any, Dict
import pandas as pd

class Strategy:
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {}

    def generate_signal(self, data: pd.DataFrame, current_position: Dict = None) -> int:
        """
        Return 1 (buy/long), -1 (sell/short), or 0 (hold)
        """
        raise NotImplementedError("generate_signal must be implemented by subclasses")
    