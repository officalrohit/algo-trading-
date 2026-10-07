from strategies.base_strategy import BaseStrategy, Signal
from strategies.breakout_donchian import DonchianBreakoutStrategy
from strategies.ema_crossover import EMACrossoverStrategy
from strategies.macd_momentum import MACDMomentumStrategy
from strategies.price_cross_ema import PriceCrossEMAStrategy
from strategies.rsi_bollinger import RSIBollingerStrategy

class StrategyRegistry(dict):
    """Dictionary that supports case-insensitive, prefix, and dynamic naming lookup."""

    def __getitem__(self, key):
        if not key:
            return PriceCrossEMAStrategy
        if super().__contains__(key):
            return super().__getitem__(key)
        k_str = str(key).strip().lower()
        if "price cross" in k_str or "single ema" in k_str:
            return PriceCrossEMAStrategy
        if "crossover" in k_str:
            return EMACrossoverStrategy
        if "rsi" in k_str or "bollinger" in k_str:
            return RSIBollingerStrategy
        if "macd" in k_str:
            return MACDMomentumStrategy
        if "donchian" in k_str or "breakout" in k_str:
            return DonchianBreakoutStrategy
        return super().__getitem__(list(self.keys())[0])

    def get(self, key, default=None):
        if not key:
            return default or PriceCrossEMAStrategy
        try:
            return self[key]
        except (KeyError, IndexError):
            return default or PriceCrossEMAStrategy

    def __contains__(self, key):
        if super().__contains__(key):
            return True
        k_str = str(key).strip().lower()
        return any(
            frag in k_str
            for frag in ["price cross", "crossover", "rsi", "bollinger", "macd", "donchian", "breakout"]
        )


AVAILABLE_STRATEGIES = StrategyRegistry({
    "Price Cross EMA": PriceCrossEMAStrategy,
    "Price Cross EMA (44 EMA)": PriceCrossEMAStrategy,
    "EMA Crossover": EMACrossoverStrategy,
    "RSI + Bollinger Bands": RSIBollingerStrategy,
    "MACD Momentum": MACDMomentumStrategy,
    "Donchian Breakout": DonchianBreakoutStrategy,
})


def format_strategy_name(name: str, config) -> str:
    """Formats strategy name with current parameter values."""
    n_lower = str(name).lower()
    if "price cross" in n_lower:
        period = getattr(config.strategy, "ema_period", getattr(config.strategy, "fast_period", 44))
        return f"Price Cross EMA ({period} EMA)"
    elif "crossover" in n_lower:
        fast = getattr(config.strategy, "fast_period", 9)
        slow = getattr(config.strategy, "slow_period", 21)
        return f"EMA Crossover ({fast}/{slow} EMA)"
    return name


__all__ = [
    "BaseStrategy",
    "Signal",
    "EMACrossoverStrategy",
    "PriceCrossEMAStrategy",
    "RSIBollingerStrategy",
    "MACDMomentumStrategy",
    "DonchianBreakoutStrategy",
    "AVAILABLE_STRATEGIES",
    "StrategyRegistry",
    "format_strategy_name",
]
