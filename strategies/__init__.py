from strategies.base_strategy import BaseStrategy, Signal
from strategies.breakout_donchian import DonchianBreakoutStrategy
from strategies.ema_crossover import EMACrossoverStrategy
from strategies.macd_momentum import MACDMomentumStrategy
from strategies.price_cross_ema import PriceCrossEMAStrategy
from strategies.rsi_bollinger import RSIBollingerStrategy

AVAILABLE_STRATEGIES = {
    "EMA Crossover": EMACrossoverStrategy,
    "Price Cross EMA (44 EMA)": PriceCrossEMAStrategy,
    "RSI + Bollinger Bands": RSIBollingerStrategy,
    "MACD Momentum": MACDMomentumStrategy,
    "Donchian Breakout": DonchianBreakoutStrategy,
}

__all__ = [
    "BaseStrategy",
    "Signal",
    "EMACrossoverStrategy",
    "PriceCrossEMAStrategy",
    "RSIBollingerStrategy",
    "MACDMomentumStrategy",
    "DonchianBreakoutStrategy",
    "AVAILABLE_STRATEGIES",
]
