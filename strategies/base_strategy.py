from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from core.client_interface import SymbolInfo, TickData


@dataclass
class Signal:
    symbol: str
    signal_type: str  # "BUY", "SELL", "HOLD"
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    sl_points: float = 0.0
    reason: str = ""
    time: datetime = datetime.now(timezone.utc)


class BaseStrategy(ABC):
    """Abstract Strategy class providing technical indicators and signal generation contract."""

    def __init__(self, name: str, params: Optional[dict] = None):
        self.name = name
        self.params = params or {}

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, tick: TickData, spec: SymbolInfo) -> Signal:
        """Inspects candle dataframe + current tick and returns a trading Signal."""
        pass

    # --- Built-in Pure Vectorized Technical Indicators ---

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1 / period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / period, adjust=False).mean()
        rs = gain / (loss.replace(0, np.nan))
        rsi_series = 100 - (100 / (1 + rs))
        return rsi_series.fillna(50.0)

    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        mid = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = mid + (std * std_dev)
        lower = mid - (std * std_dev)
        return upper, mid, lower

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    @staticmethod
    def macd(
        series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - signal_line
        return macd_line, signal_line, hist

    @staticmethod
    def donchian_channel(high: pd.Series, low: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series]:
        upper = high.rolling(window=period).max()
        lower = low.rolling(window=period).min()
        return upper, lower
