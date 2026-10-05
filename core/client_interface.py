from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
import pandas as pd


@dataclass
class AccountInfo:
    login: int
    balance: float
    equity: float
    margin: float
    free_margin: float
    profit: float
    currency: str
    server: str
    leverage: int = 100


@dataclass
class TickData:
    symbol: str
    bid: float
    ask: float
    last: float
    spread_points: int
    time: datetime


@dataclass
class SymbolInfo:
    name: str
    digits: int
    point: float
    spread: int
    min_lot: float
    max_lot: float
    lot_step: float

    def pip_to_price(self, pips: float) -> float:
        """Converts pips to price distance based on symbol characteristics:
        - Gold (XAUUSD) & Silver (XAGUSD): 1 pip = 0.01 (so 100 pips = $1.00)
        - Forex 3/5 digits: 1 pip = 10 points (0.00010 on EURUSD, 0.010 on USDJPY)
        - Others: 1 point per pip
        """
        sym = self.name.upper()
        if "XAU" in sym or "XAG" in sym or "GOLD" in sym:
            return pips * self.point
        elif self.digits in (3, 5):
            return pips * self.point * 10.0
        else:
            return pips * self.point


@dataclass
class Position:
    ticket: int
    symbol: str
    type: str  # "BUY" or "SELL"
    volume: float
    price_open: float
    sl: float
    tp: float
    price_current: float
    profit: float
    magic: int
    comment: str
    time: datetime


@dataclass
class OrderResult:
    success: bool
    retcode: int
    order: int = 0
    deal: int = 0
    volume: float = 0.0
    price: float = 0.0
    comment: str = ""
    error_message: str = ""


class TradingClient(ABC):
    """Abstract interface for both Live MT5 Client and Paper Trading Simulator."""

    @abstractmethod
    def connect(self) -> bool:
        """Initialize connection to broker or simulation engine."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected."""
        pass

    @abstractmethod
    def get_account_info(self) -> Optional[AccountInfo]:
        """Fetch current account balances, equity, margin."""
        pass

    @abstractmethod
    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """Get market specifications for a symbol."""
        pass

    @abstractmethod
    def get_tick(self, symbol: str) -> Optional[TickData]:
        """Get current live bid/ask quote for a symbol."""
        pass

    @abstractmethod
    def get_rates(self, symbol: str, timeframe: str, count: int = 100) -> pd.DataFrame:
        """Fetch OHLCV candlestick data dataframe."""
        pass

    @abstractmethod
    def open_market_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        magic: int = 0,
        comment: str = ""
    ) -> OrderResult:
        """Open a new market BUY or SELL order."""
        pass

    @abstractmethod
    def close_position(self, ticket: int) -> OrderResult:
        """Close an open position by ticket."""
        pass

    @abstractmethod
    def modify_position(self, ticket: int, sl: Optional[float] = None, tp: Optional[float] = None) -> OrderResult:
        """Update SL and/or TP on an existing position."""
        pass

    @abstractmethod
    def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Retrieve list of currently open positions."""
        pass

    @abstractmethod
    def close_all_positions(self) -> List[OrderResult]:
        """Emergency Kill Switch: closes all open positions immediately."""
        pass

    @abstractmethod
    def get_history_deals(self, days: int = 7) -> List[dict]:
        """Fetch past closed deals / orders."""
        pass
