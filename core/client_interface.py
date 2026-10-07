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
    lot_step: float = 0.01
    contract_size: float = 100000.0

    def __post_init__(self):
        sym = self.name.upper()
        if ("XAU" in sym or "GOLD" in sym or "XAG" in sym) and self.contract_size == 100000.0:
            self.contract_size = 100.0
        elif "BTC" in sym and self.contract_size == 100000.0:
            self.contract_size = 1.0

    def get_contract_size(self) -> float:
        """Returns standard contract size for instrument (e.g. 100 for Gold, 1 for BTC, 100,000 for Forex)."""
        sym = self.name.upper()
        if hasattr(self, "contract_size") and self.contract_size not in (0, None, 100000.0):
            return float(self.contract_size)
        if "XAU" in sym or "GOLD" in sym or "XAG" in sym:
            return 100.0
        elif "BTC" in sym:
            return 1.0
        else:
            return 100000.0

    def cash_to_price_dist(self, cash_dollars: float, volume: float) -> float:
        """Converts desired cash profit/loss in dollars to price distance based on lot size and contract size.
        Formula: price_dist = cash_dollars / (volume * contract_size)
        Example: 0.05 lots on Gold (100 contract size) -> $10.00 SL / (0.05 * 100) = 2.00 price distance.
        """
        vol = max(0.0001, float(volume) if volume else 0.01)
        c_size = self.get_contract_size()
        multiplier = vol * c_size
        if multiplier <= 0:
            return float(cash_dollars)
        return float(cash_dollars) / multiplier

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
    def close_position(self, ticket: int, comment: str = "Manual Close") -> OrderResult:
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
