import json
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict

CONFIG_FILE = Path(__file__).parent / "config.json"


class MT5Config(BaseModel):
    model_config = ConfigDict(extra="allow")
    login: Optional[int] = None
    password: Optional[str] = None
    server: Optional[str] = None
    path: Optional[str] = None  # Custom path to terminal64.exe if needed
    portable: bool = False
    timeout: int = 60000


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    max_risk_pct: float = Field(default=1.5, ge=0.1, le=10.0, description="Max % account balance risked per trade")
    fixed_lot: Optional[float] = Field(default=0.01, ge=0.01, le=50.0, description="Default lot if not dynamic")
    use_dynamic_lot: bool = Field(default=True, description="Calculate lot dynamically based on SL distance")
    max_open_positions: int = Field(default=5, ge=1, le=50, description="Maximum simultaneous open positions")
    max_daily_loss: float = Field(default=200.0, ge=1.0, description="Circuit breaker: stop all trades if daily loss reached ($)")
    max_spread_points: int = Field(default=35, ge=1, description="Max allowed spread in points before blocking trades")
    slippage_points: int = Field(default=10, ge=0, description="Allowed slippage in points")
    use_trailing_stop: bool = Field(default=True, description="Enable dynamic trailing stop loss")
    trailing_stop_dollars: float = Field(default=10.0, ge=0.01, description="Trailing stop distance in dollars")
    trailing_stop_pips: float = Field(default=20.0, ge=5.0, description="Trailing stop distance in pips")


class StrategyConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str = "EMA Crossover"
    symbol: str = "EURUSD"
    timeframe: str = "M5"  # M1, M5, M15, M30, H1, H4, D1
    ema_period: int = 44
    fast_period: int = 9
    slow_period: int = 21
    trend_filter_period: int = 200
    use_trend_filter: bool = False
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    bb_period: int = 20
    bb_std: float = 2.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    donchian_period: int = 20
    sl_dollars: float = 10.0
    tp_dollars: float = 20.0
    sl_pips: float = 1000.0
    tp_pips: float = 2000.0
    use_atr_stops: bool = False
    atr_sl_mult: float = 1.5
    atr_tp_mult: float = 2.5
    magic_number: int = 777123


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    mode: str = "paper"  # "live" or "paper"
    active_symbol: str = "EURUSD"
    active_timeframe: str = "M5"
    scan_interval_seconds: int = 3
    mt5: MT5Config = Field(default_factory=MT5Config)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    tracked_symbols: List[str] = [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "XAUUSD", "BTCUSD"
    ]


def load_config() -> AppConfig:
    """Loads configuration from config.json or returns default AppConfig."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return AppConfig(**data)
        except Exception:
            return AppConfig()
    return AppConfig()


def save_config(config: AppConfig) -> None:
    """Saves AppConfig to config.json."""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config.model_dump(), f, indent=4)
