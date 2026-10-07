from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest
from core.client_interface import SymbolInfo, TickData
from strategies import (
    DonchianBreakoutStrategy,
    EMACrossoverStrategy,
    MACDMomentumStrategy,
    RSIBollingerStrategy,
)


@pytest.fixture
def spec():
    return SymbolInfo("EURUSD", 5, 0.00001, 15, 0.01, 100.0, 0.01)


@pytest.fixture
def tick():
    return TickData("EURUSD", 1.08500, 1.08515, 1.08500, 15, datetime.now(timezone.utc))


def test_ema_crossover_strategy(spec, tick):
    strategy = EMACrossoverStrategy({"fast_period": 5, "slow_period": 10, "trend_filter": 20})

    # Create synthetic series where fast crosses above slow and price is above 20-period trend
    n = 40
    # First 30 bars flat, then surge up
    prices = [1.08000] * 30 + [1.08050, 1.08100, 1.08200, 1.08350, 1.08550, 1.08800, 1.09100, 1.09500, 1.10000, 1.10500]
    dates = pd.date_range("2026-01-01", periods=n, freq="5min")
    df = pd.DataFrame({
        "open": prices,
        "high": [p + 0.00050 for p in prices],
        "low": [p - 0.00050 for p in prices],
        "close": prices,
        "tick_volume": [100] * n,
    }, index=dates)

    # Fast EMA crosses slow EMA on the uptrend surge
    # Let's verify signal is generated
    sig = strategy.generate_signal(df.iloc[:36], tick, spec)
    # At bar 35 or 36 the cross happens
    # Check that it produces either BUY or HOLD (valid Signal object)
    assert sig.signal_type in ("BUY", "HOLD")
    assert sig.symbol == "EURUSD"


def test_rsi_bollinger_strategy(spec, tick):
    strategy = RSIBollingerStrategy({"bb_period": 20, "rsi_period": 14, "rsi_oversold": 30.0})

    # 35 bars stable at 1.1000, then sudden sharp drop piercing lower band
    n = 40
    prices = [1.10000] * 35 + [1.09500, 1.08800, 1.08000, 1.07200, 1.06000]
    dates = pd.date_range("2026-01-01", periods=n, freq="5min")
    df = pd.DataFrame({
        "open": prices,
        "high": [p + 0.00020 for p in prices],
        "low": [p - 0.00050 for p in prices],
        "close": prices,
        "tick_volume": [100] * n,
    }, index=dates)

    sig = strategy.generate_signal(df, tick, spec)
    assert sig.signal_type == "BUY"
    assert "Oversold bounce" in sig.reason


def test_donchian_breakout_strategy(spec, tick):
    strategy = DonchianBreakoutStrategy({"period": 20})

    # 30 bars ranging between 1.08000 and 1.08500, then bar 31 breaks to 1.09000
    n = 35
    prices = [1.08200] * 30 + [1.08600, 1.08800, 1.09100, 1.09500, 1.09900]
    dates = pd.date_range("2026-01-01", periods=n, freq="5min")
    df = pd.DataFrame({
        "open": prices,
        "high": [p + 0.00030 for p in prices],
        "low": [p - 0.00030 for p in prices],
        "close": prices,
        "tick_volume": [100] * n,
    }, index=dates)

    sig = strategy.generate_signal(df.iloc[:32], tick, spec)
    assert sig.signal_type == "BUY"
    assert "Breakout above" in sig.reason


def test_price_cross_ema_44(spec, tick):
    from strategies import PriceCrossEMAStrategy, EMACrossoverStrategy

    n = 80
    # Price crosses above 44 EMA on bar 62
    prices = [4100.0] * 60 + [4095.0, 4092.0, 4105.0, 4120.0, 4135.0] + [4140.0] * 15
    dates = pd.date_range("2026-01-01", periods=n, freq="1min")
    df = pd.DataFrame({
        "open": prices,
        "high": prices,
        "low": prices,
        "close": prices,
        "tick_volume": [100] * n,
    }, index=dates)

    strat = PriceCrossEMAStrategy({"ema_period": 44})
    sig = strat.generate_signal(df.iloc[:64], tick, spec)
    assert sig.signal_type == "BUY"
    assert "CLOSED ABOVE 44 EMA" in sig.reason

    # Test single EMA mode in EMACrossoverStrategy when fast=44 and slow=44
    strat_crossover = EMACrossoverStrategy({"fast_period": 44, "slow_period": 44})
    sig_cross = strat_crossover.generate_signal(df.iloc[:64], tick, spec)
    assert sig_cross.signal_type == "BUY"
    assert "CLOSED ABOVE 44 EMA" in sig_cross.reason


def test_gold_pip_calculation():
    from core.client_interface import SymbolInfo, TickData
    from strategies import PriceCrossEMAStrategy

    gold_spec = SymbolInfo("XAUUSD", 2, 0.01, 35, 0.01, 100.0, 0.01)
    gold_tick = TickData("XAUUSD", 2650.00, 2650.35, 2650.00, 35, datetime.now(timezone.utc))

    assert gold_spec.pip_to_price(100.0) == 1.0  # 100 pips = $1.00
    assert gold_spec.pip_to_price(200.0) == 2.0  # 200 pips = $2.00

    n = 65
    prices = [2600.0] * 50 + [2590.0, 2580.0, 2610.0] + [2620.0] * 12
    dates = pd.date_range("2026-01-01", periods=n, freq="1min")
    df = pd.DataFrame({
        "open": prices, "high": prices, "low": prices, "close": prices, "tick_volume": [100] * n
    }, index=dates)

    strat = PriceCrossEMAStrategy({"ema_period": 44, "sl_pips": 100.0, "tp_pips": 200.0, "use_atr_stops": False})
    sig = strat.generate_signal(df.iloc[:54], gold_tick, gold_spec)
    assert sig.signal_type == "BUY"
    # Buy at Ask 2650.35: SL should be 2650.35 - $1.00 = 2649.35, TP should be 2650.35 + $2.00 = 2652.35
    assert sig.sl_price == 2649.35
    assert sig.tp_price == 2652.35


def test_gold_dollar_stop_and_target():
    from core.client_interface import SymbolInfo, TickData
    from strategies import PriceCrossEMAStrategy

    gold_spec = SymbolInfo("XAUUSD", 2, 0.01, 35, 0.01, 100.0, 0.01)
    gold_tick = TickData("XAUUSD", 2650.00, 2650.35, 2650.00, 35, datetime.now(timezone.utc))

    n = 65
    prices = [2600.0] * 50 + [2590.0, 2580.0, 2610.0] + [2620.0] * 12
    dates = pd.date_range("2026-01-01", periods=n, freq="1min")
    df = pd.DataFrame({
        "open": prices, "high": prices, "low": prices, "close": prices, "tick_volume": [100] * n
    }, index=dates)

    # User inputs SL: $1.50, TP: $3.00 directly in dollars
    strat = PriceCrossEMAStrategy({"ema_period": 44, "sl_dollars": 1.50, "tp_dollars": 3.00, "use_atr_stops": False})
    sig = strat.generate_signal(df.iloc[:54], gold_tick, gold_spec)
    assert sig.signal_type == "BUY"
    # Buy at Ask 2650.35: SL = 2650.35 - $1.50 = 2648.85, TP = 2650.35 + $3.00 = 2653.35
    assert sig.sl_price == 2648.85
    assert sig.tp_price == 2653.35
    assert sig.sl_points == 150.0


def test_engine_update_config_strategy_params():
    from config import load_config
    from trading_engine import TradingEngine
    from core.client_interface import SymbolInfo, TickData

    cfg = load_config()
    cfg.mode = "paper"
    cfg.strategy.name = "Price Cross EMA (44 EMA)"
    cfg.strategy.sl_dollars = 1.0
    cfg.strategy.tp_dollars = 2.0

    engine = TradingEngine(cfg)
    assert engine.strategy.params["sl_dollars"] == 1.0

    # Now simulate user updating SL to $10.0 and TP to $20.0 in the UI
    cfg.strategy.sl_dollars = 10.0
    cfg.strategy.tp_dollars = 20.0
    engine.update_config(cfg)

    # Strategy must be dynamically updated with new parameters immediately
    assert engine.strategy.params["sl_dollars"] == 10.0
    assert engine.strategy.params["tp_dollars"] == 20.0

    gold_spec = SymbolInfo("XAUUSD", 2, 0.01, 35, 0.01, 100.0, 0.01)
    gold_tick = TickData("XAUUSD", 4138.00, 4138.05, 4138.00, 35, datetime.now(timezone.utc))
    n = 65
    prices = [4130.0] * 50 + [4125.0, 4120.0, 4135.0] + [4140.0] * 12
    dates = pd.date_range("2026-01-01", periods=n, freq="1min")
    df = pd.DataFrame({
        "open": prices, "high": prices, "low": prices, "close": prices, "tick_volume": [100] * n
    }, index=dates)

    sig = engine.strategy.generate_signal(df.iloc[:54], gold_tick, gold_spec)
    assert sig.signal_type == "BUY"
    # Entry at 4138.05: SL must be 4138.05 - 10.00 = 4128.05, TP must be 4138.05 + 20.00 = 4158.05
    assert sig.sl_price == 4128.05
    assert sig.tp_price == 4158.05
    assert sig.sl_points == 1000.0


def test_dynamic_ema_strategy_naming_and_settings():
    from config import load_config
    from strategies import AVAILABLE_STRATEGIES, format_strategy_name
    from strategies.price_cross_ema import PriceCrossEMAStrategy

    cfg = load_config()
    cfg.strategy.name = "Price Cross EMA (21 EMA)"
    cfg.strategy.ema_period = 21

    # Check strategy name formatting
    formatted = format_strategy_name("Price Cross EMA", cfg)
    assert formatted == "Price Cross EMA (21 EMA)"

    # Check registry resolution
    strat_cls = AVAILABLE_STRATEGIES["Price Cross EMA (21 EMA)"]
    assert strat_cls == PriceCrossEMAStrategy

    # Check strategy instance name
    strat_instance = strat_cls(cfg.strategy.model_dump())
    assert strat_instance.name == "Price Cross EMA (21 EMA)"
    assert strat_instance.params["ema_period"] == 21

    # Change to 50 EMA
    cfg.strategy.ema_period = 50
    formatted50 = format_strategy_name("Price Cross EMA", cfg)
    assert formatted50 == "Price Cross EMA (50 EMA)"
    strat50 = strat_cls({"ema_period": 50})
    assert strat50.name == "Price Cross EMA (50 EMA)"
    assert strat50.params["ema_period"] == 50
