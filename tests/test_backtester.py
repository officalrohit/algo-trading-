from datetime import datetime
import numpy as np
import pandas as pd
import pytest
from backtester.engine import BacktestEngine
from core.client_interface import SymbolInfo
from strategies.ema_crossover import EMACrossoverStrategy


@pytest.fixture
def spec():
    return SymbolInfo("EURUSD", 5, 0.00001, 15, 0.01, 100.0, 0.01)


def test_backtest_engine_run(spec):
    # Generate 150 bars of synthetic trending price data
    np.random.seed(42)
    n = 150
    dates = pd.date_range("2026-01-01", periods=n, freq="5min")
    returns = np.random.normal(0.0001, 0.001, n)
    price_series = 1.08500 * np.cumprod(1 + returns)

    df = pd.DataFrame({
        "open": price_series,
        "high": price_series * 1.0005,
        "low": price_series * 0.9995,
        "close": price_series,
        "tick_volume": [250] * n,
    }, index=dates)

    strat = EMACrossoverStrategy({"fast_period": 5, "slow_period": 12, "trend_filter": 20})
    engine = BacktestEngine(
        strategy=strat,
        symbol_info=spec,
        initial_capital=10000.0,
        risk_pct=1.0,
    )

    result = engine.run(df)

    assert result.metrics is not None
    assert "Net Profit ($)" in result.metrics
    assert "Win Rate (%)" in result.metrics
    assert "Max Drawdown (%)" in result.metrics
    assert "Total Trades" in result.metrics
    assert not result.equity_curve.empty
    assert "equity" in result.equity_curve.columns
    assert "balance" in result.equity_curve.columns
