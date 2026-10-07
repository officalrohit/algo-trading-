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


def test_backtest_gold_fixed_qty_dollar_stops():
    from strategies.price_cross_ema import PriceCrossEMAStrategy

    gold_spec = SymbolInfo("XAUUSD", 2, 0.01, 35, 0.01, 100.0, 0.01)
    n = 120
    dates = pd.date_range("2026-01-01", periods=n, freq="15min")
    # Base prices around 4130, with clear crossovers
    prices = [4130.0] * 50 + [4120.0, 4115.0, 4140.0] + [4155.0] * 67
    highs = [p + 2.0 for p in prices]
    lows = [p - 2.0 for p in prices]

    df = pd.DataFrame({
        "open": prices,
        "high": highs,
        "low": lows,
        "close": prices,
        "tick_volume": [100] * n,
    }, index=dates)

    strat = PriceCrossEMAStrategy({
        "ema_period": 44,
        "sl_dollars": 10.0,
        "tp_dollars": 20.0,
        "use_atr_stops": False,
    })

    bt = BacktestEngine(
        strategy=strat,
        symbol_info=gold_spec,
        initial_capital=10000.0,
        fixed_lot=0.01,
    )
    result = bt.run(df)

    assert len(result.trades) > 0
    # Every trade must have the configured fixed quantity (0.01 lots)
    for trade in result.trades:
        assert trade.volume == 0.01
        if trade.exit_reason == "Stop Loss":
            assert abs(trade.pnl - (-10.0)) < 1.0
        elif trade.exit_reason == "Take Profit":
            assert abs(trade.pnl - 20.0) < 1.0


def test_backtest_cash_dollar_sl_tp_with_custom_lot():
    """Verify that setting $10 SL and $20 Target with 0.05 lots yields exact $10 loss and $20 profit (not $50 and $100)."""
    from strategies.price_cross_ema import PriceCrossEMAStrategy

    gold_spec = SymbolInfo("XAUUSD", 2, 0.01, 35, 0.01, 100.0, 0.01)
    n = 120
    dates = pd.date_range("2026-01-01", periods=n, freq="15min")
    # Base prices around 4130, with clear crossovers
    prices = [4130.0] * 50 + [4120.0, 4115.0, 4140.0] + [4155.0] * 67
    highs = [p + 5.0 for p in prices]
    lows = [p - 5.0 for p in prices]

    df = pd.DataFrame({
        "open": prices,
        "high": highs,
        "low": lows,
        "close": prices,
        "tick_volume": [100] * n,
    }, index=dates)

    qty = 0.05
    cash_sl = 10.0
    cash_tp = 20.0
    sl_dist = gold_spec.cash_to_price_dist(cash_sl, qty)
    tp_dist = gold_spec.cash_to_price_dist(cash_tp, qty)

    # 10 / (0.05 * 100) = 2.0 price move
    assert abs(sl_dist - 2.0) < 1e-4
    # 20 / (0.05 * 100) = 4.0 price move
    assert abs(tp_dist - 4.0) < 1e-4

    strat = PriceCrossEMAStrategy({
        "ema_period": 44,
        "sl_dollars": sl_dist,
        "tp_dollars": tp_dist,
        "use_atr_stops": False,
    })

    bt = BacktestEngine(
        strategy=strat,
        symbol_info=gold_spec,
        initial_capital=10000.0,
        fixed_lot=qty,
    )
    result = bt.run(df)

    assert len(result.trades) > 0
    for trade in result.trades:
        assert trade.volume == 0.05
        if trade.exit_reason == "Stop Loss":
            assert abs(trade.pnl - (-10.0)) < 1.0
        elif trade.exit_reason == "Take Profit":
            assert abs(trade.pnl - 20.0) < 1.0
