import pytest
from datetime import datetime, timezone
import pandas as pd
from config import AppConfig, StrategyConfig, RiskConfig
from core.paper_client import PaperTradingClient
from strategies.base_strategy import BaseStrategy, Signal
from trading_engine import TradingEngine


class MockOscillatorStrategy(BaseStrategy):
    """Predictable mock strategy returning alternating signals."""

    def __init__(self, params=None):
        super().__init__("MockStrategy", params or {})
        self.current_signal = "BUY"

    def set_signal(self, sig_type: str):
        self.current_signal = sig_type

    def generate_signal(self, df, tick, spec):
        return Signal(
            symbol=tick.symbol,
            signal_type=self.current_signal,
            sl_price=tick.bid - 2.0 if self.current_signal == "BUY" else tick.ask + 2.0,
            tp_price=tick.bid + 4.0 if self.current_signal == "BUY" else tick.ask - 4.0,
            sl_points=200.0,
            reason=f"Mock {self.current_signal}",
            time=df.index[-2] if len(df) >= 2 else datetime.now(timezone.utc),
        )


def _build_test_df(n=50):
    dates = pd.date_range("2026-01-01", periods=n, freq="5min")
    prices = [4150.0] * n
    return pd.DataFrame({
        "open": prices,
        "high": [p + 1.0 for p in prices],
        "low": [p - 1.0 for p in prices],
        "close": prices,
        "tick_volume": [100] * n,
    }, index=dates)


def test_exit_and_reverse_buy_to_sell():
    config = AppConfig(
        mode="paper",
        active_symbol="XAUUSD",
        active_timeframe="M5",
        risk=RiskConfig(fixed_lot=0.01, use_dynamic_lot=False),
        strategy=StrategyConfig(exit_on_opposite=True, magic_number=777123, sl_dollars=10.0, tp_dollars=20.0),
    )
    engine = TradingEngine(config)
    mock_strat = MockOscillatorStrategy()
    engine.strategy = mock_strat

    # Initial prices and candle df
    df = _build_test_df(50)
    engine.client.get_rates = lambda sym, tf, count=350: df

    # Cycle 1: First BUY signal
    mock_strat.set_signal("BUY")
    engine._execute_cycle()

    positions = engine.client.get_open_positions()
    assert len(positions) == 1
    buy_pos = positions[0]
    assert buy_pos.type == "BUY"

    # Advance candle by 1 to simulate next candle close
    df2 = _build_test_df(51)
    engine.client.get_rates = lambda sym, tf, count=350: df2

    # Cycle 2: Same BUY signal -> must NOT duplicate
    mock_strat.set_signal("BUY")
    engine._execute_cycle()
    assert len(engine.client.get_open_positions()) == 1
    assert engine.client.get_open_positions()[0].ticket == buy_pos.ticket

    # Advance candle by 1 again
    df3 = _build_test_df(52)
    engine.client.get_rates = lambda sym, tf, count=350: df3

    # Cycle 3: Confirmed OPPOSITE SELL signal -> must EXIT BUY and ENTER SELL
    mock_strat.set_signal("SELL")
    engine._execute_cycle()

    new_positions = engine.client.get_open_positions()
    assert len(new_positions) == 1
    sell_pos = new_positions[0]
    assert sell_pos.type == "SELL"
    assert sell_pos.ticket != buy_pos.ticket

    # Check that previous BUY trade was closed with Exit & Reverse
    deals = engine.client.get_history_deals()
    assert len(deals) >= 1
    closed_buy_deal = [d for d in deals if d["ticket"] == buy_pos.ticket][0]
    assert closed_buy_deal["type"] == "BUY"
    assert closed_buy_deal["exit_reason"] == "Exit&Reverse"


def test_exit_and_reverse_disabled():
    config = AppConfig(
        mode="paper",
        active_symbol="XAUUSD",
        active_timeframe="M5",
        risk=RiskConfig(fixed_lot=0.01, use_dynamic_lot=False),
        strategy=StrategyConfig(exit_on_opposite=False, magic_number=777123),
    )
    engine = TradingEngine(config)
    mock_strat = MockOscillatorStrategy()
    engine.strategy = mock_strat

    df = _build_test_df(50)
    engine.client.get_rates = lambda sym, tf, count=350: df

    # Cycle 1: Open BUY
    mock_strat.set_signal("BUY")
    engine._execute_cycle()
    positions = engine.client.get_open_positions()
    assert len(positions) == 1
    buy_ticket = positions[0].ticket

    # Advance candle
    df2 = _build_test_df(51)
    engine.client.get_rates = lambda sym, tf, count=350: df2

    # Cycle 2: SELL signal arrives, but exit_on_opposite is False -> Should KEEP BUY position
    mock_strat.set_signal("SELL")
    engine._execute_cycle()

    positions_after = engine.client.get_open_positions()
    assert len(positions_after) == 1
    assert positions_after[0].ticket == buy_ticket
    assert positions_after[0].type == "BUY"
