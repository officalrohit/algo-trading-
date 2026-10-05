from datetime import datetime, timezone
import pytest
from config import RiskConfig
from core.client_interface import AccountInfo, Position, SymbolInfo, TickData
from core.risk_manager import RiskManager


@pytest.fixture
def risk_cfg():
    return RiskConfig(
        max_risk_pct=1.0,
        fixed_lot=0.05,
        use_dynamic_lot=True,
        max_open_positions=3,
        max_daily_loss=150.0,
        max_spread_points=30,
        use_trailing_stop=True,
        trailing_stop_pips=20.0,
    )


@pytest.fixture
def risk_mgr(risk_cfg):
    return RiskManager(risk_cfg)


@pytest.fixture
def eur_spec():
    return SymbolInfo(
        name="EURUSD",
        digits=5,
        point=0.00001,
        spread=12,
        min_lot=0.01,
        max_lot=50.0,
        lot_step=0.01,
    )


@pytest.fixture
def sample_account():
    return AccountInfo(
        login=12345,
        balance=10000.0,
        equity=10000.0,
        margin=0.0,
        free_margin=10000.0,
        profit=0.0,
        currency="USD",
        server="TestServer",
    )


def test_dynamic_lot_sizing(risk_mgr, eur_spec, sample_account):
    # Risk 1% of $10,000 = $100.
    # 250 points SL = 25 pips.
    # Standard forex: 1 point per lot = $1.00. 250 points * $1.00 = $250 per lot.
    # Lot = $100 / $250 = 0.40 lots.
    lot = risk_mgr.calculate_lot_size(sample_account, eur_spec, sl_points=250.0)
    assert lot == 0.40


def test_lot_size_clamping(risk_mgr, eur_spec, sample_account):
    # Tiny SL would cause huge lot size -> must clamp to max_lot
    huge_lot = risk_mgr.calculate_lot_size(sample_account, eur_spec, sl_points=0.01)
    assert huge_lot <= eur_spec.max_lot

    # Giant SL would cause tiny lot -> must clamp to min_lot
    tiny_lot = risk_mgr.calculate_lot_size(sample_account, eur_spec, sl_points=999999.0)
    assert tiny_lot >= eur_spec.min_lot


def test_circuit_breaker_trips(risk_mgr, sample_account):
    # Initial state
    tripped, _ = risk_mgr.is_circuit_breaker_active(sample_account)
    assert not tripped

    # Record small loss
    risk_mgr.record_deal_pnl(-50.0)
    tripped, _ = risk_mgr.is_circuit_breaker_active(sample_account)
    assert not tripped

    # Record loss exceeding limit (150.0)
    risk_mgr.record_deal_pnl(-110.0)
    tripped, reason = risk_mgr.is_circuit_breaker_active(sample_account)
    assert tripped
    assert "Daily loss reached" in reason

    # Manual reset
    risk_mgr.reset_circuit_breaker()
    tripped, _ = risk_mgr.is_circuit_breaker_active(sample_account)
    assert not tripped


def test_spread_filter(risk_mgr, eur_spec, sample_account):
    # Normal spread (15 points <= 30 max)
    normal_tick = TickData(
        symbol="EURUSD",
        bid=1.08500,
        ask=1.08515,
        last=1.08500,
        spread_points=15,
        time=datetime.now(timezone.utc),
    )
    can_trade, _ = risk_mgr.can_open_position(sample_account, eur_spec, normal_tick, current_open_positions_count=1)
    assert can_trade

    # Excessive spread (45 points > 30 max)
    wide_tick = TickData(
        symbol="EURUSD",
        bid=1.08500,
        ask=1.08545,
        last=1.08500,
        spread_points=45,
        time=datetime.now(timezone.utc),
    )
    can_trade, reason = risk_mgr.can_open_position(sample_account, eur_spec, wide_tick, current_open_positions_count=1)
    assert not can_trade
    assert "Spread too wide" in reason


def test_trailing_stop_calculation(risk_mgr, eur_spec):
    # BUY position opened at 1.08000 with SL 1.07800 (20 pips away)
    pos = Position(
        ticket=101,
        symbol="EURUSD",
        type="BUY",
        volume=0.1,
        price_open=1.08000,
        sl=1.07800,
        tp=1.08600,
        price_current=1.08000,
        profit=0.0,
        magic=123,
        comment="Test",
        time=datetime.now(timezone.utc),
    )

    # Price moves up to 1.08300 (+30 pips). Trailing distance is 20 pips (0.00200).
    # Proposed new SL = 1.08300 - 0.00200 = 1.08100 (which is in profit, lock in 10 pips!)
    tick = TickData(
        symbol="EURUSD",
        bid=1.08300,
        ask=1.08310,
        last=1.08300,
        spread_points=10,
        time=datetime.now(timezone.utc),
    )

    new_sl = risk_mgr.calculate_trailing_stop(pos, tick, eur_spec)
    assert new_sl is not None
    assert new_sl == 1.08100
    assert new_sl > pos.price_open
