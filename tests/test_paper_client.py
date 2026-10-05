import pytest
from core.paper_client import PaperTradingClient


@pytest.fixture
def paper_client():
    client = PaperTradingClient(initial_balance=10000.0)
    client.connect()
    return client


def test_paper_client_connection(paper_client):
    assert paper_client.is_connected()
    acc = paper_client.get_account_info()
    assert acc is not None
    assert acc.balance == 10000.0
    assert acc.equity == 10000.0
    assert acc.free_margin == 10000.0


def test_open_and_close_buy_position(paper_client):
    # Open Buy Order
    res = paper_client.open_market_order(
        symbol="EURUSD",
        order_type="BUY",
        volume=0.1,
        sl=1.07000,
        tp=1.10000,
        magic=777,
        comment="TestBuy",
    )
    assert res.success
    assert res.order > 0

    # Verify Open Positions
    positions = paper_client.get_open_positions()
    assert len(positions) == 1
    pos = positions[0]
    assert pos.symbol == "EURUSD"
    assert pos.type == "BUY"
    assert pos.volume == 0.1
    assert pos.sl == 1.07000
    assert pos.tp == 1.10000

    # Modify SL
    mod_res = paper_client.modify_position(pos.ticket, sl=1.07500)
    assert mod_res.success
    assert paper_client.positions[pos.ticket].sl == 1.07500

    # Close Position
    close_res = paper_client.close_position(pos.ticket)
    assert close_res.success
    assert len(paper_client.get_open_positions()) == 0

    # Verify History Deal
    deals = paper_client.get_history_deals()
    assert len(deals) >= 1
    assert deals[0]["ticket"] == pos.ticket


def test_emergency_close_all(paper_client):
    paper_client.open_market_order("EURUSD", "BUY", 0.05)
    paper_client.open_market_order("GBPUSD", "SELL", 0.05)
    paper_client.open_market_order("USDJPY", "BUY", 0.05)
    assert len(paper_client.get_open_positions()) == 3

    results = paper_client.close_all_positions()
    assert len(results) == 3
    assert all(r.success for r in results)
    assert len(paper_client.get_open_positions()) == 0
