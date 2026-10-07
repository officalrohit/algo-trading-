from datetime import datetime, timedelta, timezone
import math
import random
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from core.client_interface import (
    AccountInfo,
    OrderResult,
    Position,
    SymbolInfo,
    TickData,
    TradingClient,
)

BASE_PRICES = {
    "EURUSD": 1.08500,
    "GBPUSD": 1.29500,
    "USDJPY": 152.300,
    "AUDUSD": 0.65500,
    "USDCAD": 1.38500,
    "USDCHF": 0.86500,
    "NZDUSD": 0.59500,
    "XAUUSD": 2680.50,
    "BTCUSD": 68500.0,
}

SYMBOL_SPECS = {
    "EURUSD": SymbolInfo("EURUSD", 5, 0.00001, 12, 0.01, 100.0, 0.01),
    "GBPUSD": SymbolInfo("GBPUSD", 5, 0.00001, 15, 0.01, 100.0, 0.01),
    "USDJPY": SymbolInfo("USDJPY", 3, 0.001, 14, 0.01, 100.0, 0.01),
    "AUDUSD": SymbolInfo("AUDUSD", 5, 0.00001, 16, 0.01, 100.0, 0.01),
    "USDCAD": SymbolInfo("USDCAD", 5, 0.00001, 18, 0.01, 100.0, 0.01),
    "USDCHF": SymbolInfo("USDCHF", 5, 0.00001, 15, 0.01, 100.0, 0.01),
    "NZDUSD": SymbolInfo("NZDUSD", 5, 0.00001, 20, 0.01, 100.0, 0.01),
    "XAUUSD": SymbolInfo("XAUUSD", 2, 0.01, 25, 0.01, 50.0, 0.01, contract_size=100.0),
    "BTCUSD": SymbolInfo("BTCUSD", 2, 0.01, 150, 0.01, 20.0, 0.01, contract_size=1.0),
}


class PaperTradingClient(TradingClient):
    """Realistic Paper Trading Simulator with dynamic market fluctuations,
    slippage, margin calculation, SL/TP triggers, and deal recording."""

    def __init__(self, initial_balance: float = 10000.0, leverage: int = 100):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.margin = 0.0
        self.free_margin = initial_balance
        self.leverage = leverage
        self._connected = False

        self.current_prices: Dict[str, float] = BASE_PRICES.copy()
        self.positions: Dict[int, Position] = {}
        self.history_deals: List[dict] = []
        self._next_ticket = 1000001
        self._rates_cache: Dict[str, pd.DataFrame] = {}

    def connect(self) -> bool:
        self._connected = True
        self._seed_historical_rates()
        return True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def _seed_historical_rates(self) -> None:
        """Generates realistic candlestick history for each symbol."""
        now = datetime.now(timezone.utc)
        for sym, price in self.current_prices.items():
            spec = SYMBOL_SPECS.get(sym, SymbolInfo(sym, 5, 0.00001, 15, 0.01, 100.0, 0.01))
            n_bars = 300
            times = [now - timedelta(minutes=5 * (n_bars - i)) for i in range(n_bars)]
            volatility = 0.0008 if spec.digits >= 4 else 0.0015

            # Random walk
            returns = np.random.normal(0, volatility, n_bars)
            price_path = price * np.cumprod(1 + returns)

            opens, highs, lows, closes, volumes = [], [], [], [], []
            for p in price_path:
                o = p
                c = o * (1 + np.random.normal(0, volatility * 0.5))
                h = max(o, c) * (1 + abs(np.random.normal(0, volatility * 0.3)))
                l = min(o, c) * (1 - abs(np.random.normal(0, volatility * 0.3)))
                v = int(random.randint(50, 1500))
                opens.append(round(o, spec.digits))
                highs.append(round(h, spec.digits))
                lows.append(round(l, spec.digits))
                closes.append(round(c, spec.digits))
                volumes.append(v)

            df = pd.DataFrame(
                {"open": opens, "high": highs, "low": lows, "close": closes, "tick_volume": volumes},
                index=pd.to_datetime(times),
            )
            self._rates_cache[sym] = df
            self.current_prices[sym] = closes[-1]

    def _update_market_prices(self) -> None:
        """Simulates realistic micro tick fluctuations and updates open positions."""
        for sym in self.current_prices.keys():
            spec = SYMBOL_SPECS.get(sym, SymbolInfo(sym, 5, 0.00001, 15, 0.01, 100.0, 0.01))
            drift = random.uniform(-0.0003, 0.0003)
            new_p = self.current_prices[sym] * (1.0 + drift)
            self.current_prices[sym] = round(new_p, spec.digits)

            # Update cache's latest close
            if sym in self._rates_cache and not self._rates_cache[sym].empty:
                self._rates_cache[sym].iloc[-1, self._rates_cache[sym].columns.get_loc("close")] = new_p

        # Check open positions for SL / TP and update floating PnL
        self._evaluate_positions()

    def _get_contract_size(self, symbol: str) -> float:
        if "XAU" in symbol:
            return 100.0
        elif "BTC" in symbol:
            return 1.0
        return 100000.0  # Standard forex lot

    def _evaluate_positions(self) -> None:
        """Calculates floating PnL, triggers SL/TP closures, updates margin and equity."""
        closed_tickets = []
        total_margin = 0.0
        total_profit = 0.0

        for ticket, pos in list(self.positions.items()):
            tick = self.get_tick(pos.symbol)
            if not tick:
                continue

            contract_size = self._get_contract_size(pos.symbol)
            margin_per_lot = (pos.price_open * contract_size) / self.leverage
            total_margin += margin_per_lot * pos.volume

            if pos.type == "BUY":
                pos.price_current = tick.bid
                pos.profit = (pos.price_current - pos.price_open) * pos.volume * contract_size
                # Check Stop Loss
                if pos.sl > 0 and tick.bid <= pos.sl:
                    self._execute_close(pos, pos.sl, "Stop Loss Hit")
                    closed_tickets.append(ticket)
                    continue
                # Check Take Profit
                if pos.tp > 0 and tick.bid >= pos.tp:
                    self._execute_close(pos, pos.tp, "Take Profit Hit")
                    closed_tickets.append(ticket)
                    continue
            else:  # SELL
                pos.price_current = tick.ask
                pos.profit = (pos.price_open - pos.price_current) * pos.volume * contract_size
                # Check Stop Loss
                if pos.sl > 0 and tick.ask >= pos.sl:
                    self._execute_close(pos, pos.sl, "Stop Loss Hit")
                    closed_tickets.append(ticket)
                    continue
                # Check Take Profit
                if pos.tp > 0 and tick.ask <= pos.tp:
                    self._execute_close(pos, pos.tp, "Take Profit Hit")
                    closed_tickets.append(ticket)
                    continue

            total_profit += pos.profit

        for t in closed_tickets:
            self.positions.pop(t, None)

        self.margin = round(total_margin, 2)
        self.equity = round(self.balance + total_profit, 2)
        self.free_margin = round(self.equity - self.margin, 2)

    def _execute_close(self, pos: Position, close_price: float, reason: str) -> None:
        contract_size = self._get_contract_size(pos.symbol)
        if pos.type == "BUY":
            profit = (close_price - pos.price_open) * pos.volume * contract_size
        else:
            profit = (pos.price_open - close_price) * pos.volume * contract_size

        self.balance = round(self.balance + profit, 2)
        open_time_str = pos.time.strftime("%Y-%m-%d %H:%M:%S") if isinstance(pos.time, datetime) else str(pos.time)
        close_time_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        deal = {
            "ticket": pos.ticket,
            "order": pos.ticket,
            "time": close_time_str,
            "symbol": pos.symbol,
            "type": pos.type,
            "volume": pos.volume,
            "open_time": open_time_str,
            "open_price": pos.price_open,
            "close_time": close_time_str,
            "close_price": round(close_price, 5),
            "sl": pos.sl,
            "tp": pos.tp,
            "profit": round(profit, 2),
            "commission": 0.0,
            "swap": 0.0,
            "exit_reason": reason,
            "comment": f"{pos.comment} [{reason}]",
        }
        self.history_deals.append(deal)

    def get_account_info(self) -> Optional[AccountInfo]:
        self._update_market_prices()
        return AccountInfo(
            login=9998881,
            balance=self.balance,
            equity=self.equity,
            margin=self.margin,
            free_margin=self.free_margin,
            profit=round(self.equity - self.balance, 2),
            currency="USD",
            server="Simulation-PaperBroker",
            leverage=self.leverage,
        )

    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        return SYMBOL_SPECS.get(
            symbol, SymbolInfo(symbol, 5, 0.00001, 15, 0.01, 100.0, 0.01)
        )

    def get_tick(self, symbol: str) -> Optional[TickData]:
        if symbol not in self.current_prices:
            self.current_prices[symbol] = 1.0000
        mid = self.current_prices[symbol]
        spec = self.get_symbol_info(symbol)
        spread_val = (spec.spread if spec else 15) * (spec.point if spec else 0.00001)

        bid = round(mid - spread_val / 2.0, spec.digits if spec else 5)
        ask = round(mid + spread_val / 2.0, spec.digits if spec else 5)

        return TickData(
            symbol=symbol,
            bid=bid,
            ask=ask,
            last=mid,
            spread_points=spec.spread if spec else 15,
            time=datetime.now(timezone.utc),
        )

    def get_rates(self, symbol: str, timeframe: str, count: int = 100) -> pd.DataFrame:
        self._update_market_prices()
        if symbol not in self._rates_cache:
            self._seed_historical_rates()
        df = self._rates_cache.get(symbol, pd.DataFrame())
        return df.tail(count).copy()

    def open_market_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        magic: int = 0,
        comment: str = "",
    ) -> OrderResult:
        self._update_market_prices()
        spec = self.get_symbol_info(symbol)
        if not spec:
            return OrderResult(success=False, retcode=-2, error_message=f"Symbol {symbol} invalid")

        # Clamp volume
        volume = max(spec.min_lot, min(round(volume, 2), spec.max_lot))

        tick = self.get_tick(symbol)
        if not tick:
            return OrderResult(success=False, retcode=-3, error_message="No quote")

        is_buy = order_type.upper() == "BUY"
        fill_price = tick.ask if is_buy else tick.bid

        # Margin check
        contract_size = self._get_contract_size(symbol)
        req_margin = (fill_price * contract_size * volume) / self.leverage
        if req_margin > self.free_margin:
            return OrderResult(
                success=False,
                retcode=10019,
                error_message=f"No money: required margin ${req_margin:.2f} > free margin ${self.free_margin:.2f}",
            )

        ticket = self._next_ticket
        self._next_ticket += 1

        pos = Position(
            ticket=ticket,
            symbol=symbol,
            type="BUY" if is_buy else "SELL",
            volume=volume,
            price_open=fill_price,
            sl=sl or 0.0,
            tp=tp or 0.0,
            price_current=fill_price,
            profit=0.0,
            magic=magic,
            comment=comment,
            time=datetime.now(timezone.utc),
        )
        self.positions[ticket] = pos
        self._evaluate_positions()

        return OrderResult(
            success=True,
            retcode=10009,  # TRADE_RETCODE_DONE
            order=ticket,
            deal=ticket,
            volume=volume,
            price=fill_price,
            comment="Order Executed",
        )

    def close_position(self, ticket: int, comment: str = "Manual Close") -> OrderResult:
        self._update_market_prices()
        pos = self.positions.get(ticket)
        if not pos:
            return OrderResult(success=False, retcode=-4, error_message=f"Position #{ticket} not found")

        tick = self.get_tick(pos.symbol)
        close_price = tick.bid if pos.type == "BUY" else tick.ask
        self._execute_close(pos, close_price, comment)
        self.positions.pop(ticket, None)
        self._evaluate_positions()

        return OrderResult(
            success=True,
            retcode=10009,
            order=ticket,
            deal=ticket,
            volume=pos.volume,
            price=close_price,
            comment=f"Closed ({comment})",
        )

    def modify_position(self, ticket: int, sl: Optional[float] = None, tp: Optional[float] = None) -> OrderResult:
        pos = self.positions.get(ticket)
        if not pos:
            return OrderResult(success=False, retcode=-4, error_message=f"Position #{ticket} not found")

        if sl is not None:
            pos.sl = float(sl)
        if tp is not None:
            pos.tp = float(tp)

        return OrderResult(success=True, retcode=10009, order=ticket, comment="SL/TP modified")

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        self._update_market_prices()
        if symbol:
            return [p for p in self.positions.values() if p.symbol == symbol]
        return list(self.positions.values())

    def close_all_positions(self) -> List[OrderResult]:
        self._update_market_prices()
        results = []
        for ticket in list(self.positions.keys()):
            res = self.close_position(ticket, comment="Kill Switch")
            results.append(res)
        return results

    def get_history_deals(self, days: int = 7) -> List[dict]:
        return list(reversed(self.history_deals))
