import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import pandas as pd

from core.client_interface import (
    AccountInfo,
    OrderResult,
    Position,
    SymbolInfo,
    TickData,
    TradingClient,
)

logger = logging.getLogger("MT5Client")

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False


TIMEFRAME_MAP = {}
if MT5_AVAILABLE:
    TIMEFRAME_MAP = {
        "M1": mt5.TIMEFRAME_M1,
        "M2": mt5.TIMEFRAME_M2,
        "M3": mt5.TIMEFRAME_M3,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }


class MT5Client(TradingClient):
    """Production MetaTrader 5 API client wrapper with automatic error handling,
    symbol verification, filling mode detection, and position management."""

    def __init__(
        self,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        path: Optional[str] = None,
        timeout: int = 60000,
        portable: bool = False,
    ):
        self.login = login
        self.password = password
        self.server = server
        self.path = path
        self.timeout = timeout
        self.portable = portable
        self._connected = False
        self._reconnecting = False
        self._last_reconnect_time = 0.0

    def connect(self) -> bool:
        if not MT5_AVAILABLE:
            logger.error("MetaTrader5 python module is not available.")
            return False

        init_kwargs = {"timeout": self.timeout}
        if self.path:
            init_kwargs["path"] = self.path
        if self.portable:
            init_kwargs["portable"] = True

        # Initialize connection to terminal
        if not mt5.initialize(**init_kwargs):
            err = mt5.last_error()
            logger.error(f"mt5.initialize() failed, error code: {err}")
            self._connected = False
            return False

        # Login if credentials provided
        if self.login and self.password and self.server:
            authorized = mt5.login(
                login=int(self.login),
                password=str(self.password),
                server=str(self.server),
                timeout=self.timeout,
            )
            if not authorized:
                err = mt5.last_error()
                logger.error(f"mt5.login() failed for account {self.login}, error code: {err}")
                self._connected = False
                return False

        self._connected = True
        info = mt5.account_info()
        logger.info(f"Connected to MT5 successfully. Account: {info.login if info else 'Unknown'}")
        return True

    def _ensure_connected(self) -> bool:
        if not MT5_AVAILABLE:
            return False
        if self._reconnecting:
            return self._connected

        if self._connected:
            try:
                term = mt5.terminal_info()
                if term is not None and term.connected:
                    return True
            except Exception:
                pass

        now = time.time()
        # Debounce reconnection attempts so we don't spam if terminal is closed
        if (now - self._last_reconnect_time) < 3.0:
            return self._connected

        self._last_reconnect_time = now
        try:
            self._reconnecting = True
            logger.info("MT5 connection lost or uninitialized. Auto-reconnecting...")
            return self.connect()
        finally:
            self._reconnecting = False

    def disconnect(self) -> None:
        if MT5_AVAILABLE and self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("MT5 disconnected.")

    def is_connected(self) -> bool:
        return self._ensure_connected()

    def get_account_info(self) -> Optional[AccountInfo]:
        if not self._ensure_connected():
            return None
        info = mt5.account_info()
        if info is None:
            return None
        return AccountInfo(
            login=info.login,
            balance=float(info.balance),
            equity=float(info.equity),
            margin=float(info.margin),
            free_margin=float(info.margin_free),
            profit=float(info.profit),
            currency=info.currency,
            server=info.server,
            leverage=info.leverage,
        )

    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        if not self._ensure_connected():
            return None

        # Ensure symbol is visible in Market Watch
        if not mt5.symbol_select(symbol, True):
            logger.warning(f"Could not select symbol {symbol} in Market Watch")

        info = mt5.symbol_info(symbol)
        if info is None:
            return None

        return SymbolInfo(
            name=info.name,
            digits=info.digits,
            point=info.point,
            spread=info.spread,
            min_lot=info.volume_min,
            max_lot=info.volume_max,
            lot_step=info.volume_step,
            contract_size=float(getattr(info, "trade_contract_size", 100000.0) or 100000.0),
        )

    def get_tick(self, symbol: str) -> Optional[TickData]:
        if not self._ensure_connected():
            return None
        # Ensure symbol is active in Market Watch
        mt5.symbol_select(symbol, True)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        sym_info = self.get_symbol_info(symbol)
        digits = sym_info.digits if sym_info else 5
        point = sym_info.point if sym_info else (10 ** -digits)
        spread_points = int(round((tick.ask - tick.bid) / point)) if point > 0 else 0

        return TickData(
            symbol=symbol,
            bid=tick.bid,
            ask=tick.ask,
            last=tick.last,
            spread_points=spread_points,
            time=datetime.fromtimestamp(tick.time, tz=timezone.utc),
        )

    def get_rates(self, symbol: str, timeframe: str, count: int = 300) -> pd.DataFrame:
        if not self._ensure_connected():
            return pd.DataFrame()

        mt5.symbol_select(symbol, True)
        tf = TIMEFRAME_MAP.get(timeframe.upper(), mt5.TIMEFRAME_M5)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        return df

    def _determine_filling_type(self, symbol: str) -> int:
        """Determines compatible execution filling mode for the broker symbol."""
        info = mt5.symbol_info(symbol)
        if info is None:
            return mt5.ORDER_FILLING_IOC
        fill_flags = info.filling_mode
        # In MT5: bit 1 (2) = IOC, bit 0 (1) = FOK, bit 2 (4) = RETURN
        if fill_flags & 2:
            return mt5.ORDER_FILLING_IOC
        elif fill_flags & 1:
            return mt5.ORDER_FILLING_FOK
        return mt5.ORDER_FILLING_RETURN

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
        if not self._ensure_connected():
            return OrderResult(success=False, retcode=-1, error_message="Not connected to MT5")

        sym_info = self.get_symbol_info(symbol)
        if not sym_info:
            return OrderResult(success=False, retcode=-2, error_message=f"Symbol {symbol} not found")

        # Clamp volume to lot limits
        step = sym_info.lot_step
        volume = round(round(volume / step) * step, 2)
        volume = max(sym_info.min_lot, min(volume, sym_info.max_lot))

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return OrderResult(success=False, retcode=-3, error_message=f"No quote tick for {symbol}")

        is_buy = order_type.upper() == "BUY"
        mt5_order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
        price = tick.ask if is_buy else tick.bid

        digits = sym_info.digits
        sl_val = round(sl, digits) if sl is not None and sl > 0 else 0.0
        tp_val = round(tp, digits) if tp is not None and tp > 0 else 0.0

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": mt5_order_type,
            "price": float(price),
            "sl": float(sl_val),
            "tp": float(tp_val),
            "deviation": 20,
            "magic": int(magic),
            "comment": comment[:31],  # MT5 comment limit
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._determine_filling_type(symbol),
        }

        result = mt5.order_send(request)
        if result is None:
            err = mt5.last_error()
            return OrderResult(success=False, retcode=err[0], error_message=f"order_send failed: {err[1]}")

        success = result.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED)
        return OrderResult(
            success=success,
            retcode=result.retcode,
            order=result.order,
            deal=result.deal,
            volume=result.volume,
            price=result.price,
            comment=result.comment,
            error_message="" if success else f"Return code {result.retcode}: {result.comment}",
        )

    def close_position(self, ticket: int, comment: str = "Manual Close") -> OrderResult:
        if not self._ensure_connected():
            return OrderResult(success=False, retcode=-1, error_message="Not connected")

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return OrderResult(success=False, retcode=-4, error_message=f"Position #{ticket} not found")

        pos = positions[0]
        is_buy = pos.type == mt5.ORDER_TYPE_BUY
        opposite_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY

        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            return OrderResult(success=False, retcode=-3, error_message=f"No tick for {pos.symbol}")

        price = tick.bid if is_buy else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": opposite_type,
            "price": price,
            "deviation": 20,
            "magic": pos.magic,
            "comment": comment[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._determine_filling_type(pos.symbol),
        }

        result = mt5.order_send(request)
        if result is None:
            err = mt5.last_error()
            return OrderResult(success=False, retcode=err[0], error_message=f"close failed: {err[1]}")

        success = result.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED)
        return OrderResult(
            success=success,
            retcode=result.retcode,
            order=result.order,
            deal=result.deal,
            volume=result.volume,
            price=result.price,
            comment=result.comment,
            error_message="" if success else f"Return code {result.retcode}: {result.comment}",
        )

    def modify_position(self, ticket: int, sl: Optional[float] = None, tp: Optional[float] = None) -> OrderResult:
        if not self._ensure_connected():
            return OrderResult(success=False, retcode=-1, error_message="Not connected")

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return OrderResult(success=False, retcode=-4, error_message=f"Position #{ticket} not found")

        pos = positions[0]
        sym_info = self.get_symbol_info(pos.symbol)
        digits = sym_info.digits if sym_info else 5

        new_sl = round(sl, digits) if sl is not None else pos.sl
        new_tp = round(tp, digits) if tp is not None else pos.tp

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": pos.symbol,
            "sl": float(new_sl),
            "tp": float(new_tp),
        }

        result = mt5.order_send(request)
        if result is None:
            err = mt5.last_error()
            return OrderResult(success=False, retcode=err[0], error_message=f"modify failed: {err[1]}")

        success = result.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED)
        return OrderResult(
            success=success,
            retcode=result.retcode,
            order=result.order,
            comment=result.comment,
            error_message="" if success else f"Modify error {result.retcode}: {result.comment}",
        )

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        if not self._ensure_connected():
            return []

        raw_positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if raw_positions is None:
            return []

        results = []
        for p in raw_positions:
            results.append(
                Position(
                    ticket=p.ticket,
                    symbol=p.symbol,
                    type="BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
                    volume=p.volume,
                    price_open=p.price_open,
                    sl=p.sl,
                    tp=p.tp,
                    price_current=p.price_current,
                    profit=p.profit,
                    magic=p.magic,
                    comment=p.comment,
                    time=datetime.fromtimestamp(p.time, tz=timezone.utc),
                )
            )
        return results

    def close_all_positions(self) -> List[OrderResult]:
        positions = self.get_open_positions()
        results = []
        for pos in positions:
            res = self.close_position(pos.ticket, comment="Kill Switch")
            results.append(res)
        return results

    def get_history_deals(self, days: int = 7) -> List[dict]:
        if not self._ensure_connected():
            return []
        now = datetime.now(timezone.utc)
        from_time = now - timedelta(days=days)
        deals = mt5.history_deals_get(from_time, now)
        if deals is None:
            return []
        orders = mt5.history_orders_get(from_time, now)
        orders_by_pos = {}
        if orders:
            for o in orders:
                pid = o.position_id or o.ticket
                if pid not in orders_by_pos:
                    orders_by_pos[pid] = []
                orders_by_pos[pid].append(o)

        pos_deals = {}
        for d in deals:
            # Skip non-trade deals (such as initial deposit or account balance adjustments)
            if d.entry not in (0, 1, 2, 3) or not d.symbol:
                continue
            pid = d.position_id
            if not pid:
                continue
            if pid not in pos_deals:
                pos_deals[pid] = []
            pos_deals[pid].append(d)

        out = []
        for pid, dlist in pos_deals.items():
            in_deals = [d for d in dlist if d.entry == 0]
            out_deals = [d for d in dlist if d.entry in (1, 2, 3)]
            if not out_deals:
                continue  # Position is still open or has no exit yet

            first_in = in_deals[0] if in_deals else dlist[0]
            last_out = out_deals[-1]
            total_profit = sum(d.profit for d in dlist)
            total_commission = sum(d.commission for d in dlist)
            total_swap = sum(d.swap for d in dlist)

            pos_orders = orders_by_pos.get(pid, [])
            in_orders = [o for o in pos_orders if o.type in (0, 1) and (o.sl > 0 or o.tp > 0)]
            sl_val = in_orders[0].sl if in_orders else 0.0
            tp_val = in_orders[0].tp if in_orders else 0.0

            # Determine human-friendly exit reason
            reason_str = "Closed"
            if last_out.reason == 4 or "[sl" in (last_out.comment or "").lower():
                reason_str = "Stop Loss Hit"
            elif last_out.reason == 5 or "[tp" in (last_out.comment or "").lower():
                reason_str = "Take Profit Hit"
            elif last_out.reason == 6 or "so" in (last_out.comment or "").lower():
                reason_str = "Stop Out"
            elif last_out.comment:
                reason_str = last_out.comment

            out.append({
                "ticket": pid,
                "symbol": first_in.symbol,
                "type": "BUY" if first_in.type == 0 else "SELL",
                "volume": first_in.volume,
                "open_time": datetime.fromtimestamp(first_in.time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "open_price": first_in.price,
                "close_time": datetime.fromtimestamp(last_out.time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "close_price": last_out.price,
                "sl": sl_val,
                "tp": tp_val,
                "profit": round(total_profit, 2),
                "commission": round(total_commission, 2),
                "swap": round(total_swap, 2),
                "exit_reason": reason_str,
            })

        out.sort(key=lambda x: x["close_time"], reverse=True)
        return out
