from collections import deque
from datetime import datetime, timezone
import logging
import threading
import time
from typing import Deque, Dict, List, Optional

from config import AppConfig
from core.client_interface import AccountInfo, OrderResult, Position, SymbolInfo, TickData, TradingClient
from core.mt5_client import MT5Client
from core.paper_client import PaperTradingClient
from core.risk_manager import RiskManager
from strategies.base_strategy import BaseStrategy, Signal
from strategies import AVAILABLE_STRATEGIES

logger = logging.getLogger("TradingEngine")


class TradingEngine:
    """Thread-safe core trading orchestrator connecting market data,
    active quantitative strategies, risk management, and order execution."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.risk_manager = RiskManager(config.risk)
        self.client: TradingClient = self._create_client()
        self.strategy: BaseStrategy = self._create_strategy()

        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Thread-safe activity and diagnostic log buffer
        self.activity_log: Deque[dict] = deque(maxlen=200)
        self.last_signal: Optional[Signal] = None
        self.last_scan_time: Optional[datetime] = None

        self._log("Trading Engine initialized", "INFO")

    def _create_client(self) -> TradingClient:
        if self.config.mode.lower() == "live":
            mt5_cfg = self.config.mt5
            return MT5Client(
                login=mt5_cfg.login,
                password=mt5_cfg.password,
                server=mt5_cfg.server,
                path=mt5_cfg.path,
                timeout=mt5_cfg.timeout,
                portable=mt5_cfg.portable,
            )
        else:
            return PaperTradingClient(initial_balance=10000.0)

    def _create_strategy(self) -> BaseStrategy:
        strat_cls = AVAILABLE_STRATEGIES.get(self.config.strategy.name, list(AVAILABLE_STRATEGIES.values())[0])
        return strat_cls(self.config.strategy.model_dump())

    def update_config(self, new_config: AppConfig) -> None:
        with self._lock:
            is_live_client = isinstance(self.client, MT5Client)
            target_is_live = new_config.mode.lower() == "live"
            mode_changed = is_live_client != target_is_live
            mt5_changed = self.config.mt5.model_dump() != new_config.mt5.model_dump()

            self.config = new_config
            self.risk_manager.update_config(new_config.risk)

            # ALWAYS recreate strategy with updated parameters so SL/TP take effect immediately
            self.strategy = self._create_strategy()
            sl_val = getattr(new_config.strategy, "sl_dollars", getattr(new_config.strategy, "sl_pips", "N/A"))
            tp_val = getattr(new_config.strategy, "tp_dollars", getattr(new_config.strategy, "tp_pips", "N/A"))
            rev_val = "Enabled" if getattr(new_config.strategy, "exit_on_opposite", True) else "Disabled"
            self._log(f"Strategy updated: {self.config.strategy.name} (SL: ${sl_val}, TP: ${tp_val}, Reversal: {rev_val})", "INFO")

            if mode_changed or mt5_changed:
                was_running = self.is_running
                if was_running:
                    self.stop()
                self.client.disconnect()
                self.client = self._create_client()
                self.client.connect()
                if was_running:
                    self.start()
                self._log(f"Client reconnected for {new_config.mode.upper()} mode", "INFO")

    def _log(self, message: str, level: str = "INFO") -> None:
        entry = {
            "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "level": level,
            "message": message,
        }
        self.activity_log.append(entry)
        if level == "ERROR":
            logger.error(message)
        elif level == "WARNING":
            logger.warning(message)
        else:
            logger.info(message)

    def start(self) -> bool:
        with self._lock:
            if self.is_running:
                return True

            if not self.client.is_connected():
                ok = self.client.connect()
                if not ok:
                    self._log(f"Failed to connect {self.config.mode.upper()} client", "ERROR")
                    return False

            self.is_running = True
            self._thread = threading.Thread(target=self._run_loop, daemon=True, name="AlgoWorker")
            self._thread.start()
            self._log(f"Algorithmic Trading Bot STARTED on {self.config.active_symbol} ({self.config.active_timeframe})", "INFO")
            return True

    def stop(self) -> None:
        with self._lock:
            if not self.is_running:
                return
            self.is_running = False
            self._log("Algorithmic Trading Bot STOPPED", "WARNING")

    def _run_loop(self) -> None:
        while self.is_running:
            try:
                self._execute_cycle()
            except Exception as e:
                self._log(f"Engine cycle error: {str(e)}", "ERROR")

            time.sleep(max(1, self.config.scan_interval_seconds))

    def _execute_cycle(self) -> None:
        symbol = self.config.active_symbol
        tf = self.config.active_timeframe
        self.last_scan_time = datetime.now(timezone.utc)

        # 1. Fetch Account Info
        account = self.client.get_account_info()
        if not account:
            return

        # 2. Check Circuit Breaker
        tripped, reason = self.risk_manager.is_circuit_breaker_active(account)
        if tripped:
            self._log(f"Trading halted: {reason}", "ERROR")
            self.stop()
            return

        # 3. Fetch Symbol Info and Tick
        sym_info = self.client.get_symbol_info(symbol)
        tick = self.client.get_tick(symbol)
        if not sym_info or not tick:
            return

        # 4. Trailing Stop Loss Management on Open Positions
        positions = self.client.get_open_positions()
        for pos in positions:
            if pos.symbol == symbol and self.config.risk.use_trailing_stop:
                new_sl = self.risk_manager.calculate_trailing_stop(pos, tick, sym_info)
                if new_sl is not None:
                    res = self.client.modify_position(pos.ticket, sl=new_sl)
                    if res.success:
                        self._log(f"Trailing SL updated for #{pos.ticket} -> {new_sl}", "INFO")

        # 5. Fetch Candle Rates and Evaluate Strategy Signal
        df = self.client.get_rates(symbol, tf, count=350)
        if df.empty or len(df) < 30:
            return

        sig = self.strategy.generate_signal(df, tick, sym_info)
        self.last_signal = sig

        if sig.signal_type in ("BUY", "SELL"):
            # Ensure we only enter once per confirmed closed candle
            closed_candle_time = df.index[-2]
            if hasattr(self, "_last_traded_candle") and self._last_traded_candle == closed_candle_time:
                return

            # Check existing positions for this symbol and strategy magic
            existing = [p for p in positions if p.symbol == symbol and p.magic == self.config.strategy.magic_number]
            same_positions = [p for p in existing if p.type == sig.signal_type]
            opp_positions = [p for p in existing if p.type != sig.signal_type]

            # 1. Prevent duplicate stacking in the same direction
            if same_positions:
                return

            # 2. Exit & Reverse on opposite signal
            exit_on_opp = bool(getattr(self.config.strategy, "exit_on_opposite", True))
            if opp_positions:
                if not exit_on_opp:
                    return  # Feature disabled: keep existing opposite trade

                self._log(
                    f"🔄 Confirmed opposite {sig.signal_type} signal! Closing {len(opp_positions)} opposite {opp_positions[0].type} position(s) for Exit & Reverse.",
                    "INFO",
                )
                for opp_p in opp_positions:
                    close_res = self.client.close_position(opp_p.ticket, comment="Exit&Reverse")
                    if close_res.success:
                        self._log(f"Closed opposite #{opp_p.ticket} ({opp_p.type}) for Exit & Reverse", "INFO")
                    else:
                        self._log(f"Failed to close opposite #{opp_p.ticket}: {close_res.error_message}", "ERROR")

                # Refresh tick, open positions and account balance after closing opposite position
                tick = self.client.get_tick(symbol) or tick
                positions = self.client.get_open_positions()
                account = self.client.get_account_info()
                if not account:
                    return

            # Risk Checks
            can_trade, risk_msg = self.risk_manager.can_open_position(
                account=account,
                symbol_info=sym_info,
                tick=tick,
                current_open_positions_count=len(positions),
            )
            if not can_trade:
                self._log(f"Signal {sig.signal_type} blocked by Risk Manager: {risk_msg}", "WARNING")
                return

            # Position Sizing
            if sig.sl_points > 0:
                sl_points = sig.sl_points
            elif hasattr(self.config.strategy, "sl_dollars") and float(getattr(self.config.strategy, "sl_dollars", 0)) > 0:
                sl_points = float(self.config.strategy.sl_dollars) / sym_info.point
            else:
                sl_points = sym_info.pip_to_price(self.config.strategy.sl_pips) / sym_info.point
            lot_size = self.risk_manager.calculate_lot_size(account, sym_info, sl_points)

            # Adjust SL and TP price levels to match exact cash dollars for this lot size
            sl_price = sig.sl_price
            tp_price = sig.tp_price
            use_atr = bool(getattr(self.config.strategy, "use_atr_stops", False))
            sl_cash = float(getattr(self.config.strategy, "sl_dollars", 0.0))
            tp_cash = float(getattr(self.config.strategy, "tp_dollars", 0.0))
            if not use_atr and sl_cash > 0:
                sl_dist = sym_info.cash_to_price_dist(sl_cash, lot_size)
                tp_dist = sym_info.cash_to_price_dist(tp_cash, lot_size) if tp_cash > 0 else 0.0
                if sig.signal_type == "BUY":
                    sl_price = round(tick.ask - sl_dist, sym_info.digits)
                    tp_price = round(tick.ask + tp_dist, sym_info.digits) if tp_dist > 0 else None
                elif sig.signal_type == "SELL":
                    sl_price = round(tick.bid + sl_dist, sym_info.digits)
                    tp_price = round(tick.bid - tp_dist, sym_info.digits) if tp_dist > 0 else None

            self._log(
                f"Executing {sig.signal_type} {lot_size} lots on {symbol} (SL: {sl_price}, TP: {tp_price}) - Reason: {sig.reason}",
                "INFO",
            )

            res = self.client.open_market_order(
                symbol=symbol,
                order_type=sig.signal_type,
                volume=lot_size,
                sl=sl_price,
                tp=tp_price,
                magic=self.config.strategy.magic_number,
                comment=f"Bot:{self.config.strategy.name[:12]}",
            )

            if res.success:
                self._last_traded_candle = closed_candle_time
                self._log(f"Order filled: #{res.order} at {res.price}", "INFO")
            else:
                self._log(f"Order failed: {res.error_message}", "ERROR")

    # Manual UI Actions
    def place_manual_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> OrderResult:
        res = self.client.open_market_order(
            symbol=symbol,
            order_type=order_type,
            volume=volume,
            sl=sl,
            tp=tp,
            magic=999999,
            comment="Manual-UI",
        )
        self._log(
            f"Manual {order_type} {volume} on {symbol}: {'SUCCESS #' + str(res.order) if res.success else 'FAILED: ' + res.error_message}",
            "INFO" if res.success else "ERROR",
        )
        return res

    def close_all_emergency(self) -> List[OrderResult]:
        self._log("EMERGENCY KILL SWITCH ACTIVATED - CLOSING ALL POSITIONS", "WARNING")
        results = self.client.close_all_positions()
        successes = sum(1 for r in results if r.success)
        self._log(f"Closed {successes}/{len(results)} open positions", "INFO")
        return results

    def reset_circuit_breaker(self) -> None:
        self.risk_manager.reset_circuit_breaker()
        self._log("Circuit breaker reset", "INFO")
