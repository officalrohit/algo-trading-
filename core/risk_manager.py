from datetime import date, datetime, timezone
import logging
from typing import Optional, Tuple
from config import RiskConfig
from core.client_interface import AccountInfo, Position, SymbolInfo, TickData

logger = logging.getLogger("RiskManager")


class RiskManager:
    """Rigorous risk control layer enforcing dynamic position sizing,
    daily loss circuit breakers, spread monitoring, and trailing stops."""

    def __init__(self, config: RiskConfig):
        self.config = config
        self._current_day: date = datetime.now(timezone.utc).date()
        self._daily_realized_loss: float = 0.0
        self._circuit_breaker_tripped: bool = False

    def update_config(self, config: RiskConfig) -> None:
        self.config = config

    def _check_and_reset_day(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self._current_day:
            self._current_day = today
            self._daily_realized_loss = 0.0
            self._circuit_breaker_tripped = False
            logger.info(f"New trading day {today} - Daily risk metrics reset.")

    def record_deal_pnl(self, profit: float) -> None:
        """Call whenever a position is closed to track daily loss."""
        self._check_and_reset_day()
        if profit < 0:
            self._daily_realized_loss += abs(profit)
            if self._daily_realized_loss >= self.config.max_daily_loss:
                self._circuit_breaker_tripped = True
                logger.error(
                    f"CIRCUIT BREAKER TRIPPED! Daily loss ${self._daily_realized_loss:.2f} "
                    f"exceeded maximum limit of ${self.config.max_daily_loss:.2f}."
                )

    def is_circuit_breaker_active(self, account: AccountInfo) -> Tuple[bool, str]:
        """Checks if trading is blocked due to daily loss limit or floating drawdown."""
        self._check_and_reset_day()

        if self._daily_realized_loss >= self.config.max_daily_loss:
            self._circuit_breaker_tripped = True
            return True, f"Circuit breaker active: Daily loss reached ${self._daily_realized_loss:.2f}"

        # Check current floating negative equity + realized loss
        current_loss = self._daily_realized_loss + (abs(account.profit) if account.profit < 0 else 0.0)
        if current_loss >= self.config.max_daily_loss:
            self._circuit_breaker_tripped = True
            return True, f"Circuit breaker tripped: Floating + Realized loss (${current_loss:.2f}) exceeds limit of ${self.config.max_daily_loss:.2f}"

        # Loss is within acceptable bounds
        self._circuit_breaker_tripped = False
        return False, ""

    def reset_circuit_breaker(self) -> None:
        """Manual reset by user from UI."""
        self._circuit_breaker_tripped = False
        self._daily_realized_loss = 0.0
        logger.info("Circuit breaker manually reset by user.")

    def can_open_position(
        self,
        account: AccountInfo,
        symbol_info: SymbolInfo,
        tick: TickData,
        current_open_positions_count: int,
    ) -> Tuple[bool, str]:
        """Validates all risk constraints before an order is placed."""
        self._check_and_reset_day()

        # 1. Circuit breaker check
        tripped, reason = self.is_circuit_breaker_active(account)
        if tripped:
            return False, reason

        # 2. Maximum open positions check
        if current_open_positions_count >= self.config.max_open_positions:
            return False, f"Max open positions reached ({current_open_positions_count}/{self.config.max_open_positions})"

        # 3. Spread check
        if tick.spread_points > self.config.max_spread_points:
            return False, f"Spread too wide ({tick.spread_points} pts > max {self.config.max_spread_points} pts)"

        # 4. Free margin availability
        if account.free_margin <= 0 or (account.free_margin / account.equity < 0.20):
            return False, f"Insufficient free margin (${account.free_margin:.2f})"

        return True, "Risk checks passed"

    def calculate_lot_size(
        self,
        account: AccountInfo,
        symbol_info: SymbolInfo,
        sl_points: float,
    ) -> float:
        """Computes optimal lot size based on account balance/equity and SL distance."""
        if not self.config.use_dynamic_lot or sl_points <= 0:
            fixed = self.config.fixed_lot or symbol_info.min_lot
            return self._normalize_lots(fixed, symbol_info)

        # Money to risk
        risk_cash = account.equity * (self.config.max_risk_pct / 100.0)

        # Pip/point value approximation
        # For standard 1.0 lot forex (100k contract), 1 point (0.00001) = $1.00 approx
        # For Gold (100 oz), 1 point (0.01) = $1.00 approx
        # For BTC (1 unit), 1 point (0.01) = $0.01 approx
        if "BTC" in symbol_info.name:
            point_value_per_lot = symbol_info.point * 1.0
        elif "XAU" in symbol_info.name:
            point_value_per_lot = symbol_info.point * 100.0
        else:
            point_value_per_lot = symbol_info.point * 100000.0

        if point_value_per_lot <= 0:
            point_value_per_lot = 1.0

        raw_lot = risk_cash / (sl_points * point_value_per_lot)
        return self._normalize_lots(raw_lot, symbol_info)

    def _normalize_lots(self, lot: float, spec: SymbolInfo) -> float:
        """Clamps lot to min/max and rounds to nearest lot_step."""
        step = spec.lot_step
        if step <= 0:
            step = 0.01
        normalized = round(round(lot / step) * step, 2)
        return max(spec.min_lot, min(normalized, spec.max_lot))

    def calculate_trailing_stop(
        self,
        position: Position,
        tick: TickData,
        symbol_info: SymbolInfo,
    ) -> Optional[float]:
        """Calculates new Stop Loss price if trailing condition is met, else returns None."""
        if not self.config.use_trailing_stop:
            return None

        point = symbol_info.point
        trailing_dist = symbol_info.pip_to_price(self.config.trailing_stop_pips)
        digits = symbol_info.digits

        if position.type == "BUY":
            current_profit_dist = tick.bid - position.price_open
            if current_profit_dist >= trailing_dist:
                proposed_sl = round(tick.bid - trailing_dist, digits)
                if position.sl == 0 or proposed_sl > (position.sl + (point * 5)):
                    return proposed_sl

        elif position.type == "SELL":
            current_profit_dist = position.price_open - tick.ask
            if current_profit_dist >= trailing_dist:
                proposed_sl = round(tick.ask + trailing_dist, digits)
                if position.sl == 0 or proposed_sl < (position.sl - (point * 5)):
                    return proposed_sl

        return None
