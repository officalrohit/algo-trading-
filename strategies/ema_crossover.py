import pandas as pd
from core.client_interface import SymbolInfo, TickData
from strategies.base_strategy import BaseStrategy, Signal


class EMACrossoverStrategy(BaseStrategy):
    """Dual / Triple EMA Trend Following Strategy with 200 EMA filter and ATR dynamic stops."""

    def __init__(self, params: dict = None):
        default_params = {
            "fast_period": 9,
            "slow_period": 21,
            "trend_filter": 200,
            "use_atr_stops": False,
            "atr_period": 14,
            "atr_sl_mult": 1.5,
            "atr_tp_mult": 2.5,
            "sl_pips": 25.0,
            "tp_pips": 50.0,
            "fixed_sl_pips": 25.0,
            "fixed_tp_pips": 50.0,
        }
        if params:
            default_params.update(params)
            if "sl_pips" in params:
                default_params["fixed_sl_pips"] = params["sl_pips"]
            if "tp_pips" in params:
                default_params["fixed_tp_pips"] = params["tp_pips"]
        super().__init__("EMA Crossover", default_params)

    def generate_signal(self, df: pd.DataFrame, tick: TickData, spec: SymbolInfo) -> Signal:
        is_single_ema_mode = self.params["fast_period"] == self.params["slow_period"]
        use_trend = self.params.get("use_trend_filter", False if is_single_ema_mode else True)
        min_bars = (max(self.params["trend_filter"], self.params["slow_period"]) + 5) if use_trend else (self.params["fast_period"] + 5)
        if len(df) < min_bars:
            return Signal(symbol=tick.symbol, signal_type="HOLD", reason="Insufficient candle history")

        closes = df["close"]
        highs = df["high"]
        lows = df["low"]

        fast_ema = self.ema(closes, self.params["fast_period"])
        slow_ema = self.ema(closes, self.params["slow_period"])
        trend_ema = self.ema(closes, self.params["trend_filter"])
        atr = self.atr(highs, lows, closes, self.params["atr_period"])

        # Compare previous candle (-2) with current/last completed candle (-1)
        prev_fast, curr_fast = fast_ema.iloc[-2], fast_ema.iloc[-1]
        prev_slow, curr_slow = slow_ema.iloc[-2], slow_ema.iloc[-1]
        curr_trend = trend_ema.iloc[-1]
        curr_price = closes.iloc[-1]
        curr_atr = atr.iloc[-1]

        point = spec.point
        digits = spec.digits

        # Calculate SL / TP distances
        use_atr = self.params.get("use_atr_stops", False)
        if use_atr and curr_atr > 0:
            sl_dist = curr_atr * self.params.get("atr_sl_mult", 1.5)
            tp_dist = curr_atr * self.params.get("atr_tp_mult", 2.5)
        else:
            sl_pips = self.params.get("sl_pips", self.params.get("fixed_sl_pips", 25.0))
            tp_pips = self.params.get("tp_pips", self.params.get("fixed_tp_pips", 50.0))
            sl_dist = spec.pip_to_price(sl_pips)
            tp_dist = spec.pip_to_price(tp_pips)

        sl_points = sl_dist / point

        # Support both Dual EMA Crossover (Fast vs Slow) and Single EMA Price Crossover (Price vs EMA)
        is_single_ema_mode = self.params["fast_period"] == self.params["slow_period"]
        use_trend = self.params.get("use_trend_filter", False if is_single_ema_mode else True)

        # Evaluate strictly on CLOSED candles: iloc[-3] (previous) and iloc[-2] (just closed)
        close_before = closes.iloc[-3]
        close_crossed = closes.iloc[-2]

        if is_single_ema_mode:
            ema_val = self.params["fast_period"]
            ema_before = fast_ema.iloc[-3]
            ema_crossed = fast_ema.iloc[-2]
            trend_val = trend_ema.iloc[-2]

            # Price crossed and CLOSED ABOVE EMA
            if close_before <= ema_before and close_crossed > ema_crossed:
                if not use_trend or close_crossed >= trend_val:
                    sl_price = round(tick.ask - sl_dist, digits)
                    tp_price = round(tick.ask + tp_dist, digits)
                    return Signal(
                        symbol=tick.symbol,
                        signal_type="BUY",
                        sl_price=sl_price,
                        tp_price=tp_price,
                        sl_points=sl_points,
                        reason=f"Confirmed: Candle CLOSED ABOVE {ema_val} EMA (Close {close_crossed:.2f} > EMA {ema_crossed:.2f})",
                        time=df.index[-2].to_pydatetime() if hasattr(df.index[-2], "to_pydatetime") else df.index[-2],
                    )

            # Price crossed and CLOSED BELOW EMA
            if close_before >= ema_before and close_crossed < ema_crossed:
                if not use_trend or close_crossed <= trend_val:
                    sl_price = round(tick.bid + sl_dist, digits)
                    tp_price = round(tick.bid - tp_dist, digits)
                    return Signal(
                        symbol=tick.symbol,
                        signal_type="SELL",
                        sl_price=sl_price,
                        tp_price=tp_price,
                        sl_points=sl_points,
                        reason=f"Confirmed: Candle CLOSED BELOW {ema_val} EMA (Close {close_crossed:.2f} < EMA {ema_crossed:.2f})",
                        time=df.index[-2].to_pydatetime() if hasattr(df.index[-2], "to_pydatetime") else df.index[-2],
                    )
        else:
            # Dual EMA Crossover confirmed on closed candle
            fast_before = fast_ema.iloc[-3]
            slow_before = slow_ema.iloc[-3]
            fast_crossed = fast_ema.iloc[-2]
            slow_crossed = slow_ema.iloc[-2]
            trend_val = trend_ema.iloc[-2]

            if fast_before <= slow_before and fast_crossed > slow_crossed:
                if not use_trend or close_crossed >= trend_val:
                    sl_price = round(tick.ask - sl_dist, digits)
                    tp_price = round(tick.ask + tp_dist, digits)
                    return Signal(
                        symbol=tick.symbol,
                        signal_type="BUY",
                        sl_price=sl_price,
                        tp_price=tp_price,
                        sl_points=sl_points,
                        reason=f"Confirmed: Fast EMA crossed ABOVE Slow EMA on closed candle ({fast_crossed:.5f} > {slow_crossed:.5f})",
                        time=df.index[-2].to_pydatetime() if hasattr(df.index[-2], "to_pydatetime") else df.index[-2],
                    )

            if fast_before >= slow_before and fast_crossed < slow_crossed:
                if not use_trend or close_crossed <= trend_val:
                    sl_price = round(tick.bid + sl_dist, digits)
                    tp_price = round(tick.bid - tp_dist, digits)
                    return Signal(
                        symbol=tick.symbol,
                        signal_type="SELL",
                        sl_price=sl_price,
                        tp_price=tp_price,
                        sl_points=sl_points,
                        reason=f"Confirmed: Fast EMA crossed BELOW Slow EMA on closed candle ({fast_crossed:.5f} < {slow_crossed:.5f})",
                        time=df.index[-2].to_pydatetime() if hasattr(df.index[-2], "to_pydatetime") else df.index[-2],
                    )

        return Signal(symbol=tick.symbol, signal_type="HOLD", reason=f"Waiting for candle to cross and CLOSE across {self.params['fast_period']} EMA" if is_single_ema_mode else "No EMA crossover condition met")
