import pandas as pd
from core.client_interface import SymbolInfo, TickData
from strategies.base_strategy import BaseStrategy, Signal


class PriceCrossEMAStrategy(BaseStrategy):
    """Single EMA Price Action Strategy (e.g. 44 EMA):
    - Price crosses ABOVE 44 EMA -> BUY
    - Price crosses BELOW 44 EMA -> SELL
    """

    def __init__(self, params: dict = None):
        default_params = {
            "ema_period": 44,
            "use_trend_filter": False,
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
            # Support fast_period if passed from generic config
            if "fast_period" in params and params["fast_period"]:
                default_params["ema_period"] = params["fast_period"]
            if "sl_pips" in params:
                default_params["fixed_sl_pips"] = params["sl_pips"]
            if "tp_pips" in params:
                default_params["fixed_tp_pips"] = params["tp_pips"]

        super().__init__("Price Cross EMA", default_params)

    def generate_signal(self, df: pd.DataFrame, tick: TickData, spec: SymbolInfo) -> Signal:
        ema_len = self.params["ema_period"]
        min_bars = ema_len + 10
        if len(df) < min_bars:
            return Signal(symbol=tick.symbol, signal_type="HOLD", reason=f"Insufficient history for {ema_len} EMA")

        closes = df["close"]
        highs = df["high"]
        lows = df["low"]

        ema_series = self.ema(closes, ema_len)
        atr = self.atr(highs, lows, closes, self.params["atr_period"])

        # Use closed candles: iloc[-3] (before cross) and iloc[-2] (crossed closed candle)
        close_before = closes.iloc[-3]
        close_crossed = closes.iloc[-2]
        ema_before = ema_series.iloc[-3]
        ema_crossed = ema_series.iloc[-2]
        curr_atr = atr.iloc[-2]

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

        # Optional Trend Filter
        use_trend = self.params.get("use_trend_filter", False)
        if use_trend:
            trend_series = self.ema(closes, self.params["trend_filter"])
            curr_trend = trend_series.iloc[-2]
        else:
            curr_trend = 0.0

        # Bullish: Previous close <= EMA, and crossing candle CLOSED ABOVE EMA
        if close_before <= ema_before and close_crossed > ema_crossed:
            if not use_trend or close_crossed >= curr_trend:
                sl_price = round(tick.ask - sl_dist, digits)
                tp_price = round(tick.ask + tp_dist, digits)
                return Signal(
                    symbol=tick.symbol,
                    signal_type="BUY",
                    sl_price=sl_price,
                    tp_price=tp_price,
                    sl_points=sl_points,
                    reason=f"Confirmed: Candle CLOSED ABOVE {ema_len} EMA (Close {close_crossed:.2f} > EMA {ema_crossed:.2f})",
                    time=df.index[-2].to_pydatetime() if hasattr(df.index[-2], "to_pydatetime") else df.index[-2],
                )

        # Bearish: Previous close >= EMA, and crossing candle CLOSED BELOW EMA
        if close_before >= ema_before and close_crossed < ema_crossed:
            if not use_trend or close_crossed <= curr_trend:
                sl_price = round(tick.bid + sl_dist, digits)
                tp_price = round(tick.bid - tp_dist, digits)
                return Signal(
                    symbol=tick.symbol,
                    signal_type="SELL",
                    sl_price=sl_price,
                    tp_price=tp_price,
                    sl_points=sl_points,
                    reason=f"Confirmed: Candle CLOSED BELOW {ema_len} EMA (Close {close_crossed:.2f} < EMA {ema_crossed:.2f})",
                    time=df.index[-2].to_pydatetime() if hasattr(df.index[-2], "to_pydatetime") else df.index[-2],
                )

        return Signal(symbol=tick.symbol, signal_type="HOLD", reason=f"Waiting for candle to cross and CLOSE across {ema_len} EMA")
