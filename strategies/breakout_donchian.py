import pandas as pd
from core.client_interface import SymbolInfo, TickData
from strategies.base_strategy import BaseStrategy, Signal


class DonchianBreakoutStrategy(BaseStrategy):
    """Turtle-style Donchian Channel Breakout Strategy for capturing large directional runs."""

    def __init__(self, params: dict = None):
        default_params = {
            "period": 20,
            "atr_period": 14,
            "atr_sl_mult": 2.0,
            "atr_tp_mult": 3.5,
        }
        if params:
            default_params.update(params)
        super().__init__("Donchian Breakout", default_params)

    def generate_signal(self, df: pd.DataFrame, tick: TickData, spec: SymbolInfo) -> Signal:
        min_bars = self.params["period"] + 5
        if len(df) < min_bars:
            return Signal(symbol=tick.symbol, signal_type="HOLD", reason="Insufficient candle history")

        highs = df["high"]
        lows = df["low"]
        closes = df["close"]

        upper_chan, lower_chan = self.donchian_channel(highs, lows, self.params["period"])
        atr = self.atr(highs, lows, closes, self.params["atr_period"])

        # Compare with previous candle channel levels to avoid lookahead bias
        prev_upper = upper_chan.iloc[-2]
        prev_lower = lower_chan.iloc[-2]
        curr_close = closes.iloc[-1]
        curr_atr = atr.iloc[-1]

        point = spec.point
        digits = spec.digits
        sl_dist = curr_atr * self.params["atr_sl_mult"]
        tp_dist = curr_atr * self.params["atr_tp_mult"]
        sl_points = sl_dist / point

        # Breakout above highest high
        if curr_close > prev_upper:
            sl_price = round(tick.ask - sl_dist, digits)
            tp_price = round(tick.ask + tp_dist, digits)
            return Signal(
                symbol=tick.symbol,
                signal_type="BUY",
                sl_price=sl_price,
                tp_price=tp_price,
                sl_points=sl_points,
                reason=f"Breakout above {self.params['period']}-period Donchian Upper ({prev_upper:.5f})",
            )

        # Breakdown below lowest low
        if curr_close < prev_lower:
            sl_price = round(tick.bid + sl_dist, digits)
            tp_price = round(tick.bid - tp_dist, digits)
            return Signal(
                symbol=tick.symbol,
                signal_type="SELL",
                sl_price=sl_price,
                tp_price=tp_price,
                sl_points=sl_points,
                reason=f"Breakdown below {self.params['period']}-period Donchian Lower ({prev_lower:.5f})",
            )

        return Signal(symbol=tick.symbol, signal_type="HOLD", reason="Within Donchian channel bounds")
