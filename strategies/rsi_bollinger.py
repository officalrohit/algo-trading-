import pandas as pd
from core.client_interface import SymbolInfo, TickData
from strategies.base_strategy import BaseStrategy, Signal


class RSIBollingerStrategy(BaseStrategy):
    """Mean Reversion Strategy exploiting extreme Bollinger Band touches confirmed by RSI overbought/oversold."""

    def __init__(self, params: dict = None):
        default_params = {
            "rsi_period": 14,
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0,
            "bb_period": 20,
            "bb_std": 2.0,
            "sl_pips": 25.0,
            "tp_pips": 45.0,
        }
        if params:
            default_params.update(params)
        super().__init__("RSI + Bollinger Bands", default_params)

    def generate_signal(self, df: pd.DataFrame, tick: TickData, spec: SymbolInfo) -> Signal:
        min_bars = max(self.params["bb_period"], self.params["rsi_period"]) + 5
        if len(df) < min_bars:
            return Signal(symbol=tick.symbol, signal_type="HOLD", reason="Insufficient candle history")

        closes = df["close"]
        lows = df["low"] if "low" in df.columns else closes
        highs = df["high"] if "high" in df.columns else closes

        upper_bb, mid_bb, lower_bb = self.bollinger_bands(
            closes, self.params["bb_period"], self.params["bb_std"]
        )
        rsi = self.rsi(closes, self.params["rsi_period"])

        curr_close = closes.iloc[-1]
        curr_low = lows.iloc[-1]
        curr_high = highs.iloc[-1]
        curr_rsi = rsi.iloc[-1]
        curr_lower = lower_bb.iloc[-1]
        curr_upper = upper_bb.iloc[-1]

        point = spec.point
        digits = spec.digits
        sym = spec.name.upper()
        is_dollar_asset = "XAU" in sym or "XAG" in sym or "GOLD" in sym or "BTC" in sym
        if is_dollar_asset and self.params.get("sl_dollars") is not None and float(self.params.get("sl_dollars", 0)) > 0:
            sl_dist = float(self.params["sl_dollars"])
            tp_dist = float(self.params.get("tp_dollars", self.params["sl_dollars"] * 2.0))
        else:
            sl_dist = spec.pip_to_price(self.params.get("sl_pips", 25.0))
            tp_dist = spec.pip_to_price(self.params.get("tp_pips", 50.0))
        sl_points = sl_dist / point

        # Buy condition: Price low/close pierced Lower BB and RSI is oversold
        if (curr_low <= curr_lower or curr_close <= curr_lower) and curr_rsi <= self.params["rsi_oversold"]:
            sl_price = round(tick.ask - sl_dist, digits)
            tp_price = round(tick.ask + tp_dist, digits)
            return Signal(
                symbol=tick.symbol,
                signal_type="BUY",
                sl_price=sl_price,
                tp_price=tp_price,
                sl_points=sl_points,
                reason=f"Oversold bounce: Low {curr_low:.5f} <= Lower BB {curr_lower:.5f} with RSI {curr_rsi:.1f}",
            )

        # Sell condition: Price high/close pierced Upper BB and RSI is overbought
        if (curr_high >= curr_upper or curr_close >= curr_upper) and curr_rsi >= self.params["rsi_overbought"]:
            sl_price = round(tick.bid + sl_dist, digits)
            tp_price = round(tick.bid - tp_dist, digits)
            return Signal(
                symbol=tick.symbol,
                signal_type="SELL",
                sl_price=sl_price,
                tp_price=tp_price,
                sl_points=sl_points,
                reason=f"Overbought rejection: Close {curr_close:.5f} >= Upper BB {curr_upper:.5f} with RSI {curr_rsi:.1f}",
            )

        return Signal(symbol=tick.symbol, signal_type="HOLD", reason="Inside normal Bollinger/RSI bounds")
