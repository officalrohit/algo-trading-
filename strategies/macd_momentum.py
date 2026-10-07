import pandas as pd
from core.client_interface import SymbolInfo, TickData
from strategies.base_strategy import BaseStrategy, Signal


class MACDMomentumStrategy(BaseStrategy):
    """MACD Momentum Crossover Strategy detecting trend accelerations."""

    def __init__(self, params: dict = None):
        default_params = {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
            "sl_pips": 30.0,
            "tp_pips": 60.0,
        }
        if params:
            default_params.update(params)
        super().__init__("MACD Momentum", default_params)

    def generate_signal(self, df: pd.DataFrame, tick: TickData, spec: SymbolInfo) -> Signal:
        min_bars = self.params["slow_period"] + self.params["signal_period"] + 5
        if len(df) < min_bars:
            return Signal(symbol=tick.symbol, signal_type="HOLD", reason="Insufficient candle history")

        closes = df["close"]
        macd_line, signal_line, hist = self.macd(
            closes,
            self.params["fast_period"],
            self.params["slow_period"],
            self.params["signal_period"],
        )

        prev_macd, curr_macd = macd_line.iloc[-2], macd_line.iloc[-1]
        prev_sig, curr_sig = signal_line.iloc[-2], signal_line.iloc[-1]
        curr_hist = hist.iloc[-1]

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

        # Bullish MACD cross
        if prev_macd <= prev_sig and curr_macd > curr_sig and curr_hist > 0:
            sl_price = round(tick.ask - sl_dist, digits)
            tp_price = round(tick.ask + tp_dist, digits)
            return Signal(
                symbol=tick.symbol,
                signal_type="BUY",
                sl_price=sl_price,
                tp_price=tp_price,
                sl_points=sl_points,
                reason=f"Bullish MACD cross: MACD {curr_macd:.5f} > Signal {curr_sig:.5f}",
            )

        # Bearish MACD cross
        if prev_macd >= prev_sig and curr_macd < curr_sig and curr_hist < 0:
            sl_price = round(tick.bid + sl_dist, digits)
            tp_price = round(tick.bid - tp_dist, digits)
            return Signal(
                symbol=tick.symbol,
                signal_type="SELL",
                sl_price=sl_price,
                tp_price=tp_price,
                sl_points=sl_points,
                reason=f"Bearish MACD cross: MACD {curr_macd:.5f} < Signal {curr_sig:.5f}",
            )

        return Signal(symbol=tick.symbol, signal_type="HOLD", reason="No MACD crossover")
