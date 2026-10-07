from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from core.client_interface import SymbolInfo, TickData
from strategies.base_strategy import BaseStrategy


@dataclass
class BacktestTrade:
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    symbol: str
    direction: str  # BUY or SELL
    entry_price: float
    exit_price: float
    volume: float
    sl: float
    tp: float
    pnl: float
    exit_reason: str


@dataclass
class BacktestResult:
    metrics: Dict[str, float]
    trades: List[BacktestTrade]
    trades_df: pd.DataFrame
    equity_curve: pd.DataFrame


class BacktestEngine:
    """Event-driven bar-by-bar historical backtesting engine with realistic
    intrabar execution, slippage/spread modeling, and performance metrics."""

    def __init__(
        self,
        strategy: BaseStrategy,
        symbol_info: SymbolInfo,
        initial_capital: float = 10000.0,
        risk_pct: float = 1.5,
        spread_pips: float = 1.5,
        fixed_lot: Optional[float] = None,
        exit_on_opposite: bool = True,
    ):
        self.strategy = strategy
        self.spec = symbol_info
        self.initial_capital = initial_capital
        self.risk_pct = risk_pct
        self.spread_points = spread_pips * 10
        self.fixed_lot = fixed_lot
        self.exit_on_opposite = exit_on_opposite

    def run(self, df: pd.DataFrame) -> BacktestResult:
        if df.empty or len(df) < 30:
            return BacktestResult({}, [], pd.DataFrame(), pd.DataFrame())

        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"])
                df.set_index("time", inplace=True)

        balance = self.initial_capital
        equity = balance
        open_trade: Optional[BacktestTrade] = None
        closed_trades: List[BacktestTrade] = []
        equity_records = []

        point = self.spec.point
        spread_cost = self.spread_points * point

        contract_size = (
            self.spec.get_contract_size()
            if hasattr(self.spec, "get_contract_size")
            else (
                100.0
                if ("XAU" in self.spec.name.upper() or "GOLD" in self.spec.name.upper())
                else (1.0 if "BTC" in self.spec.name.upper() else 100000.0)
            )
        )

        min_required = max(30, self.strategy.params.get("ema_period", 30) + 10, self.strategy.params.get("slow_period", 30) + 10)
        warmup = min(len(df) // 3, min_required)
        for i in range(warmup, len(df)):
            current_bar = df.iloc[i]
            bar_time = df.index[i]
            high = current_bar["high"]
            low = current_bar["low"]
            close = current_bar["close"]

            hist_slice = df.iloc[: i + 1]
            mock_tick = TickData(
                symbol=self.spec.name,
                bid=close,
                ask=close + spread_cost,
                last=close,
                spread_points=int(self.spread_points),
                time=bar_time.to_pydatetime() if hasattr(bar_time, "to_pydatetime") else bar_time,
            )

            # 1. Manage currently open position against this bar's High and Low
            if open_trade is not None:
                trade_closed = False
                exit_price = close
                reason = "Close"

                if open_trade.direction == "BUY":
                    # Check SL
                    if open_trade.sl > 0 and low <= open_trade.sl:
                        exit_price = open_trade.sl
                        reason = "Stop Loss"
                        trade_closed = True
                    # Check TP
                    elif open_trade.tp > 0 and high >= open_trade.tp:
                        exit_price = open_trade.tp
                        reason = "Take Profit"
                        trade_closed = True

                elif open_trade.direction == "SELL":
                    # Check SL
                    if open_trade.sl > 0 and high >= open_trade.sl:
                        exit_price = open_trade.sl
                        reason = "Stop Loss"
                        trade_closed = True
                    # Check TP
                    elif open_trade.tp > 0 and low <= open_trade.tp:
                        exit_price = open_trade.tp
                        reason = "Take Profit"
                        trade_closed = True

                # Check for opposite signal exit if SL/TP not hit
                if not trade_closed and self.exit_on_opposite and i < len(df) - 1:
                    bar_sig = self.strategy.generate_signal(hist_slice, mock_tick, self.spec)
                    if open_trade.direction == "BUY" and bar_sig.signal_type == "SELL":
                        exit_price = mock_tick.bid
                        reason = "Opposite Signal (SELL)"
                        trade_closed = True
                    elif open_trade.direction == "SELL" and bar_sig.signal_type == "BUY":
                        exit_price = mock_tick.ask
                        reason = "Opposite Signal (BUY)"
                        trade_closed = True

                if trade_closed:
                    if open_trade.direction == "BUY":
                        pnl = (exit_price - open_trade.entry_price) * open_trade.volume * contract_size
                    else:
                        pnl = (open_trade.entry_price - exit_price) * open_trade.volume * contract_size

                    balance += pnl
                    equity = balance
                    open_trade.exit_time = bar_time
                    open_trade.exit_price = exit_price
                    open_trade.pnl = round(pnl, 2)
                    open_trade.exit_reason = reason
                    closed_trades.append(open_trade)
                    open_trade = None
                else:
                    if open_trade.direction == "BUY":
                        equity = balance + (close - open_trade.entry_price) * open_trade.volume * contract_size
                    else:
                        equity = balance + (open_trade.entry_price - close) * open_trade.volume * contract_size

            # 2. Check for new signals if no position currently open
            if open_trade is None and i < len(df) - 1:
                sig = self.strategy.generate_signal(hist_slice, mock_tick, self.spec)

                if sig.signal_type in ("BUY", "SELL"):
                    # Calculate lot size
                    step = self.spec.lot_step if (self.spec and self.spec.lot_step > 0) else 0.01
                    min_l = self.spec.min_lot if (self.spec and self.spec.min_lot > 0) else 0.01
                    max_l = self.spec.max_lot if (self.spec and self.spec.max_lot > 0) else 100.0

                    if self.fixed_lot is not None and self.fixed_lot > 0:
                        vol = round(round(self.fixed_lot / step) * step, 2)
                        volume = max(min_l, min(vol, max_l))
                    else:
                        risk_amount = max(0.0, equity) * (self.risk_pct / 100.0)
                        sl_dist = sig.sl_points * point if sig.sl_points > 0 else self.spec.pip_to_price(25.0)
                        point_val = point * contract_size
                        if sl_dist * point_val > 0:
                            calc_vol = round(risk_amount / (sl_dist * point_val), 2)
                        else:
                            calc_vol = min_l
                        volume = max(min_l, min(calc_vol, max_l))

                    entry_p = mock_tick.ask if sig.signal_type == "BUY" else mock_tick.bid
                    open_trade = BacktestTrade(
                        entry_time=bar_time,
                        exit_time=None,
                        symbol=self.spec.name,
                        direction=sig.signal_type,
                        entry_price=entry_p,
                        exit_price=0.0,
                        volume=volume,
                        sl=sig.sl_price or 0.0,
                        tp=sig.tp_price or 0.0,
                        pnl=0.0,
                        exit_reason="",
                    )

            equity_records.append({"time": bar_time, "balance": balance, "equity": equity})

        # Close open trade on final bar
        if open_trade is not None:
            last_close = df.iloc[-1]["close"]
            if open_trade.direction == "BUY":
                pnl = (last_close - open_trade.entry_price) * open_trade.volume * contract_size
            else:
                pnl = (open_trade.entry_price - last_close) * open_trade.volume * contract_size

            balance += pnl
            open_trade.exit_time = df.index[-1]
            open_trade.exit_price = last_close
            open_trade.pnl = round(pnl, 2)
            open_trade.exit_reason = "End of Backtest"
            closed_trades.append(open_trade)

        trades_df = pd.DataFrame([t.__dict__ for t in closed_trades])
        equity_df = pd.DataFrame(equity_records)
        if not equity_df.empty:
            equity_df.set_index("time", inplace=True)

        metrics = self._calculate_metrics(balance, closed_trades, equity_df)
        return BacktestResult(
            metrics=metrics,
            trades=closed_trades,
            trades_df=trades_df,
            equity_curve=equity_df,
        )

    def _calculate_metrics(
        self,
        final_balance: float,
        trades: List[BacktestTrade],
        equity_df: pd.DataFrame,
    ) -> Dict[str, float]:
        total_trades = len(trades)
        if total_trades == 0:
            return {
                "Initial Balance": self.initial_capital,
                "Final Balance": final_balance,
                "Net Profit ($)": 0.0,
                "Total Return (%)": 0.0,
                "Total Trades": 0,
                "Win Rate (%)": 0.0,
                "Profit Factor": 0.0,
                "Max Drawdown (%)": 0.0,
                "Sharpe Ratio": 0.0,
            }

        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        net_profit = final_balance - self.initial_capital
        total_return_pct = (net_profit / self.initial_capital) * 100.0
        win_rate = (len(wins) / total_trades) * 100.0 if total_trades > 0 else 0.0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 99.0

        # Drawdown calculation
        max_dd_pct = 0.0
        if not equity_df.empty and "equity" in equity_df.columns:
            peak = equity_df["equity"].cummax()
            dd = (peak - equity_df["equity"]) / peak * 100.0
            max_dd_pct = round(float(dd.max()), 2)

        # Sharpe Ratio (annualized based on trade returns)
        sharpe = 0.0
        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe = round(float((np.mean(pnls) / np.std(pnls)) * np.sqrt(252)), 2)

        return {
            "Initial Balance": self.initial_capital,
            "Final Balance": round(final_balance, 2),
            "Net Profit ($)": round(net_profit, 2),
            "Total Return (%)": round(total_return_pct, 2),
            "Total Trades": total_trades,
            "Winning Trades": len(wins),
            "Losing Trades": len(losses),
            "Win Rate (%)": round(win_rate, 1),
            "Profit Factor": profit_factor,
            "Max Drawdown (%)": max_dd_pct,
            "Sharpe Ratio": sharpe,
        }
