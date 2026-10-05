import os
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Ensure root folder in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import AppConfig, load_config, save_config
from core.client_interface import Position
from core.mt5_client import MT5_AVAILABLE
from strategies import AVAILABLE_STRATEGIES
from backtester.engine import BacktestEngine
from trading_engine import TradingEngine

# Set Page Config
st.set_page_config(
    page_title="MetaTrader 5 Algorithmic Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Professional Financial Terminal look
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1e222d;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #2a2e39;
        margin-bottom: 10px;
    }
    .metric-title {
        color: #787b86;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-val {
        font-size: 24px;
        font-weight: 700;
        margin-top: 5px;
    }
    .green-text { color: #089981 !important; }
    .red-text { color: #f23645 !important; }
    .neutral-text { color: #d1d4dc !important; }
    .badge-live {
        background-color: #08998122;
        color: #089981;
        border: 1px solid #089981;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-paper {
        background-color: #f59e0b22;
        color: #f59e0b;
        border: 1px solid #f59e0b;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_engine() -> TradingEngine:
    config = load_config()
    engine = TradingEngine(config)
    engine.client.connect()
    return engine


engine = get_engine()

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("⚡ MT5 AlgoBot")

    # Mode Selector
    current_mode = engine.config.mode
    mode_choice = st.radio(
        "Execution Engine",
        options=["Paper Trading (Simulated)", "Live MT5 Connection"],
        index=0 if current_mode == "paper" else 1,
    )
    new_mode = "paper" if "Paper" in mode_choice else "live"
    if new_mode != engine.config.mode:
        engine.config.mode = new_mode
        save_config(engine.config)
        engine.update_config(engine.config)
        st.rerun()

    if engine.config.mode == "live":
        st.markdown('<div class="badge-live">● LIVE MT5 BROKER</div>', unsafe_allow_html=True)
        if not MT5_AVAILABLE:
            st.error("⚠️ MetaTrader5 python library not loaded.")
    else:
        st.markdown('<div class="badge-paper">● PAPER TRADING SIMULATOR</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Active Symbol & Timeframe
    active_sym = st.selectbox(
        "Trading Symbol",
        options=engine.config.tracked_symbols,
        index=engine.config.tracked_symbols.index(engine.config.active_symbol)
        if engine.config.active_symbol in engine.config.tracked_symbols
        else 0,
    )
    timeframe_options = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]
    active_tf = st.selectbox(
        "Timeframe",
        options=timeframe_options,
        index=timeframe_options.index(engine.config.active_timeframe)
        if engine.config.active_timeframe in timeframe_options
        else 1,
    )

    if active_sym != engine.config.active_symbol or active_tf != engine.config.active_timeframe:
        engine.config.active_symbol = active_sym
        engine.config.active_timeframe = active_tf
        save_config(engine.config)

    # Strategy Selector
    strat_names = list(AVAILABLE_STRATEGIES.keys())
    selected_strat_name = st.selectbox(
        "Active Strategy",
        options=strat_names,
        index=strat_names.index(engine.config.strategy.name)
        if engine.config.strategy.name in strat_names
        else 0,
    )
    if selected_strat_name != engine.config.strategy.name:
        engine.config.strategy.name = selected_strat_name
        save_config(engine.config)
        engine.update_config(engine.config)

    st.markdown("---")

    # Bot Start / Stop
    st.subheader("Bot Execution")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("▶ START BOT", type="primary", use_container_width=True, disabled=engine.is_running):
            ok = engine.start()
            if ok:
                st.toast("Algorithmic Trading Bot Started!", icon="🚀")
                st.rerun()
            else:
                st.error("Failed to start bot. Check log for details.")

    with col_btn2:
        if st.button("⏹ STOP BOT", use_container_width=True, disabled=not engine.is_running):
            engine.stop()
            st.toast("Bot stopped.", icon="⏸️")
            st.rerun()

    # Emergency Panic Button
    st.markdown("---")
    st.subheader("🚨 Safety Controls")
    if st.button("KILL SWITCH: CLOSE ALL", use_container_width=True, type="secondary"):
        results = engine.close_all_emergency()
        st.toast(f"Emergency close executed for {len(results)} positions!", icon="⚠️")
        st.rerun()

    if st.button("🔄 Reset Circuit Breaker", use_container_width=True):
        engine.reset_circuit_breaker()
        st.toast("Circuit breaker reset.", icon="✅")
        st.rerun()

    st.markdown("---")
    st.subheader("⚡ Live Auto-Refresh")
    refresh_option = st.selectbox(
        "Screen Update Rate",
        options=["1 second (Real-Time)", "2 seconds", "5 seconds", "Manual Only"],
        index=0,
    )
    refresh_map = {
        "1 second (Real-Time)": "1s",
        "2 seconds": "2s",
        "5 seconds": "5s",
        "Manual Only": None,
    }
    refresh_rate = refresh_map[refresh_option]


# --- FETCH LIVE ACCOUNT & MARKET DATA IN AUTO-REFRESH FRAGMENT ---
@st.fragment(run_every=refresh_rate)
def render_live_kpi_row():
    account = engine.client.get_account_info()
    tripped, trip_reason = engine.risk_manager.is_circuit_breaker_active(account) if account else (False, "")
    if tripped:
        st.error(f"🛑 **CIRCUIT BREAKER ENGAGED**: {trip_reason}. Automated order execution is blocked for your capital protection.")

    col1, col2, col3, col4, col5 = st.columns(5)
    balance_val = f"${account.balance:,.2f}" if account else "$0.00"
    equity_val = f"${account.equity:,.2f}" if account else "$0.00"
    free_margin_val = f"${account.free_margin:,.2f}" if account else "$0.00"
    profit_val = account.profit if account else 0.0
    profit_class = "green-text" if profit_val > 0 else ("red-text" if profit_val < 0 else "neutral-text")

    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Account Balance</div><div class="metric-val neutral-text">{balance_val}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Equity</div><div class="metric-val neutral-text">{equity_val}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Floating P&L</div><div class="metric-val {profit_class}">${profit_val:+,.2f}</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Free Margin</div><div class="metric-val neutral-text">{free_margin_val}</div></div>', unsafe_allow_html=True)
    with col5:
        status_label = "🟢 ACTIVE" if engine.is_running else "⚪ IDLE"
        st.markdown(f'<div class="metric-card"><div class="metric-title">Bot Status</div><div class="metric-val neutral-text">{status_label}</div></div>', unsafe_allow_html=True)

render_live_kpi_row()

# TABS NAVIGATION
tab_terminal, tab_positions, tab_strategy, tab_risk, tab_backtest, tab_logs = st.tabs(
    ["📊 Live Terminal", "💼 Positions & Deals", "🛠️ Strategy Settings", "🛡️ Risk Management", "🧪 Backtesting Lab", "📜 Logs & MT5 Setup"]
)

# =========================================================================
# TAB 1: LIVE TERMINAL
# =========================================================================
with tab_terminal:
    @st.fragment(run_every=refresh_rate)
    def render_live_terminal():
        tick = engine.client.get_tick(engine.config.active_symbol)
        open_positions = engine.client.get_open_positions()

        col_chart, col_orderpad = st.columns([3, 1])

        with col_chart:
            # Fetch rates
            rates_df = engine.client.get_rates(engine.config.active_symbol, engine.config.active_timeframe, count=250)
            if not rates_df.empty:
                fig = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.03, row_heights=[0.75, 0.25]
                )

                # Candlestick
                fig.add_trace(
                    go.Candlestick(
                        x=rates_df.index,
                        open=rates_df["open"],
                        high=rates_df["high"],
                        low=rates_df["low"],
                        close=rates_df["close"],
                        name=engine.config.active_symbol,
                        increasing_line_color="#089981",
                        decreasing_line_color="#f23645",
                    ),
                    row=1, col=1,
                )

                # Technical indicator overlays based on selected strategy
                closes = rates_df["close"]
                if "EMA" in engine.config.strategy.name:
                    fast_p = engine.config.strategy.fast_period
                    slow_p = engine.config.strategy.slow_period
                    if fast_p == slow_p or "Price Cross" in engine.config.strategy.name:
                        single_ema = engine.strategy.ema(closes, fast_p)
                        fig.add_trace(go.Scatter(x=rates_df.index, y=single_ema, name=f"EMA {fast_p}", line=dict(color="#ffd700", width=2.2)), row=1, col=1)
                    else:
                        fast_ema = engine.strategy.ema(closes, fast_p)
                        slow_ema = engine.strategy.ema(closes, slow_p)
                        fig.add_trace(go.Scatter(x=rates_df.index, y=fast_ema, name=f"Fast EMA {fast_p}", line=dict(color="#2962ff", width=1.5)), row=1, col=1)
                        fig.add_trace(go.Scatter(x=rates_df.index, y=slow_ema, name=f"Slow EMA {slow_p}", line=dict(color="#ff9800", width=1.5)), row=1, col=1)

                elif engine.config.strategy.name == "RSI + Bollinger Bands":
                    upper_bb, mid_bb, lower_bb = engine.strategy.bollinger_bands(closes, engine.config.strategy.bb_period, engine.config.strategy.bb_std)
                    fig.add_trace(go.Scatter(x=rates_df.index, y=upper_bb, name="Upper BB", line=dict(color="#9c27b0", width=1, dash="dot")), row=1, col=1)
                    fig.add_trace(go.Scatter(x=rates_df.index, y=mid_bb, name="Mid BB", line=dict(color="#787b86", width=1)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=rates_df.index, y=lower_bb, name="Lower BB", line=dict(color="#9c27b0", width=1, dash="dot")), row=1, col=1)

                # Mark open positions on chart
                for p in open_positions:
                    if p.symbol == engine.config.active_symbol:
                        fig.add_hline(
                            y=p.price_open,
                            line_dash="dash",
                            line_color="#2962ff" if p.type == "BUY" else "#e91e63",
                            annotation_text=f"#{p.ticket} {p.type} @ {p.price_open}",
                            row=1, col=1,
                        )
                        if p.sl > 0:
                            fig.add_hline(y=p.sl, line_dash="dot", line_color="#f23645", annotation_text=f"SL #{p.ticket}", row=1, col=1)
                        if p.tp > 0:
                            fig.add_hline(y=p.tp, line_dash="dot", line_color="#089981", annotation_text=f"TP #{p.ticket}", row=1, col=1)

                # Volume sub-chart
                vol_col = "tick_volume" if "tick_volume" in rates_df.columns else "volume"
                if vol_col in rates_df.columns:
                    vol_colors = ["#089981" if c >= o else "#f23645" for c, o in zip(rates_df["close"], rates_df["open"])]
                    fig.add_trace(go.Bar(x=rates_df.index, y=rates_df[vol_col], marker_color=vol_colors, name="Volume"), row=2, col=1)

                bid_str = f"{tick.bid:.2f}" if tick else "N/A"
                ask_str = f"{tick.ask:.2f}" if tick else "N/A"
                fig.update_layout(
                    title=f"{engine.config.active_symbol} ({engine.config.active_timeframe}) | 🔴 BID: {bid_str} | 🟢 ASK: {ask_str}",
                    xaxis_rangeslider_visible=False,
                    height=560,
                    margin=dict(l=10, r=10, t=40, b=10),
                    template="plotly_dark",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Loading chart data... Check connection settings.")

        with col_orderpad:
            st.subheader("Manual Quick Order")
            spec = engine.client.get_symbol_info(engine.config.active_symbol)
            min_lot = spec.min_lot if spec else 0.01
            max_lot = spec.max_lot if spec else 10.0
            step_lot = spec.lot_step if spec else 0.01

            order_lots = st.number_input("Lots", min_value=min_lot, max_value=max_lot, value=min_lot, step=step_lot, key="order_lots_in")
            order_sl_pips = st.number_input("Stop Loss (pips)", min_value=0.0, value=25.0, step=5.0, key="order_sl_in")
            order_tp_pips = st.number_input("Take Profit (pips)", min_value=0.0, value=50.0, step=5.0, key="order_tp_in")

            # Quotes display
            if tick and spec:
                st.markdown(f"**Live Ask**: `{tick.ask}` | **Live Bid**: `{tick.bid}`")
                st.markdown(f"**Spread**: `{tick.spread_points}` points")
                col_buy, col_sell = st.columns(2)
                with col_buy:
                    if st.button("🟢 BUY (Ask)", use_container_width=True, type="primary", key="btn_quick_buy"):
                        sl_p = round(tick.ask - spec.pip_to_price(order_sl_pips), spec.digits) if order_sl_pips > 0 else None
                        tp_p = round(tick.ask + spec.pip_to_price(order_tp_pips), spec.digits) if order_tp_pips > 0 else None
                        res = engine.place_manual_order(engine.config.active_symbol, "BUY", order_lots, sl_p, tp_p)
                        if res.success:
                            st.success(f"BUY order executed #{res.order}")
                            st.rerun()
                        else:
                            st.error(f"Failed: {res.error_message}")

                with col_sell:
                    if st.button("🔴 SELL (Bid)", use_container_width=True, key="btn_quick_sell"):
                        sl_p = round(tick.bid + spec.pip_to_price(order_sl_pips), spec.digits) if order_sl_pips > 0 else None
                        tp_p = round(tick.bid - spec.pip_to_price(order_tp_pips), spec.digits) if order_tp_pips > 0 else None
                        res = engine.place_manual_order(engine.config.active_symbol, "SELL", order_lots, sl_p, tp_p)
                        if res.success:
                            st.success(f"SELL order executed #{res.order}")
                            st.rerun()
                        else:
                            st.error(f"Failed: {res.error_message}")

            # Last Strategy Signal box
            st.markdown("---")
            st.markdown("##### Latest Strategy Signal")
            if engine.last_signal:
                sig = engine.last_signal
                sig_badge = "🟢 BUY" if sig.signal_type == "BUY" else ("🔴 SELL" if sig.signal_type == "SELL" else "⚪ HOLD")
                st.markdown(f"**Signal**: {sig_badge}")
                st.caption(f"**Reason**: {sig.reason}")
                if sig.sl_price:
                    st.caption(f"**Suggested SL**: `{sig.sl_price}` | **TP**: `{sig.tp_price}`")
            else:
                st.caption("No signals evaluated yet. Start bot or select symbol.")

    render_live_terminal()


# =========================================================================
# TAB 2: POSITIONS & DEALS
# =========================================================================
with tab_positions:
    @st.fragment(run_every=refresh_rate)
    def render_positions_tab():
        open_positions = engine.client.get_open_positions()
        st.subheader("Open Positions")
        if open_positions:
            pos_data = []
            for p in open_positions:
                pos_data.append({
                    "Ticket": p.ticket,
                    "Symbol": p.symbol,
                    "Type": p.type,
                    "Volume": p.volume,
                    "Open Price": p.price_open,
                    "Current Price": p.price_current,
                    "SL": p.sl,
                    "TP": p.tp,
                    "Profit ($)": f"{p.profit:+,.2f}",
                    "Time": p.time.strftime("%H:%M:%S") if isinstance(p.time, datetime) else str(p.time),
                })
            st.dataframe(pd.DataFrame(pos_data), use_container_width=True)

            col_close1, col_close2 = st.columns([2, 1])
            with col_close1:
                tickets_list = [p.ticket for p in open_positions]
                sel_ticket = st.selectbox("Select Ticket to Close", options=tickets_list, key="ticket_sel_close")
            with col_close2:
                st.write("")
                st.write("")
                if st.button("Close Position", type="primary", key="btn_close_single"):
                    res = engine.client.close_position(sel_ticket)
                    if res.success:
                        st.success(f"Position #{sel_ticket} closed successfully.")
                        st.rerun()
                    else:
                        st.error(f"Close failed: {res.error_message}")
        else:
            st.info("No open positions currently.")

        st.markdown("---")
        st.subheader("Closed Deals History")
        deals = engine.client.get_history_deals(days=7)
        if deals:
            deals_df = pd.DataFrame(deals)
            st.dataframe(deals_df, use_container_width=True)
        else:
            st.info("No deals recorded yet.")

    render_positions_tab()


# =========================================================================
# TAB 3: STRATEGY SETTINGS
# =========================================================================
with tab_strategy:
    st.subheader(f"Strategy Configuration: {engine.config.strategy.name}")

    with st.form("strategy_config_form"):
        col_s1, col_s2 = st.columns(2)

        with col_s1:
            fast_p = st.number_input("Fast EMA Period", value=engine.config.strategy.fast_period, min_value=2, max_value=200)
            slow_p = st.number_input("Slow EMA Period", value=engine.config.strategy.slow_period, min_value=5, max_value=500)
            trend_p = st.number_input("Trend Filter EMA Period", value=engine.config.strategy.trend_filter_period, min_value=20, max_value=1000)
            rsi_p = st.number_input("RSI Period", value=engine.config.strategy.rsi_period, min_value=2, max_value=100)
            rsi_os = st.slider("RSI Oversold Level", min_value=10.0, max_value=45.0, value=float(engine.config.strategy.rsi_oversold))
            rsi_ob = st.slider("RSI Overbought Level", min_value=55.0, max_value=90.0, value=float(engine.config.strategy.rsi_overbought))

        with col_s2:
            bb_p = st.number_input("Bollinger Bands Period", value=engine.config.strategy.bb_period, min_value=5, max_value=100)
            bb_std_val = st.number_input("Bollinger Bands Std Dev", value=float(engine.config.strategy.bb_std), min_value=1.0, max_value=4.0, step=0.1)
            macd_f = st.number_input("MACD Fast", value=engine.config.strategy.macd_fast, min_value=2, max_value=50)
            macd_s = st.number_input("MACD Slow", value=engine.config.strategy.macd_slow, min_value=10, max_value=100)
            macd_sig = st.number_input("MACD Signal", value=engine.config.strategy.macd_signal, min_value=2, max_value=50)
            donch_p = st.number_input("Donchian Period", value=engine.config.strategy.donchian_period, min_value=5, max_value=100)

        st.markdown("##### Stop Loss & Target Settings")
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            sl_pips_in = st.number_input("Stop Loss (pips)", value=float(engine.config.strategy.sl_pips), min_value=1.0, max_value=500.0)
        with col_t2:
            tp_pips_in = st.number_input("Take Profit (pips)", value=float(engine.config.strategy.tp_pips), min_value=1.0, max_value=1000.0)
        with col_t3:
            magic_in = st.number_input("Strategy Magic Number", value=int(engine.config.strategy.magic_number), min_value=1000)

        use_atr_in = st.checkbox(
            "Use ATR Volatility Stops (Dynamic based on 14-candle market volatility)",
            value=bool(getattr(engine.config.strategy, "use_atr_stops", False)),
            help="If checked, SL and TP adjust automatically to market volatility using ATR. If unchecked, exact Fixed Pips above are used.",
        )
        col_atr1, col_atr2 = st.columns(2)
        with col_atr1:
            atr_sl_in = st.number_input("ATR SL Multiplier", value=float(getattr(engine.config.strategy, "atr_sl_mult", 1.5)), min_value=0.5, max_value=5.0, step=0.1)
        with col_atr2:
            atr_tp_in = st.number_input("ATR TP Multiplier", value=float(getattr(engine.config.strategy, "atr_tp_mult", 2.5)), min_value=0.5, max_value=10.0, step=0.1)

        if st.form_submit_button("💾 Save Strategy Settings", type="primary"):
            engine.config.strategy.fast_period = int(fast_p)
            engine.config.strategy.slow_period = int(slow_p)
            engine.config.strategy.trend_filter_period = int(trend_p)
            engine.config.strategy.rsi_period = int(rsi_p)
            engine.config.strategy.rsi_oversold = float(rsi_os)
            engine.config.strategy.rsi_overbought = float(rsi_ob)
            engine.config.strategy.bb_period = int(bb_p)
            engine.config.strategy.bb_std = float(bb_std_val)
            engine.config.strategy.macd_fast = int(macd_f)
            engine.config.strategy.macd_slow = int(macd_s)
            engine.config.strategy.macd_signal = int(macd_sig)
            engine.config.strategy.donchian_period = int(donch_p)
            engine.config.strategy.sl_pips = float(sl_pips_in)
            engine.config.strategy.tp_pips = float(tp_pips_in)
            engine.config.strategy.use_atr_stops = bool(use_atr_in)
            engine.config.strategy.atr_sl_mult = float(atr_sl_in)
            engine.config.strategy.atr_tp_mult = float(atr_tp_in)
            engine.config.strategy.magic_number = int(magic_in)

            save_config(engine.config)
            engine.update_config(engine.config)
            st.success("Strategy settings successfully saved and applied to active bot!")


# =========================================================================
# TAB 4: RISK MANAGEMENT
# =========================================================================
with tab_risk:
    st.subheader("Risk & Capital Protection Parameters")

    with st.form("risk_config_form"):
        col_r1, col_r2 = st.columns(2)

        with col_r1:
            max_risk = st.slider("Risk Per Trade (% of Equity)", min_value=0.2, max_value=5.0, value=float(engine.config.risk.max_risk_pct), step=0.1)
            use_dyn_lot = st.checkbox("Calculate Lot Dynamically based on SL Distance", value=engine.config.risk.use_dynamic_lot)
            fix_lot = st.number_input("Fallback Fixed Lot Size", min_value=0.01, max_value=10.0, value=float(engine.config.risk.fixed_lot), step=0.01)
            max_pos = st.number_input("Maximum Simultaneous Open Positions", min_value=1, max_value=20, value=int(engine.config.risk.max_open_positions))

        with col_r2:
            daily_loss = st.number_input("Circuit Breaker: Max Daily Loss ($)", min_value=10.0, max_value=10000.0, value=float(engine.config.risk.max_daily_loss), step=50.0)
            max_spread = st.number_input("Max Spread Filter (points)", min_value=5, max_value=100, value=int(engine.config.risk.max_spread_points))
            use_ts = st.checkbox("Enable Trailing Stop Loss", value=engine.config.risk.use_trailing_stop)
            ts_pips = st.number_input("Trailing Stop Distance (pips)", min_value=5.0, max_value=100.0, value=float(engine.config.risk.trailing_stop_pips))

        if st.form_submit_button("💾 Save Risk Settings", type="primary"):
            engine.config.risk.max_risk_pct = float(max_risk)
            engine.config.risk.use_dynamic_lot = use_dyn_lot
            engine.config.risk.fixed_lot = float(fix_lot)
            engine.config.risk.max_open_positions = int(max_pos)
            engine.config.risk.max_daily_loss = float(daily_loss)
            engine.config.risk.max_spread_points = int(max_spread)
            engine.config.risk.use_trailing_stop = use_ts
            engine.config.risk.trailing_stop_pips = float(ts_pips)

            save_config(engine.config)
            engine.update_config(engine.config)
            st.success("Risk settings updated!")


# =========================================================================
# TAB 5: BACKTESTING LAB
# =========================================================================
with tab_backtest:
    st.subheader("Historical Quantitative Strategy Backtester")

    col_bt1, col_bt2, col_bt3, col_bt4 = st.columns(4)
    with col_bt1:
        bt_sym = st.selectbox("Backtest Symbol", options=engine.config.tracked_symbols, index=0)
    with col_bt2:
        bt_strat_name = st.selectbox("Strategy to Backtest", options=list(AVAILABLE_STRATEGIES.keys()))
    with col_bt3:
        bt_capital = st.number_input("Starting Capital ($)", value=10000.0, step=1000.0)
    with col_bt4:
        bt_bars = st.slider("Historical Bars", min_value=100, max_value=500, value=250, step=50)

    if st.button("🧪 Run Backtest", type="primary"):
        with st.spinner("Executing simulation backtest..."):
            hist_df = engine.client.get_rates(bt_sym, "M5", count=bt_bars)
            spec = engine.client.get_symbol_info(bt_sym)
            strat_cls = AVAILABLE_STRATEGIES[bt_strat_name]
            strat_instance = strat_cls(engine.config.strategy.model_dump())

            bt_engine = BacktestEngine(
                strategy=strat_instance,
                symbol_info=spec,
                initial_capital=bt_capital,
                risk_pct=engine.config.risk.max_risk_pct,
            )
            result = bt_engine.run(hist_df)

            # Display Backtest Metrics
            m = result.metrics
            m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
            with m_col1:
                st.metric("Net Profit", f"${m.get('Net Profit ($)', 0):+,.2f}")
            with m_col2:
                st.metric("Total Return", f"{m.get('Total Return (%)', 0):+.2f}%")
            with m_col3:
                st.metric("Win Rate", f"{m.get('Win Rate (%)', 0):.1f}%")
            with m_col4:
                st.metric("Profit Factor", f"{m.get('Profit Factor', 0):.2f}")
            with m_col5:
                st.metric("Max Drawdown", f"{m.get('Max Drawdown (%)', 0):.2f}%")

            # Equity Curve Plot
            if not result.equity_curve.empty:
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(
                    x=result.equity_curve.index,
                    y=result.equity_curve["equity"],
                    mode="lines",
                    name="Portfolio Equity",
                    line=dict(color="#089981", width=2),
                ))
                fig_eq.add_trace(go.Scatter(
                    x=result.equity_curve.index,
                    y=result.equity_curve["balance"],
                    mode="lines",
                    name="Realized Balance",
                    line=dict(color="#2962ff", width=1.5, dash="dash"),
                ))
                fig_eq.update_layout(
                    title=f"Equity Growth Curve: {bt_strat_name} on {bt_sym}",
                    template="plotly_dark",
                    height=420,
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig_eq, use_container_width=True)

            # Trades Table
            st.markdown("##### Executed Backtest Trades")
            if not result.trades_df.empty:
                st.dataframe(result.trades_df, use_container_width=True)
            else:
                st.info("No trades were generated under these market conditions during this window.")


# =========================================================================
# TAB 6: LOGS & MT5 SETUP
# =========================================================================
with tab_logs:
    col_l1, col_l2 = st.columns([2, 1])

    with col_l1:
        st.subheader("Live Activity & Audit Log")
        log_entries = list(engine.activity_log)
        if log_entries:
            for item in reversed(log_entries[-40:]):
                lvl = item["level"]
                color = "#089981" if lvl == "INFO" else ("#f59e0b" if lvl == "WARNING" else "#f23645")
                st.markdown(
                    f"<span style='color: #787b86;'>[{item['time']}]</span> "
                    f"<strong style='color: {color};'>[{lvl}]</strong> "
                    f"<span>{item['message']}</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No log entries yet.")

    with col_l2:
        st.subheader("MetaTrader 5 Connection Settings")
        with st.form("mt5_conn_form"):
            login_val = st.text_input("Account Login", value=str(engine.config.mt5.login or ""))
            pw_val = st.text_input("Account Password", value=str(engine.config.mt5.password or ""), type="password")
            server_val = st.text_input("Broker Server", value=str(engine.config.mt5.server or ""))
            path_val = st.text_input("MT5 Terminal Path (optional)", value=str(engine.config.mt5.path or ""))

            col_cf1, col_cf2 = st.columns(2)
            with col_cf1:
                save_cred_btn = st.form_submit_button("💾 Save Credentials", type="primary")
            with col_cf2:
                test_cred_btn = st.form_submit_button("🔌 Test Connection")

            if save_cred_btn or test_cred_btn:
                engine.config.mt5.login = int(login_val) if login_val.strip().isdigit() else None
                engine.config.mt5.password = pw_val if pw_val.strip() else None
                engine.config.mt5.server = server_val if server_val.strip() else None
                engine.config.mt5.path = path_val if path_val.strip() else None

                save_config(engine.config)
                engine.update_config(engine.config)

                # Test connection
                connected = engine.client.connect()
                acc_info = engine.client.get_account_info() if connected else None

                if connected and acc_info:
                    st.success(
                        f"✅ Connected to MT5! Account #{acc_info.login} ({acc_info.server}) "
                        f"| Balance: ${acc_info.balance:,.2f} {acc_info.currency}"
                    )
                    st.rerun()
                else:
                    st.error(
                        "❌ Could not connect to MT5. Please verify MT5 terminal is open on your PC "
                        "and credentials match your broker account."
                    )

        st.markdown("---")
        st.markdown(
            """
            **How to connect to your Live MT5 broker**:
            1. Open MetaTrader 5 desktop terminal.
            2. In MT5, click **Tools** -> **Options** -> **Expert Advisors**.
            3. Check **Allow algorithmic trading**.
            4. Click the **Algo Trading** icon in the toolbar (it will turn green).
            5. Switch the Execution Engine in the sidebar to **Live MT5 Connection**.
            """
        )
