"""app.py — HJB Stochastic Control Engine · Streamlit Dashboard."""

from __future__ import annotations
import os
from io import StringIO
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="HJB Stochastic Control · P2Quant",
                   layout="wide", page_icon="📐")

HF_TOKEN = os.environ.get("HF_TOKEN")
BASE_RAW = f"https://huggingface.co/datasets/{config.HF_OUTPUT_REPO}/resolve/main"
BASE_API = f"https://huggingface.co/api/datasets/{config.HF_OUTPUT_REPO}/tree/main"
HEADERS  = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
PALETTE  = ["#1B4F8A","#27AE60","#E74C3C","#F39C12","#8E44AD","#148F77",
            "#CA6F1E","#2471A3","#CB4335","#1A5276","#117A65","#B7950B",
            "#884EA0","#1F618D","#B9770E","#922B21"]

def score_colour(v):
    if v >= 0.5: return "#1D9E75"
    if v >= 0.0: return "#82C3A9"
    if v >= -0.5: return "#F0A07A"
    return "#E74C3C"

def fmt(v, d=4): return f"{v:+.{d}f}"

@st.cache_data(ttl=3600, show_spinner="Loading HJB results…")
def load_json(universe):
    slug = universe.lower().replace("_","-")
    try:
        r = requests.get(BASE_API, headers=HEADERS, timeout=30)
        if r.status_code != 200: return None
        files   = sorted(f["path"] for f in r.json() if f["path"].endswith(".json"))
        matches = [f for f in files if f"_{slug}.json" in f]
        if not matches: return None
        resp = requests.get(f"{BASE_RAW}/{matches[-1]}", headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception: return None

@st.cache_data(ttl=3600, show_spinner="Loading history…")
def load_csv(filename):
    try:
        r = requests.get(f"{BASE_RAW}/{filename}", headers=HEADERS, timeout=60)
        if r.status_code != 200: return None
        df = pd.read_csv(StringIO(r.text), index_col=0, parse_dates=True)
        return df if not df.empty else None
    except Exception: return None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    universe = st.selectbox("Universe", list(config.UNIVERSES.keys()))
    gamma_display = st.slider("Risk aversion γ (display only)", 1.0, 8.0,
                               float(config.GAMMA), 0.5,
                               help="γ=1 log utility, γ=3 moderate, γ=5 conservative")
    st.divider()
    st.markdown(f"**Horizon:** {config.HORIZON_DAYS}d")
    st.markdown(f"**Grid:** {config.NV} × {config.NT} (v × t)")
    st.markdown(f"**Heston window:** {config.HESTON_WINDOW}d")
    st.markdown(f"**Refit every:** {config.HESTON_REFIT_FREQ}d")
    st.markdown(f"**Myopic wt:** {config.MYOPIC_WT}")
    st.markdown(f"**Hedging wt:** {config.HEDGING_WT}")
    st.markdown(f"**Stress VIX:** {config.VIX_STRESS_THRESHOLD}")
    st.markdown(f"**OOS from:** {config.OOS_START}")
    st.markdown(f"**Next trading day:** {next_trading_day()}")
    st.divider()
    st.markdown("**Score formula:**")
    st.code("w* = myopic + hedging\n"
            "myopic  = μ/(γ·σ·v)\n"
            "hedging = -(fᵥ/f)·ρ·ξ/(γ·σ)", language="python")
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 📐 HJB Stochastic Control Engine")
st.caption(
    "Heston stochastic volatility → HJB PDE (Crank-Nicolson finite differences) → "
    "Optimal weights = myopic demand + hedging demand · "
    "Only engine with explicit vol-hedging allocation"
)

slug       = universe.lower().replace("_","-")
data       = load_json(universe)
daily_df   = load_csv(f"daily_{slug}.csv")
score_df   = load_csv(f"scores_{slug}.csv")
myopic_df  = load_csv(f"myopic_{slug}.csv")
hedging_df = load_csv(f"hedging_{slug}.csv")
ranking_df = load_csv(f"rankings_{slug}.csv")

if data is None:
    st.warning("⚠️ No results found. Run `python trainer.py` first.")
    st.stop()

latest_scores = data.get("latest_scores", {})
latest_ranked = data.get("latest_ranked", [])
latest_date   = data.get("latest_date", "?")
run_date      = data.get("run_date", "?")
cfg           = data.get("config", {})
v_grid        = np.array(data.get("v_grid", []))
f_surface     = np.array(data.get("f_surface", []))

# ── KPI row ───────────────────────────────────────────────────────────────────
k1,k2,k3,k4 = st.columns(4)
k1.metric("Run Date",    run_date)
k2.metric("Latest Date", latest_date)
k3.metric("Universe",    universe)
k4.metric("ETFs Scored", len(latest_scores))

if latest_ranked:
    top  = latest_ranked[0]
    cash = top.get("composite_score",0) < config.CASH_THRESHOLD
    regime = "?"
    if daily_df is not None and "regime" in daily_df.columns:
        regime = str(daily_df["regime"].iloc[-1])
    vix_now = float(daily_df["vix"].iloc[-1]) if daily_df is not None and "vix" in daily_df.columns else 0.0

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("🏆 Top Pick", "CASH" if cash else top["ticker"])
    m2.metric("Top Score", fmt(top.get("composite_score",0)))
    m3.metric("VIX Regime", f"{regime} ({vix_now:.1f})")
    m4.metric("CASH Signal", "Yes ⚠️" if cash else "No ✅")

st.divider()

tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "🎯 Rankings & Scores",
    "🔬 Value Function f(v)",
    "⚖️ Myopic vs Hedging",
    "📈 Score History",
    "📋 Full Heston Table",
])

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Rankings & Scores
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader(f"HJB Rankings as of {latest_date}")
    tickers_r = [r["ticker"] for r in latest_ranked]
    scores_r  = [r.get("composite_score",0)  for r in latest_ranked]
    myopic_r  = [r.get("myopic_demand",0)    for r in latest_ranked]
    hedging_r = [r.get("hedging_demand",0)   for r in latest_ranked]
    colours_r = [score_colour(s) for s in scores_r]

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Composite HJB Score**")
        fig = go.Figure(go.Bar(y=tickers_r, x=scores_r, orientation="h",
            marker_color=colours_r,
            text=[fmt(s) for s in scores_r], textposition="outside"))
        fig.add_vline(x=0, line_dash="dot", line_color="gray")
        fig.update_layout(
            title="Score = 0.60×myopic_z + 0.40×hedging_z",
            xaxis_title="Composite z-score",
            yaxis=dict(autorange="reversed"),
            height=max(300,len(tickers_r)*30),
            margin=dict(t=50,b=20,l=60,r=80),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, key="rank_bar")

    with col_r:
        st.markdown("**Myopic vs Hedging Demand (stacked)**")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Myopic demand", y=tickers_r, x=myopic_r,
            orientation="h", marker_color="#1B4F8A"))
        fig2.add_trace(go.Bar(name="Hedging demand", y=tickers_r, x=hedging_r,
            orientation="h", marker_color="#E74C3C"))
        fig2.add_vline(x=0, line_dash="dot", line_color="gray")
        fig2.update_layout(
            barmode="relative",
            title="w* = myopic + hedging (red = vol-hedge allocation)",
            xaxis_title="Optimal weight",
            yaxis=dict(autorange="reversed"),
            height=max(300,len(tickers_r)*30),
            margin=dict(t=50,b=20,l=60,r=80),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig2, use_container_width=True, key="myopic_hedging_bar")

    st.markdown(f"### 🎯 Top {config.TOP_N} for {next_trading_day()}")
    cols = st.columns(config.TOP_N)
    for i, row in enumerate(latest_ranked[:config.TOP_N]):
        with cols[i]:
            sc  = row.get("composite_score",0)
            my  = row.get("myopic_demand",0)
            hed = row.get("hedging_demand",0)
            bg  = score_colour(sc)
            st.markdown(
                f"**#{i+1} {row['ticker']}**\n\n"
                f"Score: `{fmt(sc)}`\n\n"
                f"Myopic: `{fmt(my)}`\n\n"
                f"Hedging: `{fmt(hed)}`\n\n"
                f'<span style="background:{bg};color:white;padding:2px 8px;'
                f'border-radius:8px;font-size:11px">Rank #{row.get("rank",i+1)}</span>',
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Value Function
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Value Function f(t=0, v) — HJB PDE Solution")
    st.caption(
        "f(v) is the solution to the reduced HJB PDE. "
        "V(t,W,v) = W^(1-γ)/(1-γ) × f(t,v). "
        "f is decreasing in v — higher volatility reduces investor welfare. "
        "f_v/f drives the hedging demand: where f drops steeply, hedge more aggressively."
    )
    if len(v_grid) > 0 and len(f_surface) > 0:
        vix_equiv = np.sqrt(v_grid) * 100  # convert variance to VIX equivalent

        col_a, col_b = st.columns(2)
        with col_a:
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=vix_equiv, y=f_surface, mode="lines",
                name="f(v)", line=dict(color="#1B4F8A", width=2)))
            # Mark current VIX
            if daily_df is not None and "vix" in daily_df.columns:
                vix_cur = float(daily_df["vix"].iloc[-1])
                f_cur   = float(np.interp(vix_cur**2/1e4, v_grid, f_surface))
                fig_f.add_trace(go.Scatter(x=[vix_cur], y=[f_cur], mode="markers",
                    name=f"Today (VIX={vix_cur:.1f})",
                    marker=dict(color="#E74C3C", size=10)))
            fig_f.add_vline(x=config.VIX_STRESS_THRESHOLD, line_dash="dash",
                            line_color="#F39C12", annotation_text="Stress threshold")
            fig_f.update_layout(
                title="Value function f(v) — decreasing in volatility",
                xaxis_title="VIX equivalent (%)",
                yaxis_title="f(v)",
                height=350, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_f, use_container_width=True, key="vf_plot")

        with col_b:
            # fv/f ratio — drives hedging demand
            fv = np.gradient(f_surface, v_grid)
            fv_over_f = fv / (f_surface + 1e-10)
            fig_fv = go.Figure()
            fig_fv.add_trace(go.Scatter(x=vix_equiv, y=fv_over_f, mode="lines",
                name="f_v/f", line=dict(color="#8E44AD", width=2)))
            fig_fv.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_fv.update_layout(
                title="f_v/f — hedging demand multiplier (more negative = hedge more)",
                xaxis_title="VIX equivalent (%)",
                yaxis_title="f_v / f",
                height=350, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_fv, use_container_width=True, key="fv_plot")

        st.info(
            "**How to read this:** When VIX is high, f_v/f is more negative → "
            "hedging demand = -(f_v/f) × ρ × ξ / (γ × σ) is larger positive for "
            "ETFs with ρ < 0 (TLT, GLD). The engine automatically increases their "
            "allocation when VIX spikes — not because their returns improved, "
            "but because they hedge the investor's exposure to rising volatility."
        )
    else:
        st.info("Value function data not available.")

    # Interactive: w*(v) as function of VIX for selected ETF
    st.markdown("**Optimal weight w*(v) vs VIX — how allocation shifts with volatility**")
    if latest_ranked and len(v_grid) > 0:
        sel_etf = st.selectbox("Select ETF", [r["ticker"] for r in latest_ranked], key="vf_etf")
        sel_data = next((r for r in latest_ranked if r["ticker"] == sel_etf), None)
        if sel_data:
            my_val  = sel_data.get("myopic_demand", 0)
            hed_val = sel_data.get("hedging_demand", 0)
            fv_over_f_arr = np.gradient(f_surface, v_grid) / (f_surface + 1e-10)
            # Reconstruct w*(v) across grid (approximate: hedging demand varies with fv/f)
            w_myopic_v  = my_val * (v_grid.mean() / (v_grid + 1e-6))   # scales with 1/v
            w_hedging_v = hed_val * (fv_over_f_arr / (fv_over_f_arr.mean() + 1e-10))
            w_total_v   = w_myopic_v + w_hedging_v

            fig_wv = go.Figure()
            fig_wv.add_trace(go.Scatter(x=vix_equiv, y=w_myopic_v, mode="lines",
                name="Myopic demand", line=dict(color="#1B4F8A", width=1.5, dash="dot")))
            fig_wv.add_trace(go.Scatter(x=vix_equiv, y=w_hedging_v, mode="lines",
                name="Hedging demand", line=dict(color="#E74C3C", width=1.5, dash="dot")))
            fig_wv.add_trace(go.Scatter(x=vix_equiv, y=w_total_v, mode="lines",
                name="Total w*(v)", line=dict(color="#1D9E75", width=2)))
            fig_wv.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_wv.update_layout(
                title=f"w*(v) vs VIX for {sel_etf}",
                xaxis_title="VIX equivalent (%)", yaxis_title="Optimal weight",
                height=320, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig_wv, use_container_width=True, key="wv_plot")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Myopic vs Hedging
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Myopic vs Hedging Demand Over Time")
    st.caption(
        "**Myopic demand** = μ/(γ·σ·v) — pure Merton ratio, scales inversely with volatility. "
        "**Hedging demand** = -(f_v/f)·ρ·ξ/(γ·σ) — extra allocation to hedge vol risk. "
        "Hedging demand is positive for ETFs with ρ < 0 and spikes when VIX is high."
    )
    if myopic_df is not None and hedging_df is not None:
        etf_cols = [c for c in myopic_df.columns if c in config.UNIVERSES[universe]]
        sel = st.multiselect("Select ETFs", etf_cols, default=etf_cols[:4], key="mh_sel")
        period = st.radio("Period", ["Last 2 years","Last 5 years","Full OOS"],
                          horizontal=True, key="mh_period")
        df_my  = myopic_df.copy()
        df_hed = hedging_df.copy()
        if period == "Last 2 years":
            df_my  = df_my[df_my.index   >= "2024-01-01"]
            df_hed = df_hed[df_hed.index >= "2024-01-01"]
        elif period == "Last 5 years":
            df_my  = df_my[df_my.index   >= "2021-01-01"]
            df_hed = df_hed[df_hed.index >= "2021-01-01"]

        if sel:
            col_m, col_h = st.columns(2)
            with col_m:
                fig_m = go.Figure()
                for i,tkr in enumerate(sel):
                    if tkr in df_my.columns:
                        fig_m.add_trace(go.Scatter(x=df_my.index, y=df_my[tkr],
                            mode="lines", name=tkr,
                            line=dict(width=1.4, color=PALETTE[i%len(PALETTE)])))
                fig_m.add_hline(y=0, line_dash="dot", line_color="gray")
                fig_m.update_layout(title="Myopic demand μ/(γ·σ·v)",
                    yaxis_title="Myopic weight", height=360,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02))
                st.plotly_chart(fig_m, use_container_width=True, key="myopic_ts")

            with col_h:
                fig_h = go.Figure()
                for i,tkr in enumerate(sel):
                    if tkr in df_hed.columns:
                        fig_h.add_trace(go.Scatter(x=df_hed.index, y=df_hed[tkr],
                            mode="lines", name=tkr,
                            line=dict(width=1.4, color=PALETTE[i%len(PALETTE)])))
                fig_h.add_hline(y=0, line_dash="dot", line_color="gray")
                fig_h.update_layout(title="Hedging demand -(f_v/f)·ρ·ξ/(γ·σ)",
                    yaxis_title="Hedging weight", height=360,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02))
                st.plotly_chart(fig_h, use_container_width=True, key="hedging_ts")

        # VIX overlay with regime
        if daily_df is not None and "vix" in daily_df.columns:
            fig_vix = go.Figure()
            stress = daily_df[daily_df["regime"]=="STRESS"] if "regime" in daily_df.columns else pd.DataFrame()
            normal = daily_df[daily_df["regime"]=="NORMAL"] if "regime" in daily_df.columns else daily_df
            fig_vix.add_trace(go.Scatter(x=daily_df.index, y=daily_df["vix"],
                mode="lines", name="VIX", line=dict(color="#1B4F8A", width=1.2),
                fill="tozeroy", fillcolor="rgba(27,79,138,0.07)"))
            fig_vix.add_hline(y=config.VIX_STRESS_THRESHOLD, line_dash="dash",
                              line_color="#E74C3C",
                              annotation_text=f"Stress threshold ({config.VIX_STRESS_THRESHOLD})")
            fig_vix.update_layout(title="VIX over time — stress regime boosts hedging demand",
                yaxis_title="VIX", height=280,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_vix, use_container_width=True, key="vix_ts")
    else:
        st.info("No myopic/hedging history found.")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Score History
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Composite Score History")
    if score_df is not None:
        etf_cols_s = [c for c in score_df.columns if c in config.UNIVERSES[universe]]
        sel_s = st.multiselect("Select ETFs", etf_cols_s,
                               default=etf_cols_s[:6], key="score_sel")
        period_s = st.radio("Period", ["Last 2 years","Last 5 years","Full OOS"],
                            horizontal=True, key="score_period")
        df_s = score_df.copy()
        if period_s == "Last 2 years":   df_s = df_s[df_s.index >= "2024-01-01"]
        elif period_s == "Last 5 years": df_s = df_s[df_s.index >= "2021-01-01"]

        if sel_s:
            fig_s = go.Figure()
            for i,tkr in enumerate(sel_s):
                if tkr in df_s.columns:
                    fig_s.add_trace(go.Scatter(x=df_s.index, y=df_s[tkr],
                        mode="lines", name=tkr,
                        line=dict(width=1.4, color=PALETTE[i%len(PALETTE)])))
            fig_s.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_s.update_layout(title="HJB composite score over time",
                yaxis_title="Score", height=400,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig_s, use_container_width=True, key="score_ts")

        recent_s = score_df[etf_cols_s].tail(252)
        fig_sh = go.Figure(go.Heatmap(
            z=recent_s.values.T,
            x=recent_s.index.strftime("%Y-%m-%d"),
            y=list(recent_s.columns),
            colorscale="RdYlGn", zmid=0, colorbar=dict(title="Score")))
        fig_sh.update_layout(title="Score heatmap — last 252 days",
            height=max(300,len(recent_s.columns)*22+80),
            margin=dict(t=40,b=60,l=60,r=20),
            xaxis=dict(tickangle=-45,nticks=12),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_sh, use_container_width=True, key="score_heat")

        if daily_df is not None and "top_ticker" in daily_df.columns:
            picks = daily_df["top_ticker"].value_counts()
            fig_f = go.Figure(go.Bar(x=picks.index, y=picks.values,
                marker_color="#1B4F8A", text=picks.values, textposition="outside"))
            fig_f.update_layout(title="Top-pick frequency (OOS)",
                yaxis_title="Days as #1 HJB pick", height=280,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_f, use_container_width=True, key="pick_freq")
    else:
        st.info("No score history found.")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 5 — Full Heston Table
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader(f"Full HJB Parameters — {latest_date}")
    if latest_ranked:
        rows = []
        for i,row in enumerate(latest_ranked):
            hp = row.get("heston",{})
            rows.append({
                "Rank":          i+1,
                "Ticker":        row["ticker"],
                "Composite":     fmt(row.get("composite_score",0)),
                "Myopic w*":     fmt(row.get("myopic_demand",0)),
                "Hedging w*":    fmt(row.get("hedging_demand",0)),
                "κ (mean rev)":  f"{hp.get('kappa',0):.3f}",
                "θ (long-run v)":f"{hp.get('theta',0):.5f}",
                "ξ (vol-of-vol)":f"{hp.get('xi',0):.3f}",
                "ρ (ret-vol)":   f"{hp.get('rho',0):+.3f}",
                "μ (ann drift)": f"{hp.get('mu',0):+.3f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True, height=600)

        st.info(
            "**Key:** κ = how fast variance reverts to θ. "
            "θ = long-run variance (√θ × 100 ≈ long-run vol %). "
            "ξ = vol-of-vol (how much variance itself fluctuates). "
            "ρ = return-variance correlation (negative for most equities). "
            "Hedging demand is positive when ρ < 0 and VIX is elevated."
        )

    st.divider()
    st.markdown("**Engine Configuration**")
    cfg_rows = [{"Parameter":k,"Value":str(v)} for k,v in cfg.items()]
    st.dataframe(pd.DataFrame(cfg_rows), use_container_width=True,
                 hide_index=True, height=320)

    if daily_df is not None:
        st.divider()
        st.markdown("**Daily summary (last 20 days)**")
        st.dataframe(daily_df.tail(20), use_container_width=True)

    st.divider()
    st.caption(
        f"P2Quant HJB Engine · Run: {run_date} · "
        f"Heston + HJB PDE (Crank-Nicolson) · "
        f"Myopic + hedging demand decomposition · "
        f"Data: {config.HF_DATA_REPO}"
    )
