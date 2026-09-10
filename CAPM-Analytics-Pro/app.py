from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.beta import estimate_beta, prepare_returns, rolling_beta
from core.capm import calculate_capm
from core.data_quality import score_beta_quality
from core.scenarios import scenario_table, sensitivity_table
from data.damodaran import country_row, find_column, load_country_risk, load_global_beta
from data.risk_free import fetch_us_10y
from data.yahoo_finance import fetch_history, fetch_snapshot, normalize_ticker
from reports.pdf_report import build_pdf
from utils.industry import suggested_industries

st.set_page_config(
    page_title="CAPM Analytics Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# UI DESIGN SYSTEM
# Avoids custom DOM trees and explicit reruns. This is intentionally robust for
# Streamlit's React frontend and easier to maintain than heavy HTML injection.
# -----------------------------------------------------------------------------
NAVY = "#10233F"
BLUE = "#1F4E79"
RED = "#C62828"
INK = "#172033"
MUTED = "#667085"
LINE = "#E4E7EC"
SURFACE = "#FFFFFF"
BG = "#F5F7FB"
GREEN = "#087443"
AMBER = "#9A6700"

st.markdown(
    f"""
    <style>
    .stApp {{ background: {BG}; }}
    .block-container {{ max-width: 1450px; padding-top: 1.1rem; padding-bottom: 2.5rem; }}
    [data-testid="stSidebar"] {{ background: {NAVY}; }}
    [data-testid="stSidebar"] * {{ color: #F8FAFC !important; }}
    [data-testid="stSidebar"] .stCaption {{ color: #CBD5E1 !important; }}
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] [data-baseweb="select"] > div {{
        background: #172B4D !important; border-color: #34506F !important;
    }}
    .brand-box {{
        border: 1px solid {LINE}; background: {SURFACE}; border-radius: 18px;
        padding: 14px 16px; margin-bottom: 14px;
        box-shadow: 0 5px 18px rgba(16,35,63,.06);
    }}
    .brand-title {{ font-size: 1.18rem; font-weight: 800; color: {NAVY}; margin-bottom: 2px; }}
    .brand-sub {{ color: {MUTED}; font-size: .78rem; }}
    .hero {{
        background: linear-gradient(135deg, {NAVY}, #1B416A);
        border-radius: 20px; padding: 28px 30px; color: white;
        box-shadow: 0 12px 30px rgba(16,35,63,.15); margin-bottom: 16px;
    }}
    .hero-kicker {{ color: #C9D7E8; font-size: .75rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }}
    .hero-title {{ color: white; font-size: 2.15rem; font-weight: 850; line-height: 1.1; margin: 6px 0 8px; }}
    .hero-text {{ color: #DDE7F2; max-width: 900px; line-height: 1.55; margin-bottom: 16px; }}
    .formula {{
        display: inline-block; padding: 11px 14px; border-radius: 12px;
        background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.18);
        font-weight: 800;
    }}
    .section-title {{ color: {NAVY}; font-size: 1.18rem; font-weight: 800; margin: 16px 0 5px; }}
    .section-sub {{ color: {MUTED}; font-size: .82rem; margin-bottom: 10px; }}
    .card {{
        background: {SURFACE}; border: 1px solid {LINE}; border-radius: 15px;
        padding: 14px 16px; min-height: 102px;
        box-shadow: 0 4px 14px rgba(16,35,63,.04);
    }}
    .metric-label {{ color: {MUTED}; font-size: .73rem; font-weight: 800; text-transform: uppercase; letter-spacing: .04em; }}
    .metric-value {{ color: {NAVY}; font-size: 1.48rem; font-weight: 850; margin-top: 7px; }}
    .metric-help {{ color: {MUTED}; font-size: .71rem; margin-top: 4px; }}
    .callout {{
        background: #F0F6FB; border-left: 4px solid {BLUE};
        padding: 12px 14px; border-radius: 0 12px 12px 0; color: #344054;
        font-size: .84rem; line-height: 1.5;
    }}
    .footer {{ border-top: 1px solid {LINE}; margin-top: 28px; padding-top: 14px; color: {MUTED}; font-size: .74rem; }}
    @media (max-width: 800px) {{
        .hero {{ padding: 22px; }}
        .hero-title {{ font-size: 1.55rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

LOGO_URL = "https://commons.wikimedia.org/wiki/Special:Redirect/file/Escudo%20Universidad%20del%20Cauca.png"


def pct(x: float | int | None, d: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{float(x):.{d}%}"


def num(x: float | int | None, d: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{float(x):.{d}f}"


def money(x: float | int | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    value = float(x)
    if abs(value) >= 1e12:
        return f"{value/1e12:.2f}T"
    if abs(value) >= 1e9:
        return f"{value/1e9:.2f}B"
    if abs(value) >= 1e6:
        return f"{value/1e6:.2f}M"
    return f"{value:,.0f}"


def render_brand() -> None:
    c1, c2 = st.columns([1.2, 5], vertical_alignment="center")
    with c1:
        st.image(LOGO_URL, width=70)
    with c2:
        st.markdown('<div class="brand-title">CAPM Analytics Pro</div>', unsafe_allow_html=True)
        st.markdown('<div class="brand-sub">Equity Risk & Required Return Analytics · Universidad del Cauca</div>', unsafe_allow_html=True)


def metric_card(label: str, value: str, help_text: str = "") -> None:
    st.markdown(
        f'<div class="card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-help">{help_text}</div></div>',
        unsafe_allow_html=True,
    )


def clean_close(df: pd.DataFrame) -> pd.Series:
    if "Close" not in df.columns:
        raise ValueError("Yahoo Finance did not return a Close series.")
    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if close.empty:
        raise ValueError("No valid closing-price observations were returned.")
    return close


def numeric_value(row: pd.Series, df: pd.DataFrame, *terms: str) -> float | None:
    column = find_column(df, *terms)
    if column is None:
        return None
    value = pd.to_numeric(str(row[column]).replace("%", "").replace(",", ""), errors="coerce")
    if pd.isna(value):
        return None
    value = float(value)
    if abs(value) > 1.5:
        value /= 100.0
    return value


def get_industry_beta(df: pd.DataFrame, industry_name: str) -> float | None:
    if not industry_name or "industry name" not in df.columns:
        return None
    beta_col = find_column(df, "beta")
    if beta_col is None:
        return None
    rows = df[df["industry name"].astype(str).str.strip() == str(industry_name).strip()]
    if rows.empty:
        return None
    value = pd.to_numeric(rows.iloc[0][beta_col], errors="coerce")
    return float(value) if pd.notna(value) else None


def choose_beta(method: str, historical: float, industry: float | None, weight: float) -> float:
    if method == "Historical OLS":
        return historical
    if method == "Damodaran Industry":
        if industry is None:
            raise ValueError("No valid Damodaran industry beta was found for the selected industry.")
        return industry
    if industry is None:
        raise ValueError("Blended beta requires a valid Damodaran industry beta.")
    return float(weight * historical + (1 - weight) * industry)


def clear_caches() -> None:
    try:
        st.cache_data.clear()
    except Exception:
        pass
    funcs = [load_country_risk, load_global_beta, fetch_us_10y, fetch_history, fetch_snapshot]
    for fn in funcs:
        try:
            fn.cache_clear()
        except Exception:
            pass


def run_analysis(config: dict) -> dict:
    ticker = normalize_ticker(config["ticker"])
    benchmark = normalize_ticker(config["benchmark"])
    if config["beta_min"] >= config["beta_max"]:
        raise ValueError("Beta minimum must be below beta maximum.")
    if config["beta_step"] <= 0:
        raise ValueError("Beta step must be greater than zero.")

    asset_df = fetch_history(ticker, period=config["period"])
    market_df = fetch_history(benchmark, period=config["period"])
    snapshot = fetch_snapshot(ticker)

    returns = prepare_returns(clean_close(asset_df), clean_close(market_df), frequency=config["frequency"])
    min_obs = {"M": 24, "W": 52, "D": 120}[config["frequency"]]
    beta_stats = estimate_beta(returns, min_observations=min_obs)
    historical_beta = beta_stats.beta
    rw = config["rolling_window"]
    rolling = rolling_beta(returns, window=min(rw, len(returns))) if len(returns) >= rw else pd.Series(dtype=float)

    country_df = load_country_risk(refresh=True)
    global_beta_df = load_global_beta(refresh=True)

    country_col = find_column(country_df, "country") or "country"
    countries = sorted(country_df[country_col].dropna().astype(str).str.strip().unique().tolist())
    requested_country = config["country"].strip()
    selected_country = next((c for c in countries if c.lower() == requested_country.lower()), countries[0] if countries else "United States")
    country_data = country_row(country_df, selected_country) if countries else pd.Series(dtype=object)

    erp = numeric_value(country_data, country_df, "equity", "risk", "premium") if countries else None
    crp = numeric_value(country_data, country_df, "country", "risk", "premium") if countries else None
    default_spread = numeric_value(country_data, country_df, "default", "spread") if countries else None

    industry_names = sorted(global_beta_df["industry name"].dropna().astype(str).str.strip().unique().tolist()) if "industry name" in global_beta_df.columns else []
    suggestions = suggested_industries(snapshot.sector, industry_names, snapshot.industry)
    selected_industry = suggestions[0] if suggestions else (industry_names[0] if industry_names else "")
    industry_beta = get_industry_beta(global_beta_df, selected_industry)
    beta_used = choose_beta(config["beta_method"], historical_beta, industry_beta, config["blend_weight"])

    if config["rf_mode"] == "Manual":
        rf = config["manual_rf"] / 100
        rf_date = "Manual"
    else:
        try:
            rf, rf_date = fetch_us_10y()
        except Exception:
            rf = config["manual_rf"] / 100
            rf_date = "Fallback manual"

    if config["market_mode"] == "Manual MRP" or erp is None:
        mrp = config["manual_mrp"] / 100
        market_status = "Manual MRP"
    else:
        mrp = erp
        market_status = "Damodaran country ERP"

    capm = calculate_capm(rf, beta_used, rf + mrp)
    beta_gap = abs(historical_beta - industry_beta) if industry_beta is not None else None
    rolling_dispersion = float(rolling.std()) if len(rolling) else None
    quality = score_beta_quality(
        beta_stats.observations,
        beta_stats.r_squared,
        beta_stats.beta_p_value,
        config["frequency"],
        beta_gap=beta_gap,
        rolling_dispersion=rolling_dispersion,
    )

    return {
        "ticker": ticker,
        "benchmark": benchmark,
        "period": config["period"],
        "frequency": config["frequency"],
        "rolling_window": rw,
        "snapshot": snapshot,
        "returns": returns,
        "beta_stats": beta_stats,
        "rolling": rolling,
        "historical_beta": historical_beta,
        "damodaran_beta": industry_beta,
        "industry": selected_industry,
        "industry_options": industry_names,
        "suggestions": suggestions,
        "beta_used": beta_used,
        "beta_method": config["beta_method"],
        "blend_weight": config["blend_weight"],
        "rf": rf,
        "rf_date": rf_date,
        "mrp": mrp,
        "market_return": rf + mrp,
        "capm": capm,
        "selected_country": selected_country,
        "country_crp": crp,
        "country_default_spread": default_spread,
        "market_status": market_status,
        "quality": quality,
        "analysis_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "beta_min": config["beta_min"],
        "beta_max": config["beta_max"],
        "beta_step": config["beta_step"],
    }


render_brand()
st.markdown(
    '<div class="hero"><div class="hero-kicker">Systematic risk & required return</div><div class="hero-title">From ticker to an auditable CAPM view</div><div class="hero-text">Analyze a listed asset using market history, statistical beta, Damodaran benchmarks and a transparent CAPM engine. The interface prioritizes evidence, visual diagnostics and traceability.</div><div class="formula">E(Rᵢ) = R𝒇 + βᵢ × [E(Rₘ) − R𝒇]</div></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Analysis setup")
    st.caption("Configure the research case, then run the analysis.")
    with st.form("analysis_form", clear_on_submit=False):
        ticker = st.text_input("Ticker", value=st.session_state.get("ticker", "NVDA"), placeholder="AAPL · MSFT · NVDA")
        benchmark = st.text_input("Benchmark", value=st.session_state.get("benchmark", "SPY"))
        period = st.selectbox("Historical window", ["2y", "3y", "5y", "10y"], index=2)
        frequency = st.selectbox("Return frequency", ["M", "W", "D"], index=0, format_func=lambda x: {"M": "Monthly", "W": "Weekly", "D": "Daily"}[x])
        rolling_window = st.selectbox("Rolling beta window", [12, 24, 36, 60], index=2)

        st.markdown("### Beta architecture")
        beta_method = st.selectbox("Beta used in CAPM", ["Historical OLS", "Damodaran Industry", "Blended"], index=0)
        blend_weight = st.slider("Historical weight · blended", 0.0, 1.0, 0.70, 0.05, disabled=beta_method != "Blended")

        st.markdown("### CAPM assumptions")
        rf_mode = st.radio("Risk-free", ["FRED U.S. 10Y", "Manual"], index=0)
        manual_rf = st.number_input("Manual Rf (%)", -20.0, 30.0, 4.25, 0.05, format="%.2f", disabled=rf_mode != "Manual")
        market_mode = st.radio("Market premium", ["Damodaran country ERP", "Manual MRP"], index=0)
        country = st.text_input("Country / market", value="United States", disabled=market_mode != "Damodaran country ERP")
        manual_mrp = st.number_input("Manual MRP (%)", -10.0, 30.0, 5.50, 0.10, format="%.2f", disabled=market_mode != "Manual MRP")

        st.markdown("### SML sensitivity")
        beta_min = st.number_input("Beta minimum", 0.0, 10.0, 0.0, 0.10)
        beta_max = st.number_input("Beta maximum", 0.1, 10.0, 2.5, 0.10)
        beta_step = st.number_input("Beta step", 0.01, 5.0, 0.25, 0.05)
        run = st.form_submit_button("Run CAPM analysis", type="primary", use_container_width=True)

    if st.button("Refresh external data", use_container_width=True):
        clear_caches()
        st.success("External-data cache cleared. Run the analysis again.")

    st.divider()
    st.caption("Sources: Yahoo Finance · Damodaran / NYU Stern · FRED")
    st.caption("Research and education tool. Verify assumptions before investment use.")

if run:
    config = {
        "ticker": ticker,
        "benchmark": benchmark,
        "period": period,
        "frequency": frequency,
        "rolling_window": rolling_window,
        "beta_method": beta_method,
        "blend_weight": blend_weight,
        "rf_mode": rf_mode,
        "manual_rf": manual_rf,
        "market_mode": market_mode,
        "country": country,
        "manual_mrp": manual_mrp,
        "beta_min": beta_min,
        "beta_max": beta_max,
        "beta_step": beta_step,
    }
    try:
        with st.spinner("Building CAPM research case…"):
            result = run_analysis(config)
        st.session_state["analysis"] = result
        st.session_state["ticker"] = ticker
        st.session_state["benchmark"] = benchmark
        st.success(f"Analysis ready for {result['ticker']}.")
    except Exception as exc:
        st.error(f"Analysis could not be completed: {exc}")
        st.info("Check the ticker, benchmark, data availability and CAPM assumptions, then try again.")

analysis = st.session_state.get("analysis")

if analysis is None:
    st.markdown('<div class="card"><div class="section-title">Ready for analysis</div><div class="section-sub">Enter a ticker in the left panel and run the workspace. The platform will build the chain from market data to beta, risk premium and required return without requiring a page reload.</div></div>', unsafe_allow_html=True)
else:
    s = analysis
    snap = s["snapshot"]
    bs = s["beta_stats"]
    capm = s["capm"]
    q = s["quality"]

    st.markdown(f"### {s['ticker']} · {snap.name}")
    st.caption(f"{snap.exchange} · {snap.currency} · {snap.sector} · {snap.industry} · {s['frequency']} returns · {s['period']}")

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        metric_card("CAPM required return", pct(capm.required_return), "Rf + β × MRP")
    with k2:
        metric_card("Beta used", num(s["beta_used"]), s["beta_method"])
    with k3:
        metric_card("Market risk premium", pct(s["mrp"]), s["market_status"])
    with k4:
        metric_card("Risk-free rate", pct(s["rf"]), f"Source: {s['rf_date']}")
    with k5:
        metric_card("Beta quality", f"{q.score}/100", q.label)

    st.markdown("## Research workspace")
    section = st.radio("Workspace section", ["Overview", "Beta Lab", "CAPM Engine", "Risk & Report"], horizontal=True, label_visibility="collapsed")

    if section == "Overview":
        a, b = st.columns([1.45, 1])
        with a:
            st.markdown("### Normalized performance")
            asset_norm = (1 + s["returns"]["asset"]).cumprod()
            market_norm = (1 + s["returns"]["market"]).cumprod()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=asset_norm.index, y=asset_norm, mode="lines", name=s["ticker"], line={"color": BLUE, "width": 2.6}))
            fig.add_trace(go.Scatter(x=market_norm.index, y=market_norm, mode="lines", name=s["benchmark"], line={"color": "#98A2B3", "width": 2}))
            fig.update_layout(height=390, margin={"l": 5, "r": 5, "t": 10, "b": 10}, yaxis_title="Growth of 1.0", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        with b:
            st.markdown("### CAPM decomposition")
            fig = go.Figure(go.Bar(x=["Risk-free", "β × MRP"], y=[capm.risk_free_rate, capm.equity_risk_premium], text=[pct(capm.risk_free_rate), pct(capm.equity_risk_premium)], textposition="outside", marker_color=[BLUE, RED]))
            fig.add_hline(y=capm.required_return, line_dash="dash", line_color=NAVY, annotation_text=f"Required {pct(capm.required_return)}")
            fig.update_layout(height=390, yaxis_tickformat=".0%", margin={"l": 5, "r": 5, "t": 10, "b": 10})
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            metric_card("Latest price", money(snap.price), snap.currency)
        with m2:
            metric_card("Market cap", money(snap.market_cap), "Yahoo Finance")
        with m3:
            metric_card("Annualized volatility", pct(bs.asset_vol_annualized), "Based on selected frequency")
        with m4:
            metric_card("R²", pct(bs.r_squared), "OLS market explanatory power")

        st.markdown("### Research interpretation")
        st.markdown('<div class="callout">A higher beta increases the CAPM-required return when the market risk premium is positive. Use the beta quality score and regression diagnostics to judge how much confidence to place in the point estimate.</div>', unsafe_allow_html=True)

    elif section == "Beta Lab":
        st.markdown("### Beta triangulation")
        st.caption("Compare company-specific historical risk with the Damodaran industry benchmark before deciding which beta should drive CAPM.")

        if s["industry_options"]:
            selected_industry = st.selectbox("Damodaran industry benchmark", s["industry_options"], index=s["industry_options"].index(s["industry"]) if s["industry"] in s["industry_options"] else 0)
            new_industry_beta = get_industry_beta(load_global_beta(refresh=False), selected_industry)
            if selected_industry != s["industry"] and new_industry_beta is not None:
                s["industry"] = selected_industry
                s["damodaran_beta"] = new_industry_beta
                try:
                    s["beta_used"] = choose_beta(s["beta_method"], s["historical_beta"], new_industry_beta, s["blend_weight"])
                    s["capm"] = calculate_capm(s["rf"], s["beta_used"], s["market_return"])
                    s["quality"] = score_beta_quality(bs.observations, bs.r_squared, bs.beta_p_value, s["frequency"], beta_gap=abs(s["historical_beta"] - new_industry_beta), rolling_dispersion=float(s["rolling"].std()) if len(s["rolling"]) else None)
                    st.session_state["analysis"] = s
                except ValueError:
                    pass

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            metric_card("Historical OLS", num(s["historical_beta"]), "Company vs benchmark")
        with b2:
            metric_card("Damodaran Industry", num(s["damodaran_beta"]), s["industry"] or "No matched benchmark")
        with b3:
            metric_card("Downside beta", num(bs.downside_beta), "Negative market periods")
        with b4:
            metric_card("Beta p-value", num(bs.beta_p_value, 4), "OLS significance")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Beta regression")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s["returns"]["market"], y=s["returns"]["asset"], mode="markers", name="Observed", marker={"size": 7, "opacity": .62, "color": BLUE}))
            x = np.linspace(float(s["returns"]["market"].min()), float(s["returns"]["market"].max()), 100)
            y = bs.alpha_period + bs.beta * x
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"OLS β={bs.beta:.3f}", line={"color": RED, "width": 2.5}))
            fig.update_layout(height=400, xaxis_title=f"{s['benchmark']} return", yaxis_title=f"{s['ticker']} return", xaxis_tickformat=".0%", yaxis_tickformat=".0%", margin={"l": 5, "r": 5, "t": 10, "b": 10})
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        with c2:
            st.markdown(f"### Rolling beta · {s['rolling_window']} periods")
            fig = go.Figure()
            if len(s["rolling"]):
                fig.add_trace(go.Scatter(x=s["rolling"].index, y=s["rolling"], mode="lines", name="Rolling beta", line={"color": BLUE, "width": 2.4}))
                fig.add_hline(y=s["beta_used"], line_dash="dash", line_color=RED, annotation_text="CAPM beta")
            else:
                fig.add_annotation(text="Not enough observations for this window", x=.5, y=.5, xref="paper", yref="paper", showarrow=False)
            fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Beta", margin={"l": 5, "r": 5, "t": 10, "b": 10})
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

        compare = pd.DataFrame({"Source": ["Historical OLS", "Damodaran Industry", "CAPM Used"], "Beta": [s["historical_beta"], s["damodaran_beta"], s["beta_used"]]})
        st.markdown("### Beta comparison")
        fig = go.Figure(go.Bar(x=compare["Source"], y=compare["Beta"], text=["—" if pd.isna(v) else f"{v:.3f}" for v in compare["Beta"]], textposition="outside", marker_color=[BLUE, "#98A2B3", RED]))
        fig.update_layout(height=330, yaxis_title="Beta", margin={"l": 5, "r": 5, "t": 10, "b": 10})
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

        st.markdown("### Quality diagnostics")
        qc1, qc2 = st.columns([1, 2])
        with qc1:
            metric_card("Beta quality", f"{q.score}/100", q.label)
            st.progress(q.score / 100)
        with qc2:
            for note in q.notes:
                st.write(f"• {note}")
            st.caption("The quality score diagnoses evidence; it does not mechanically change CAPM.")

    elif section == "CAPM Engine":
        left, right = st.columns(2)
        with left:
            st.markdown("### Standard CAPM")
            st.latex(r"E(R_i)=R_f+\beta_i(E(R_m)-R_f)")
            calc = pd.DataFrame({"Component": ["Risk-free", "Beta", "Market risk premium", "Expected market return", "Required return"], "Value": [pct(capm.risk_free_rate), num(capm.beta), pct(capm.market_risk_premium), pct(capm.expected_market_return), pct(capm.required_return)]})
            st.dataframe(calc, hide_index=True, use_container_width=True)
            st.success(f"CAPM required return · {pct(capm.required_return)}")
        with right:
            st.markdown("### CAPM Advanced")
            st.markdown('<div class="callout">Same core equation, stronger evidence. The advanced layer links ticker history, explicit market-premium provenance, three beta views, statistical diagnostics, sensitivity and scenario analysis without changing the economic logic of CAPM.</div>', unsafe_allow_html=True)
            st.write(f"**Beta architecture:** {s['beta_method']}")
            st.write(f"**Risk-free source:** {s['rf_date']}")
            st.write(f"**Market premium source:** {s['market_status']}")
            st.write(f"**Quality:** {q.score}/100 ({q.label})")

        st.markdown("### Security Market Line")
        betas = np.arange(float(s["beta_min"]), float(s["beta_max"]) + float(s["beta_step"]) / 2, float(s["beta_step"]))
        sens = sensitivity_table(s["rf"], s["mrp"], betas)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sens["Beta"], y=sens["Required Return"], mode="lines+markers", name="SML", line={"color": BLUE, "width": 2.6}))
        fig.add_trace(go.Scatter(x=[s["beta_used"]], y=[capm.required_return], mode="markers", name=s["ticker"], marker={"size": 13, "color": RED}))
        fig.update_layout(height=380, xaxis_title="Beta", yaxis_title="Required return", yaxis_tickformat=".0%", hovermode="x unified", margin={"l": 5, "r": 5, "t": 10, "b": 10})
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

        st.markdown("### Scenario analysis")
        scenarios = scenario_table(s["rf"], s["beta_used"], {"Bear": max(0.0, s["mrp"] - 0.02), "Base": s["mrp"], "Bull": s["mrp"] + 0.02})
        st.dataframe(scenarios, hide_index=True, use_container_width=True)

        st.markdown("### Sensitivity table")
        st.dataframe(sens, hide_index=True, use_container_width=True)

    else:
        left, right = st.columns([1.15, .85])
        with left:
            st.markdown("### Data lineage")
            lineage = pd.DataFrame({
                "Source": ["Yahoo Finance", "Damodaran / NYU Stern", "FRED"],
                "Role": ["Ticker metadata + adjusted price history", "Country ERP / CRP / default spread + industry beta", "U.S. Treasury 10Y risk-free"],
                "Status": ["Loaded", s["market_status"], "Loaded" if s["rf_date"] != "Manual" else "Manual"],
            })
            st.dataframe(lineage, hide_index=True, use_container_width=True)

            st.markdown("### Country-risk context")
            country_context = pd.DataFrame({"Metric": ["Country / market", "Equity risk premium", "Country risk premium", "Default spread"], "Value": [s["selected_country"], pct(s["mrp"]), pct(s["country_crp"]), pct(s["country_default_spread"])]})
            st.dataframe(country_context, hide_index=True, use_container_width=True)
            st.markdown('<div class="callout">Damodaran industry classifications are benchmarks, not automatic valuation conclusions. Confirm the industry mapping before using a beta in a formal valuation.</div>', unsafe_allow_html=True)
        with right:
            st.markdown("### Research quality")
            metric_card("Overall beta quality", f"{q.score}/100", q.label)
            st.progress(q.score / 100)
            for note in q.notes:
                st.write(f"• {note}")

            summary = {
                "Ticker": s["ticker"], "Company": snap.name, "Benchmark": s["benchmark"], "Exchange": snap.exchange,
                "Country / market": s["selected_country"], "Historical beta": num(s["historical_beta"]),
                "Damodaran industry beta": num(s["damodaran_beta"]), "Beta used": num(s["beta_used"]), "Risk-free rate": pct(s["rf"]),
                "Market risk premium": pct(s["mrp"]), "Expected market return": pct(s["market_return"]), "CAPM required return": pct(capm.required_return),
                "R²": pct(bs.r_squared), "Correlation": num(bs.correlation), "Observations": str(bs.observations),
                "Beta p-value": num(bs.beta_p_value, 4), "Beta quality": f"{q.score}/100 ({q.label})", "Estimation history": s["period"],
                "Return frequency": {"M": "Monthly", "W": "Weekly", "D": "Daily"}[s["frequency"]], "Analysis timestamp": s["analysis_timestamp"],
            }
            pdf = build_pdf(summary, q.notes + ["Core equation: E(Ri) = Rf + beta × [E(Rm) − Rf]."])
            st.download_button("Download PDF report", pdf, file_name=f"CAPM_{s['ticker']}_report.pdf", mime="application/pdf", use_container_width=True)
            st.download_button("Download synchronized returns", s["returns"].to_csv(index=True).encode("utf-8"), file_name=f"CAPM_{s['ticker']}_returns.csv", mime="text/csv", use_container_width=True)
            st.download_button("Download sensitivity CSV", sensitivity_table(s["rf"], s["mrp"], np.arange(float(s["beta_min"]), float(s["beta_max"]) + float(s["beta_step"]) / 2, float(s["beta_step"]))).to_csv(index=False).encode("utf-8"), file_name=f"CAPM_{s['ticker']}_sensitivity.csv", mime="text/csv", use_container_width=True)
            st.caption("Reports are analytical summaries, not investment advice.")

st.markdown('<div class="footer">CAPM Analytics Pro · Universidad del Cauca · Author: Juan Carlos Muñoz · Yahoo Finance · Damodaran / NYU Stern · FRED</div>', unsafe_allow_html=True)
