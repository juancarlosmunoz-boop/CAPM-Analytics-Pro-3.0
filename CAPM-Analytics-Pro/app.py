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
from utils.formatting import money, num, pct
from utils.industry import suggested_industries

st.set_page_config(
    page_title="CAPM Analytics Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Design system — institutional-inspired navy/red, 8px rhythm, accessible type
# -----------------------------------------------------------------------------
NAVY = "#12233F"
BLUE = "#1F4E79"
RED = "#B42318"
INK = "#172033"
MUTED = "#667085"
LINE = "#E4E7EC"
SURFACE = "#FFFFFF"
SURFACE_ALT = "#F7F9FC"
POSITIVE = "#087443"
WARNING = "#9A6700"

st.markdown(
    f"""
    <style>
    :root {{
      --navy: {NAVY}; --blue: {BLUE}; --red: {RED}; --ink: {INK};
      --muted: {MUTED}; --line: {LINE}; --surface: {SURFACE}; --surface-alt: {SURFACE_ALT};
    }}
    .stApp {{ background: #F4F6FA; }}
    .block-container {{ max-width: 1480px; padding-top: 1.05rem; padding-bottom: 3rem; }}
    [data-testid="stSidebar"] {{ background: #0F1D35; border-right: 1px solid #223452; }}
    [data-testid="stSidebar"] * {{ color: #F8FAFC !important; }}
    [data-testid="stSidebar"] .stCaption {{ color: #C7D2E5 !important; }}
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] input {{ background: #172944 !important; border-color: #324765 !important; }}
    [data-testid="stSidebar"] .stButton > button {{ border-radius: 10px; min-height: 44px; }}

    .topbar {{
      display:flex; align-items:center; justify-content:space-between; gap:18px;
      background:var(--surface); border:1px solid var(--line); border-radius:18px;
      padding:14px 18px; margin-bottom:14px; box-shadow:0 5px 18px rgba(18,35,63,.05);
    }}
    .brand {{ display:flex; align-items:center; gap:12px; }}
    .brand img {{ width:54px; height:54px; object-fit:contain; border-radius:10px; }}
    .brand-fallback {{ width:54px; height:54px; border-radius:12px; background:var(--navy); color:#fff; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:13px; text-align:center; line-height:1.05; }}
    .brand-title {{ font-size:1.18rem; font-weight:800; color:var(--navy); line-height:1.05; }}
    .brand-sub {{ font-size:.78rem; color:var(--muted); margin-top:4px; }}
    .status {{ display:inline-flex; align-items:center; gap:7px; padding:7px 11px; border-radius:999px; font-size:.76rem; font-weight:700; background:#EEF4FF; color:#1D4E89; white-space:nowrap; }}
    .dot {{ width:7px; height:7px; background:#22A06B; border-radius:50%; display:inline-block; }}

    .hero {{ background:linear-gradient(135deg, #12233F 0%, #1B3A5F 66%, #274F7A 100%); color:white;
             border-radius:20px; padding:28px 30px; margin-bottom:16px; box-shadow:0 12px 30px rgba(18,35,63,.16); }}
    .hero-kicker {{ color:#C6D4E7; font-size:.76rem; font-weight:800; letter-spacing:.09em; text-transform:uppercase; }}
    .hero h1 {{ color:white; font-size:2.25rem; margin:.30rem 0 .35rem; letter-spacing:-.02em; }}
    .hero p {{ color:#DCE6F2; margin:0; max-width:840px; line-height:1.55; }}
    .formula {{ margin-top:18px; padding:12px 14px; background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.16); border-radius:12px; font-weight:800; font-size:1.14rem; display:inline-block; }}

    .section-head {{ display:flex; align-items:end; justify-content:space-between; gap:12px; margin:18px 0 10px; }}
    .section-head h2 {{ margin:0; font-size:1.18rem; color:var(--navy); }}
    .section-head p {{ margin:0; color:var(--muted); font-size:.82rem; }}

    .metric-card {{ background:var(--surface); border:1px solid var(--line); border-radius:15px; padding:15px 16px; min-height:108px; box-shadow:0 4px 15px rgba(18,35,63,.04); transition:transform .18s ease, box-shadow .18s ease; }}
    .metric-card:hover {{ transform:translateY(-2px); box-shadow:0 9px 22px rgba(18,35,63,.08); }}
    .metric-label {{ color:var(--muted); font-size:.75rem; font-weight:700; text-transform:uppercase; letter-spacing:.04em; }}
    .metric-value {{ color:var(--navy); font-size:1.55rem; font-weight:850; margin-top:7px; letter-spacing:-.02em; }}
    .metric-helper {{ color:var(--muted); font-size:.72rem; margin-top:4px; }}
    .metric-accent {{ border-top:3px solid var(--blue); }}
    .metric-accent-red {{ border-top:3px solid var(--red); }}

    .source-chip {{ display:inline-flex; padding:6px 9px; border-radius:999px; background:#F2F4F7; border:1px solid var(--line); color:#475467; font-size:.72rem; font-weight:700; margin-right:6px; margin-bottom:5px; }}
    .info-card {{ background:var(--surface); border:1px solid var(--line); border-radius:15px; padding:16px; }}
    .info-title {{ color:var(--navy); font-weight:800; margin-bottom:6px; }}
    .small {{ color:var(--muted); font-size:.78rem; line-height:1.5; }}
    .callout {{ border-left:4px solid var(--blue); background:#F0F6FB; padding:12px 14px; border-radius:0 12px 12px 0; color:#344054; font-size:.84rem; line-height:1.5; }}
    button:focus-visible, input:focus-visible, textarea:focus-visible {{ outline:3px solid rgba(31,78,121,.35) !important; outline-offset:2px; }}
    .footer {{ border-top:1px solid var(--line); margin-top:24px; padding-top:14px; color:var(--muted); font-size:.74rem; display:flex; justify-content:space-between; gap:16px; flex-wrap:wrap; }}

    /* Responsive adjustments */
    @media (max-width: 900px) {{
      .hero {{ padding:22px; }} .hero h1 {{ font-size:1.75rem; }}
      .brand-sub {{ display:none; }} .topbar {{ align-items:flex-start; }}
    }}
    @media (max-width: 640px) {{
      .block-container {{ padding-left: .85rem; padding-right: .85rem; }}
      .hero h1 {{ font-size:1.45rem; }} .formula {{ font-size:.98rem; }}
      .metric-value {{ font-size:1.32rem; }} .metric-card {{ min-height:96px; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

LOGO_URL = "https://commons.wikimedia.org/wiki/Special:Redirect/file/Escudo%20Universidad%20del%20Cauca.png"
FALLBACK_LOGO = Path(__file__).parent / "assets" / "unicauca_fallback.svg"


def _render_brand() -> None:
    st.markdown(
        f"""
        <div class="topbar">
          <div class="brand">
            <img src="{LOGO_URL}" alt="Escudo de la Universidad del Cauca" onerror="this.style.display='none';this.nextElementSibling.style.display='flex';"/>
            <div class="brand-fallback" style="display:none">U<br/>C</div>
            <div>
              <div class="brand-title">CAPM Analytics Pro</div>
              <div class="brand-sub">Equity risk analytics · Universidad del Cauca</div>
            </div>
          </div>
          <div class="status"><span class="dot"></span> Research workspace</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _clean_close(df: pd.DataFrame) -> pd.Series:
    if "Close" not in df.columns:
        raise ValueError("Yahoo Finance did not return a Close series.")
    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if close.empty:
        raise ValueError("No valid closing-price observations were returned.")
    return close


def _find_numeric_value(row: pd.Series, df: pd.DataFrame, terms: tuple[str, ...]) -> float | None:
    column = find_column(df, *terms)
    if column is None:
        return None
    cleaned = str(row[column]).replace("%", "").replace(",", "").strip()
    value = pd.to_numeric(cleaned, errors="coerce")
    if pd.isna(value):
        return None
    value = float(value)
    # Damodaran files may represent percentages as 4.25 or 0.0425 depending on the table.
    if abs(value) > 1.5:
        value /= 100.0
    return value


def _annual_return(returns: pd.Series, periods: int) -> float | None:
    if returns.empty:
        return None
    growth = float((1.0 + returns).prod())
    years = len(returns) / periods
    return growth ** (1 / years) - 1 if years > 0 and growth > 0 else None


def _metric(label: str, value: str, helper: str = "", red: bool = False) -> None:
    klass = "metric-card metric-accent-red" if red else "metric-card metric-accent"
    st.markdown(
        f"""<div class="{klass}"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-helper">{helper}</div></div>""",
        unsafe_allow_html=True,
    )


def _cache_clear() -> None:
    try:
        st.cache_data.clear()
    except Exception:
        pass
    for module_func in [load_country_risk, load_global_beta, fetch_us_10y, fetch_history, fetch_snapshot]:
        try:
            module_func.cache_clear()
        except Exception:
            pass


def _empty_state() -> None:
    st.markdown(
        """
        <div class="info-card" style="margin-top:14px;">
          <div class="info-title">Ready for analysis</div>
          <div class="small">Enter a ticker in the analysis panel and run the workspace. The platform will build the CAPM chain from market data through beta, risk premium and required return.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_snapshot_inputs() -> dict:
    defaults = {
        "ticker_input": "NVDA",
        "benchmark_input": "SPY",
        "period": "5y",
        "frequency": "M",
        "rolling_window": 36,
        "beta_method": "Historical OLS",
        "blend_weight": 0.70,
        "rf_mode": "FRED U.S. 10Y",
        "manual_rf": 4.25,
        "market_mode": "Damodaran country ERP",
        "country": "United States",
        "manual_mrp": 5.50,
        "beta_min": 0.0,
        "beta_max": 2.5,
        "beta_step": 0.25,
    }
    return defaults


_render_brand()

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Systematic risk & required return</div>
      <h1>From ticker to an auditable CAPM view</h1>
      <p>Connect real market history with statistical beta, Damodaran benchmarks and a transparent CAPM engine. The dashboard is designed for research, valuation preparation and financial education.</p>
      <div class="formula">E(Rᵢ) = R𝒇 + βᵢ × [E(Rₘ) − R𝒇]</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Analysis setup")
    st.caption("Use the form below to configure the research case. Inputs remain traceable in the report.")
    with st.form("analysis_form", clear_on_submit=False):
        ticker_input = st.text_input("Ticker", value="NVDA", placeholder="AAPL · MSFT · NVDA · AMZN", help="Yahoo Finance symbol.")
        benchmark_input = st.text_input("Market benchmark", value="SPY", help="Broad market proxy used in the beta regression.")
        period = st.selectbox("Historical window", ["2y", "3y", "5y", "10y"], index=2)
        frequency = st.selectbox("Return frequency", ["M", "W", "D"], index=0, format_func=lambda x: {"M": "Monthly", "W": "Weekly", "D": "Daily"}[x])
        rolling_window = st.selectbox("Rolling beta window", [12, 24, 36, 60], index=2)

        st.markdown("#### Beta architecture")
        beta_method = st.selectbox("Beta used in CAPM", ["Historical OLS", "Damodaran Industry", "Blended"], index=0)
        blend_weight = st.slider("Historical weight · blended", 0.0, 1.0, 0.70, 0.05, disabled=beta_method != "Blended")

        st.markdown("#### CAPM assumptions")
        rf_mode = st.radio("Risk-free rate", ["FRED U.S. 10Y", "Manual"], index=0, horizontal=True)
        manual_rf = st.number_input("Manual Rf (%)", min_value=-20.0, max_value=30.0, value=4.25, step=0.05, format="%.2f", disabled=rf_mode != "Manual")
        market_mode = st.radio("Market premium", ["Damodaran country ERP", "Manual MRP"], index=0, horizontal=False)
        country = st.text_input("Country / market", value="United States", disabled=market_mode != "Damodaran country ERP")
        manual_mrp = st.number_input("Manual MRP (%)", min_value=-10.0, max_value=30.0, value=5.50, step=0.10, format="%.2f", disabled=market_mode != "Manual MRP")

        st.markdown("#### Sensitivity")
        beta_min = st.number_input("Beta minimum", value=0.0, step=0.1)
        beta_max = st.number_input("Beta maximum", value=2.5, step=0.1)
        beta_step = st.number_input("Beta step", value=0.25, min_value=0.01, step=0.05)

        run = st.form_submit_button("Run CAPM analysis", type="primary", use_container_width=True)

    if st.button("Refresh external data", use_container_width=True):
        _cache_clear()
        st.toast("External data cache cleared.")
        st.rerun()

    st.markdown("---")
    st.caption("Sources: Yahoo Finance · Damodaran / NYU Stern · FRED")
    st.caption("Designed for research and education; verify assumptions before investment use.")

if "analysis" not in st.session_state:
    _empty_state()

if run:
    try:
        ticker = normalize_ticker(ticker_input)
        benchmark = normalize_ticker(benchmark_input)
        if beta_min >= beta_max:
            raise ValueError("Beta minimum must be below beta maximum.")
        if beta_step <= 0:
            raise ValueError("Beta step must be greater than zero.")

        with st.status("Building CAPM research case…", expanded=True) as status:
            st.write(f"Loading {ticker} and {benchmark} from Yahoo Finance")
            asset_df = fetch_history(ticker, period=period)
            market_df = fetch_history(benchmark, period=period)
            snapshot = fetch_snapshot(ticker)

            st.write("Synchronizing returns and estimating historical beta")
            asset_close = _clean_close(asset_df)
            market_close = _clean_close(market_df)
            returns = prepare_returns(asset_close, market_close, frequency=frequency)
            min_obs = 24 if frequency == "M" else 52 if frequency == "W" else 120
            beta_stats = estimate_beta(returns, min_observations=min_obs)
            historical_beta = beta_stats.beta
            rolling = rolling_beta(returns, window=min(rolling_window, len(returns))) if len(returns) >= rolling_window else pd.Series(dtype=float)

            st.write("Loading Damodaran country and industry benchmarks")
            country_df = load_country_risk(refresh=True)
            damodaran_df = load_global_beta(refresh=True)

            country_col = find_column(country_df, "country") or "country"
            available_countries = sorted(country_df[country_col].dropna().astype(str).unique().tolist())
            chosen_country = country if country in available_countries else (available_countries[0] if available_countries else "United States")
            country_data_row = country_row(country_df, chosen_country)
            erp = _find_numeric_value(country_data_row, country_df, ("equity", "risk", "premium"))
            crp = _find_numeric_value(country_data_row, country_df, ("country", "risk", "premium"))
            default_spread = _find_numeric_value(country_data_row, country_df, ("default", "spread"))

            industry_names = damodaran_df["industry name"].dropna().astype(str).unique().tolist() if "industry name" in damodaran_df.columns else []
            suggestions = suggested_industries(snapshot.sector, industry_names, snapshot.industry)
            default_industry = suggestions[0] if suggestions else (industry_names[0] if industry_names else "")
            beta_col = find_column(damodaran_df, "beta")
            industry_name = default_industry
            damo_beta = None
            if beta_col and industry_name:
                rows = damodaran_df[damodaran_df["industry name"].astype(str) == industry_name]
                if not rows.empty:
                    damo_beta = pd.to_numeric(rows.iloc[0][beta_col], errors="coerce")
                    damo_beta = float(damo_beta) if pd.notna(damo_beta) else None

            if beta_method == "Historical OLS":
                beta_used = historical_beta
            elif beta_method == "Damodaran Industry" and damo_beta is not None:
                beta_used = damo_beta
            elif beta_method == "Blended" and damo_beta is not None:
                beta_used = blend_weight * historical_beta + (1 - blend_weight) * damo_beta
            else:
                beta_used = historical_beta

            st.write("Resolving risk-free rate and CAPM market premium")
            rf_date = "Manual"
            if rf_mode == "Manual":
                rf = manual_rf / 100
            else:
                try:
                    rf, rf_date = fetch_us_10y()
                except Exception:
                    rf = manual_rf / 100
                    rf_date = "Fallback manual"

            if market_mode == "Manual MRP" or erp is None:
                mrp = manual_mrp / 100
                market_status = "Manual MRP"
            else:
                mrp = erp
                market_status = "Damodaran country ERP"

            market_return = rf + mrp
            capm = calculate_capm(rf, beta_used, market_return)
            beta_gap = abs(historical_beta - damo_beta) if damo_beta is not None else None
            rolling_dispersion = float(rolling.std()) if len(rolling) else None
            quality = score_beta_quality(beta_stats.observations, beta_stats.r_squared, beta_stats.beta_p_value, frequency, beta_gap=beta_gap, rolling_dispersion=rolling_dispersion)

            status.update(label="CAPM case ready", state="complete", expanded=False)

        st.session_state["analysis"] = {
            "ticker": ticker,
            "benchmark": benchmark,
            "period": period,
            "frequency": frequency,
            "snapshot": snapshot,
            "returns": returns,
            "beta_stats": beta_stats,
            "rolling": rolling,
            "historical_beta": historical_beta,
            "damodaran_beta": damo_beta,
            "industry": industry_name,
            "industry_options": industry_names,
            "suggestions": suggestions,
            "beta_used": float(beta_used),
            "rf": rf,
            "rf_date": rf_date,
            "mrp": mrp,
            "market_return": market_return,
            "capm": capm,
            "selected_country": chosen_country,
            "country_crp": crp,
            "country_default_spread": default_spread,
            "market_status": market_status,
            "quality": quality,
            "analysis_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "beta_method": beta_method,
            "blend_weight": blend_weight,
        }
        st.rerun()

    except Exception as exc:
        st.error(f"Analysis could not be completed: {exc}")
        st.info("Check the ticker, benchmark, historical window and external data availability, then run again.")

analysis = st.session_state.get("analysis")
if analysis:
    s = analysis
    snap = s["snapshot"]
    bs = s["beta_stats"]
    capm = s["capm"]
    q = s["quality"]

    st.markdown(
        f"""
        <div class="info-card" style="margin-top:14px;">
          <div style="display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;align-items:flex-start;">
            <div>
              <div style="font-size:1.42rem;font-weight:850;color:{NAVY};">{s['ticker']} · {snap.name}</div>
              <div class="small">{snap.exchange} · {snap.currency} · {snap.sector} · {snap.industry}</div>
            </div>
            <div style="text-align:right;">
              <div class="small">Analysis timestamp</div>
              <div style="font-weight:800;color:{NAVY};">{s['analysis_timestamp']}</div>
            </div>
          </div>
          <div style="margin-top:12px;">
            <span class="source-chip">Yahoo Finance</span>
            <span class="source-chip">Damodaran</span>
            <span class="source-chip">FRED / Treasury</span>
            <span class="source-chip">{s['frequency']} returns · {s['period']}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-head"><div><h2>Key outputs</h2><p>Core numbers generated from the configured research case.</p></div></div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        _metric("CAPM required return", pct(capm.required_return), "Selected beta × market premium")
    with k2:
        _metric("Beta used", num(s["beta_used"], 3), s["beta_method"])
    with k3:
        _metric("Market risk premium", pct(s["mrp"]), s["market_status"])
    with k4:
        _metric("Risk-free rate", pct(s["rf"]), f"Source date: {s['rf_date']}")
    with k5:
        _metric("Beta quality", f"{q.score}/100", q.label, red=q.score < 70)

    st.markdown('<div class="section-head"><div><h2>Research workspace</h2><p>Move from market behaviour to beta evidence and then to required return.</p></div></div>', unsafe_allow_html=True)
    tab_overview, tab_beta, tab_capm, tab_risk = st.tabs(["Overview", "Beta Lab", "CAPM Engine", "Risk & Report"])

    with tab_overview:
        c1, c2 = st.columns([1.55, 1])
        with c1:
            st.markdown("#### Normalized performance")
            asset_norm = s["returns"]["asset"].add(1).cumprod()
            market_norm = s["returns"]["market"].add(1).cumprod()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=asset_norm.index, y=asset_norm, mode="lines", name=s["ticker"], line=dict(color=BLUE, width=2.6)))
            fig.add_trace(go.Scatter(x=market_norm.index, y=market_norm, mode="lines", name=s["benchmark"], line=dict(color="#98A2B3", width=2)))
            fig.update_layout(height=365, margin=dict(l=8, r=8, t=12, b=10), yaxis_title="Growth of 1.0", hovermode="x unified", legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        with c2:
            st.markdown("#### CAPM decomposition")
            fig = go.Figure(go.Bar(x=["Risk-free", "β × MRP"], y=[capm.risk_free_rate, capm.equity_risk_premium], text=[pct(capm.risk_free_rate), pct(capm.equity_risk_premium)], textposition="outside", marker_color=[BLUE, RED]))
            fig.add_hline(y=capm.required_return, line_dash="dash", line_color=NAVY, annotation_text=f"Required return {pct(capm.required_return)}")
            fig.update_layout(height=365, yaxis_tickformat=".0%", yaxis_title="Return contribution", margin=dict(l=8, r=8, t=12, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            _metric("Annualized asset return", pct(_annual_return(s["returns"]["asset"], 12 if s["frequency"] == "M" else 52 if s["frequency"] == "W" else 252)))
        with r2:
            _metric("Annualized volatility", pct(bs.asset_vol_annualized))
        with r3:
            _metric("R²", pct(bs.r_squared), "Market variation explained")
        with r4:
            _metric("Observations", str(bs.observations), f"{s['frequency']} synchronized returns")

    with tab_beta:
        if s.get("industry_options"):
            current_index = s["industry_options"].index(s["industry"]) if s["industry"] in s["industry_options"] else 0
            selected_industry_ui = st.selectbox(
                "Damodaran industry benchmark",
                s["industry_options"],
                index=current_index,
                key="industry_selector",
                help="Yahoo Finance and Damodaran use different taxonomies. Select the closest industry benchmark and apply it to the CAPM case.",
            )
            if selected_industry_ui != s["industry"] and st.button("Apply industry benchmark", key="apply_industry", type="secondary"):
                damo_df_ui = load_global_beta(refresh=False)
                beta_col_ui = find_column(damo_df_ui, "beta")
                rows_ui = damo_df_ui[damo_df_ui["industry name"].astype(str) == selected_industry_ui] if beta_col_ui else pd.DataFrame()
                damo_beta_ui = None if rows_ui.empty else pd.to_numeric(rows_ui.iloc[0][beta_col_ui], errors="coerce")
                if pd.notna(damo_beta_ui):
                    if s["beta_method"] == "Historical OLS":
                        beta_used_ui = s["historical_beta"]
                    elif s["beta_method"] == "Damodaran Industry":
                        beta_used_ui = float(damo_beta_ui)
                    else:
                        beta_used_ui = s["blend_weight"] * s["historical_beta"] + (1 - s["blend_weight"]) * float(damo_beta_ui)
                    capm_ui = calculate_capm(s["rf"], beta_used_ui, s["market_return"])
                    beta_gap_ui = abs(s["historical_beta"] - float(damo_beta_ui))
                    rolling_disp_ui = float(s["rolling"].std()) if len(s["rolling"]) else None
                    q_ui = score_beta_quality(s["beta_stats"].observations, s["beta_stats"].r_squared, s["beta_stats"].beta_p_value, s["frequency"], beta_gap=beta_gap_ui, rolling_dispersion=rolling_disp_ui)
                    s.update({"industry": selected_industry_ui, "damodaran_beta": float(damo_beta_ui), "beta_used": float(beta_used_ui), "capm": capm_ui, "quality": q_ui})
                    st.session_state["analysis"] = s
                    st.toast(f"Industry benchmark updated: {selected_industry_ui}")
                    st.rerun()

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            _metric("Historical OLS", num(s["historical_beta"], 3), "Company vs benchmark")
        with b2:
            _metric("Damodaran Industry", num(s["damodaran_beta"], 3), s["industry"] or "No benchmark matched")
        with b3:
            _metric("Downside beta", num(bs.downside_beta, 3), "Negative benchmark periods")
        with b4:
            _metric("Beta p-value", num(bs.beta_p_value, 4), "OLS significance")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Beta regression")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s["returns"]["market"], y=s["returns"]["asset"], mode="markers", name="Observed returns", marker=dict(size=7, opacity=.62, color=BLUE)))
            x = np.linspace(float(s["returns"]["market"].min()), float(s["returns"]["market"].max()), 100)
            y = bs.alpha_period + bs.beta * x
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"OLS β={bs.beta:.3f}", line=dict(color=RED, width=2.5)))
            fig.update_layout(height=390, xaxis_title=f"{s['benchmark']} return", yaxis_title=f"{s['ticker']} return", xaxis_tickformat=".0%", yaxis_tickformat=".0%", margin=dict(l=8, r=8, t=12, b=10), legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        with c2:
            st.markdown(f"#### Rolling beta · {min(s['rolling'].name and int(s['rolling'].name.split('(')[-1].rstrip(')')) if len(s['rolling']) else 0, 999)} observations")
            fig = go.Figure()
            if len(s["rolling"]):
                fig.add_trace(go.Scatter(x=s["rolling"].index, y=s["rolling"], mode="lines", name="Rolling beta", line=dict(color=BLUE, width=2.4)))
                fig.add_hline(y=s["beta_used"], line_dash="dash", line_color=RED, annotation_text="CAPM beta used")
            else:
                fig.add_annotation(text="Not enough observations for this rolling window", x=.5, y=.5, xref="paper", yref="paper", showarrow=False)
            fig.update_layout(height=390, xaxis_title="Date", yaxis_title="Beta", margin=dict(l=8, r=8, t=12, b=10))
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

        st.markdown("#### Beta triangulation")
        compare = pd.DataFrame({"Source": ["Historical OLS", "Damodaran Industry", "CAPM Used"], "Beta": [s["historical_beta"], s["damodaran_beta"], s["beta_used"]]})
        fig = go.Figure(go.Bar(x=compare["Source"], y=compare["Beta"], text=["N/A" if pd.isna(v) else f"{v:.3f}" for v in compare["Beta"]], textposition="outside", marker_color=[BLUE, "#98A2B3", RED]))
        fig.update_layout(height=320, yaxis_title="Beta", margin=dict(l=8, r=8, t=12, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

        qc1, qc2 = st.columns([.23, .77])
        with qc1:
            _metric("Beta quality", f"{q.score}/100", q.label, red=q.score < 70)
        with qc2:
            st.markdown('<div class="callout"><b>How to read it:</b> the quality score diagnoses the evidence behind beta. It does not mechanically alter the CAPM result. A material gap between historical and industry beta is a prompt to investigate business mix, sample period or benchmark choice.</div>', unsafe_allow_html=True)
            for note in q.notes:
                st.caption(f"• {note}")

    with tab_capm:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("#### Standard CAPM")
            st.latex(r"E(R_i)=R_f+\beta_i(E(R_m)-R_f)")
            calc = pd.DataFrame({"Component": ["Risk-free", "Beta", "Market risk premium", "Expected market return", "Required return"], "Value": [pct(capm.risk_free_rate), num(capm.beta, 3), pct(capm.market_risk_premium), pct(capm.expected_market_return), pct(capm.required_return)]})
            st.dataframe(calc, hide_index=True, use_container_width=True)
            st.success(f"CAPM required return · {pct(capm.required_return)}")
        with c2:
            st.markdown("#### CAPM Advanced")
            st.markdown('<div class="callout">The advanced layer keeps the same CAPM equation while improving the evidence around each input: live ticker history, explicit market-premium provenance, three beta views, statistical diagnostics and scenario testing.</div>', unsafe_allow_html=True)
            advanced = pd.DataFrame({"Component": ["Rf", "Beta", "MRP", "E(Rm)", "Beta quality", "Data sources"], "Selected value": [pct(s["rf"]), num(s["beta_used"], 3), pct(s["mrp"]), pct(s["market_return"]), f"{q.score}/100", "Yahoo · Damodaran · FRED"]})
            st.dataframe(advanced, hide_index=True, use_container_width=True)

        st.markdown("#### Security Market Line")
        betas = np.arange(float(beta_min), float(beta_max) + float(beta_step) / 2, float(beta_step))
        sens = sensitivity_table(s["rf"], s["mrp"], betas)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sens["Beta"], y=sens["Required Return"], mode="lines+markers", name="SML", line=dict(color=BLUE, width=2.6)))
        fig.add_trace(go.Scatter(x=[s["beta_used"]], y=[capm.required_return], mode="markers", name=s["ticker"], marker=dict(size=13, color=RED)))
        fig.update_layout(height=370, xaxis_title="Beta", yaxis_title="Required return", yaxis_tickformat=".0%", hovermode="x unified", margin=dict(l=8, r=8, t=12, b=10), legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

        scen_mrp = {"Bear": max(0.0, s["mrp"] - 0.02), "Base": s["mrp"], "Bull": s["mrp"] + 0.02}
        scenarios = scenario_table(s["rf"], s["beta_used"], scen_mrp)
        st.markdown("#### Scenario analysis")
        sc1, sc2 = st.columns([1.0, 1.0])
        with sc1:
            st.dataframe(scenarios.style.format({c: "{:.2%}" for c in ["Risk-free", "Market Risk Premium", "Expected Market Return", "Required Return"]}), hide_index=True, use_container_width=True)
        with sc2:
            fig = go.Figure(go.Bar(x=scenarios["Scenario"], y=scenarios["Required Return"], text=[f"{v:.2%}" for v in scenarios["Required Return"]], textposition="outside", marker_color=["#98A2B3", BLUE, RED]))
            fig.update_layout(height=300, yaxis_title="Required return", yaxis_tickformat=".0%", margin=dict(l=8, r=8, t=12, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

        st.markdown("#### Sensitivity table")
        st.dataframe(sens.style.format({"Beta": "{:.2f}", "Required Return": "{:.2%}"}), hide_index=True, use_container_width=True)

    with tab_risk:
        left, right = st.columns([1.15, .85])
        with left:
            st.markdown("#### Data lineage")
            lineage = pd.DataFrame({
                "Source": ["Yahoo Finance", "Damodaran / NYU Stern", "FRED"],
                "Role": ["Ticker metadata + adjusted price history", "Country ERP / CRP / default spread + industry beta", "U.S. Treasury 10Y risk-free"],
                "Status": ["Loaded for this case", s["market_status"], f"Loaded · {s['rf_date']}" if s["rf_date"] != "Manual" else "Manual"],
            })
            st.dataframe(lineage, hide_index=True, use_container_width=True)
            st.markdown('<div class="callout">Taxonomy note: Yahoo Finance and Damodaran classify industries differently. The suggested industry is a benchmark assist, not an automated investment conclusion. Confirm the mapping before using the number in a formal valuation.</div>', unsafe_allow_html=True)

            st.markdown("#### Country-risk context")
            country_context = pd.DataFrame({"Metric": ["Country / market", "Equity risk premium", "Country risk premium", "Default spread"], "Value": [s["selected_country"], pct(s["mrp"]), pct(s["country_crp"]), pct(s["country_default_spread"])]})
            st.dataframe(country_context, hide_index=True, use_container_width=True)

        with right:
            st.markdown("#### Research quality")
            _metric("Overall beta quality", f"{q.score}/100", q.label, red=q.score < 70)
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            st.progress(q.score / 100)
            for note in q.notes:
                st.caption(f"• {note}")

            st.markdown("#### Downloads")
            report_summary = {
                "Ticker": s["ticker"], "Company": snap.name, "Benchmark": s["benchmark"], "Exchange": snap.exchange,
                "Currency": snap.currency, "Country / market": s["selected_country"], "Historical beta": num(s["historical_beta"], 3),
                "Damodaran industry beta": num(s["damodaran_beta"], 3), "Beta used": num(s["beta_used"], 3), "Risk-free rate": pct(s["rf"]),
                "Market risk premium": pct(s["mrp"]), "Expected market return": pct(s["market_return"]), "CAPM required return": pct(capm.required_return),
                "R²": pct(bs.r_squared), "Correlation": num(bs.correlation, 3), "Observations": str(bs.observations),
                "Beta p-value": num(bs.beta_p_value, 4), "Beta quality": f"{q.score}/100 ({q.label})", "Estimation history": s["period"],
                "Return frequency": {"M": "Monthly", "W": "Weekly", "D": "Daily"}[s["frequency"]], "Analysis timestamp": s["analysis_timestamp"],
            }
            notes = [
                "Core equation: E(Ri) = Rf + beta × [E(Rm) − Rf].",
                "Historical beta is estimated with OLS on synchronized asset and benchmark returns.",
                "Damodaran industry beta is a benchmark and is not automatically superior to company-specific beta.",
                "CAPM Advanced improves the evidence, provenance and diagnostics around the same CAPM equation.",
                "Beta quality is diagnostic and does not change the required return mechanically.",
            ]
            pdf = build_pdf(report_summary, notes)
            st.download_button("Download PDF report", data=pdf, file_name=f"CAPM_{s['ticker']}_report.pdf", mime="application/pdf", type="primary", use_container_width=True)
            st.download_button("Download synchronized returns CSV", data=s["returns"].to_csv(index=True).encode("utf-8"), file_name=f"CAPM_{s['ticker']}_returns.csv", mime="text/csv", use_container_width=True)
            st.download_button("Download sensitivity CSV", data=sens.to_csv(index=False).encode("utf-8"), file_name=f"CAPM_{s['ticker']}_sensitivity.csv", mime="text/csv", use_container_width=True)

    st.markdown(
        f"""
        <div class="footer">
          <span><b>CAPM Analytics Pro</b> · Research & education workspace</span>
          <span>Author: Juan Carlos Muñoz · Universidad del Cauca · Formula-driven, source-traceable analysis</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown('<div class="section-head"><div><h2>How the workspace works</h2><p>A compact workflow with the highest-value components first.</p></div></div>', unsafe_allow_html=True)
    cols = st.columns(4)
    cards = [
        ("01", "Ticker & history", "Yahoo Finance supplies the asset and benchmark history used to build synchronized returns."),
        ("02", "Beta Lab", "Historical OLS, Damodaran industry beta and a transparent blended option."),
        ("03", "CAPM Engine", "The original CAPM equation plus SML, sensitivity and scenario analysis."),
        ("04", "Risk & report", "Quality diagnostics, source lineage and downloadable research outputs."),
    ]
    for col, (n, title, body) in zip(cols, cards):
        with col:
            st.markdown(f'<div class="info-card"><div style="font-weight:900;color:{RED};font-size:.78rem;">{n}</div><div class="info-title">{title}</div><div class="small">{body}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="callout" style="margin-top:14px;"><b>Institutional identity:</b> the interface uses an institutional-inspired navy/red system and references the University of Cauca visual identity. The official university symbol and current identity guidance are published by Universidad del Cauca.</div>', unsafe_allow_html=True)
    st.markdown('<div class="footer"><span>CAPM Analytics Pro · Universidad del Cauca</span><span>Author: Juan Carlos Muñoz</span></div>', unsafe_allow_html=True)
