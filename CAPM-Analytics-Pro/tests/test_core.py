import numpy as np
import pandas as pd

from core.beta import estimate_beta, prepare_returns, rolling_beta
from core.capm import calculate_capm, calculate_from_erp
from core.data_quality import score_beta_quality
from core.scenarios import sensitivity_table, scenario_table
from reports.pdf_report import build_pdf


def test_capm_equation():
    result = calculate_capm(0.04, 1.2, 0.10)
    assert round(result.required_return, 6) == 0.112
    assert round(result.market_risk_premium, 6) == 0.06


def test_capm_from_erp():
    result = calculate_from_erp(0.04, 1.2, 0.06)
    assert round(result.expected_market_return, 6) == 0.10
    assert round(result.required_return, 6) == 0.112


def test_beta_estimation_close_to_known_beta():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2018-01-31", periods=84, freq="ME")
    market = pd.Series(rng.normal(0.01, 0.05, len(idx)), index=idx)
    asset = pd.Series(0.002 + 1.4 * market + rng.normal(0, 0.01, len(idx)), index=idx)
    result = estimate_beta(pd.DataFrame({"asset": asset, "market": market}))
    assert 1.2 < result.beta < 1.6
    assert result.observations == 84
    assert 0 <= result.r_squared <= 1


def test_returns_prepare_and_rolling_beta():
    idx = pd.date_range("2020-01-01", periods=800, freq="D")
    asset = pd.Series(np.linspace(100, 140, len(idx)), index=idx)
    market = pd.Series(np.linspace(80, 120, len(idx)), index=idx)
    out = prepare_returns(asset, market, frequency="M")
    assert set(out.columns) == {"asset", "market"}
    assert len(out) > 15
    rb = rolling_beta(out, window=12)
    assert len(rb) >= 5


def test_quality_score_and_tables():
    q = score_beta_quality(60, 0.45, 0.03, "M", beta_gap=0.12, rolling_dispersion=0.15)
    assert q.score >= 80
    sens = sensitivity_table(0.04, 0.06, [0.8, 1.0, 1.2])
    assert list(sens.columns) == ["Beta", "Required Return"]
    scen = scenario_table(0.04, 1.2, {"Bear": 0.04, "Base": 0.06, "Bull": 0.08})
    assert len(scen) == 3


def test_pdf_report_is_valid_bytes():
    pdf = build_pdf({"Ticker": "TEST", "CAPM required return": "11.20%"}, ["Yahoo Finance", "Damodaran"])
    assert pdf[:4] == b"%PDF"
    assert len(pdf) > 1000
