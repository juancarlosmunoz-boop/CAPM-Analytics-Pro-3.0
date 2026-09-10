from __future__ import annotations

import numpy as np
import pandas as pd

from .capm import calculate_capm


def sensitivity_table(risk_free_rate: float, market_risk_premium: float, betas: np.ndarray | list[float]) -> pd.DataFrame:
    rows = []
    for beta in betas:
        result = calculate_capm(risk_free_rate, float(beta), risk_free_rate + market_risk_premium)
        rows.append({"Beta": float(beta), "Required Return": result.required_return})
    return pd.DataFrame(rows)


def scenario_table(
    risk_free_rate: float,
    beta: float,
    mrp_values: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for name, mrp in mrp_values.items():
        result = calculate_capm(risk_free_rate, beta, risk_free_rate + mrp)
        rows.append({
            "Scenario": name,
            "Risk-free": risk_free_rate,
            "Market Risk Premium": mrp,
            "Expected Market Return": result.expected_market_return,
            "Required Return": result.required_return,
        })
    return pd.DataFrame(rows)
