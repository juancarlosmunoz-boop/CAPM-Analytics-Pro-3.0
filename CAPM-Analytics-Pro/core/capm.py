from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CAPMResult:
    risk_free_rate: float
    beta: float
    expected_market_return: float
    market_risk_premium: float
    required_return: float
    equity_risk_premium: float


def _finite(value: float) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def calculate_capm(risk_free_rate: float, beta: float, expected_market_return: float) -> CAPMResult:
    """Core CAPM equation: E(Ri) = Rf + beta * (E(Rm) - Rf)."""
    if not all(_finite(v) for v in (risk_free_rate, beta, expected_market_return)):
        raise ValueError("CAPM inputs must be finite numbers.")
    if not -0.50 <= risk_free_rate <= 1.50:
        raise ValueError("Risk-free rate must be between -50% and 150%.")
    if not -10.0 <= beta <= 20.0:
        raise ValueError("Beta must be between -10 and 20.")

    mrp = expected_market_return - risk_free_rate
    required = risk_free_rate + beta * mrp
    return CAPMResult(
        risk_free_rate=float(risk_free_rate),
        beta=float(beta),
        expected_market_return=float(expected_market_return),
        market_risk_premium=float(mrp),
        required_return=float(required),
        equity_risk_premium=float(beta * mrp),
    )


def calculate_from_erp(risk_free_rate: float, beta: float, equity_risk_premium: float) -> CAPMResult:
    """Calculate CAPM when a market ERP/MRP is supplied directly."""
    if not _finite(equity_risk_premium):
        raise ValueError("Equity risk premium must be finite.")
    return calculate_capm(risk_free_rate, beta, risk_free_rate + equity_risk_premium)
