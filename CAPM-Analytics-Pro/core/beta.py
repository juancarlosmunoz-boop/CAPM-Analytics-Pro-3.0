from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class BetaResult:
    beta: float
    alpha_period: float
    alpha_annualized: float
    r_squared: float
    correlation: float
    observations: int
    beta_std_error: float
    beta_p_value: float
    residual_std_period: float
    asset_vol_annualized: float
    market_vol_annualized: float
    downside_beta: Optional[float]
    period: str


def prepare_returns(
    asset_prices: pd.Series,
    market_prices: pd.Series,
    frequency: str = "M",
) -> pd.DataFrame:
    """Align adjusted prices and convert them to synchronized returns."""
    frequency = frequency.upper()
    if frequency not in {"D", "W", "M"}:
        raise ValueError("frequency must be D, W or M")

    asset = pd.to_numeric(asset_prices, errors="coerce").dropna().sort_index()
    market = pd.to_numeric(market_prices, errors="coerce").dropna().sort_index()
    asset.index = pd.to_datetime(asset.index).tz_localize(None)
    market.index = pd.to_datetime(market.index).tz_localize(None)
    frame = pd.concat({"asset": asset, "market": market}, axis=1).dropna()
    if frame.empty:
        raise ValueError("No overlapping price observations were found.")

    if frequency == "M":
        frame = frame.resample("ME").last().pct_change()
    elif frequency == "W":
        frame = frame.resample("W-FRI").last().pct_change()
    else:
        frame = frame.pct_change()
    return frame.dropna()


def estimate_beta(returns: pd.DataFrame, min_observations: int = 24) -> BetaResult:
    """OLS estimate of asset returns on market returns."""
    if not {"asset", "market"}.issubset(returns.columns):
        raise ValueError("Returns must contain 'asset' and 'market' columns.")
    frame = returns[["asset", "market"]].dropna().copy()
    if len(frame) < min_observations:
        raise ValueError(f"At least {min_observations} paired observations are required; got {len(frame)}.")
    if frame["market"].var(ddof=1) <= 0:
        raise ValueError("Market returns have zero variance; beta cannot be estimated.")

    x = sm.add_constant(frame["market"])
    model = sm.OLS(frame["asset"], x).fit()
    beta = float(model.params.iloc[1])
    alpha_p = float(model.params.iloc[0])
    r2 = float(model.rsquared)
    corr = float(frame["asset"].corr(frame["market"]))

    # Annualization depends on the estimation frequency.
    periods = 252 if getattr(returns.index, "inferred_freq", None) in {"B", "C"} else None
    if periods is None:
        # Monthly/weekly/daily are explicit in the public API; infer from cadence.
        median_days = frame.index.to_series().diff().dt.days.median()
        periods = 12 if median_days >= 20 else 52 if median_days >= 5 else 252
    alpha_a = (1 + alpha_p) ** periods - 1 if alpha_p > -1 else periods * alpha_p
    downside = frame.loc[frame["market"] < 0]
    downside_beta = None
    if len(downside) >= 12 and downside["market"].var(ddof=1) > 0:
        downside_beta = float(downside["asset"].cov(downside["market"]) / downside["market"].var(ddof=1))

    return BetaResult(
        beta=beta,
        alpha_period=alpha_p,
        alpha_annualized=float(alpha_a),
        r_squared=r2,
        correlation=corr,
        observations=len(frame),
        beta_std_error=float(model.bse.iloc[1]),
        beta_p_value=float(model.pvalues.iloc[1]),
        residual_std_period=float(model.resid.std(ddof=1)),
        asset_vol_annualized=float(frame["asset"].std(ddof=1) * np.sqrt(periods)),
        market_vol_annualized=float(frame["market"].std(ddof=1) * np.sqrt(periods)),
        downside_beta=downside_beta,
        period={12: "monthly", 52: "weekly", 252: "daily"}.get(periods, "periodic"),
    )


def rolling_beta(returns: pd.DataFrame, window: int = 36) -> pd.Series:
    if window < 12:
        raise ValueError("Rolling beta window must be at least 12 observations.")
    frame = returns[["asset", "market"]].dropna()
    values: list[float] = []
    idx = []
    for i in range(window, len(frame) + 1):
        chunk = frame.iloc[i - window:i]
        var = chunk["market"].var(ddof=1)
        values.append(np.nan if var <= 0 else float(chunk["asset"].cov(chunk["market"]) / var))
        idx.append(chunk.index[-1])
    return pd.Series(values, index=idx, name=f"Rolling Beta ({window})")
