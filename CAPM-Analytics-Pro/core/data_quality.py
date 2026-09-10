from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DataQuality:
    score: int
    label: str
    notes: list[str]


def _clamp(value: int) -> int:
    return max(0, min(100, int(value)))


def score_beta_quality(
    observations: int,
    r_squared: float,
    p_value: float,
    frequency: str,
    beta_gap: float | None = None,
    rolling_dispersion: float | None = None,
) -> DataQuality:
    """Transparent diagnostic score; it does not change the CAPM result."""
    score = 100
    notes: list[str] = []
    if observations < 36:
        score -= 20
        notes.append("Short return history for a long-horizon beta estimate.")
    elif observations < 60:
        score -= 10
        notes.append("Moderate observation count; a longer sample can improve stability.")

    if r_squared < 0.20:
        score -= 20
        notes.append("Low R²: market movements explain a limited share of return variance.")
    elif r_squared < 0.40:
        score -= 10
        notes.append("Moderate R²; beta should be interpreted with other evidence.")

    if p_value > 0.10:
        score -= 20
        notes.append("Beta is not strongly significant at the 10% level.")
    elif p_value > 0.05:
        score -= 10
        notes.append("Beta is borderline significant at the 5% level.")

    if frequency.upper() != "M":
        score -= 5
        notes.append("Monthly returns are commonly preferred for a strategic CAPM estimate.")

    if beta_gap is not None and math.isfinite(beta_gap):
        if beta_gap > 0.50:
            score -= 15
            notes.append("Historical and industry beta differ materially.")
        elif beta_gap > 0.25:
            score -= 5
            notes.append("Historical and industry beta show a noticeable gap.")

    if rolling_dispersion is not None and math.isfinite(rolling_dispersion):
        if rolling_dispersion > 0.50:
            score -= 15
            notes.append("Rolling beta is volatile across time.")
        elif rolling_dispersion > 0.30:
            score -= 5
            notes.append("Rolling beta shows moderate time variation.")

    score = _clamp(score)
    label = "High" if score >= 80 else "Medium" if score >= 60 else "Low"
    if not notes:
        notes.append("Diagnostics are consistent with a comparatively robust beta estimate for this configuration.")
    return DataQuality(score=score, label=label, notes=notes)
