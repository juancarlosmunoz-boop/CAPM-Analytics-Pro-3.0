from __future__ import annotations


def pct(x: float | None, digits: int = 2) -> str:
    return "N/A" if x is None else f"{x * 100:.{digits}f}%"


def num(x: float | None, digits: int = 2) -> str:
    return "N/A" if x is None else f"{x:,.{digits}f}"


def money(x: float | None, currency: str = "USD") -> str:
    if x is None:
        return "N/A"
    return f"{currency} {x:,.0f}"
