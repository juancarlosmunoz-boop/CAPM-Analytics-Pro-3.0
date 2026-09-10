from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TickerSnapshot:
    ticker: str
    name: str
    currency: str
    exchange: str
    country: str
    sector: str
    industry: str
    price: float | None
    market_cap: float | None
    website: str


def normalize_ticker(ticker: str) -> str:
    value = (ticker or "").strip().upper().replace(" ", "")
    if not value:
        raise ValueError("Enter a ticker, for example AAPL or NVDA.")
    if len(value) > 20:
        raise ValueError("Ticker is unusually long.")
    return value


def _yf():
    try:
        import yfinance as yf
        return yf
    except ImportError as exc:
        raise RuntimeError("yfinance is not installed. Add requirements.txt and redeploy the Streamlit app.") from exc


@lru_cache(maxsize=32)
def fetch_history(ticker: str, period: str = "5y", auto_adjust: bool = True) -> pd.DataFrame:
    ticker = normalize_ticker(ticker)
    yf = _yf()
    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=auto_adjust,
        progress=False,
        threads=False,
        group_by="column",
    )
    if df is None or df.empty:
        raise ValueError(f"Yahoo Finance returned no historical data for {ticker}.")
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance may return a two-level frame for one or many tickers.
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.dropna(how="all")


@lru_cache(maxsize=64)
def fetch_snapshot(ticker: str) -> TickerSnapshot:
    ticker = normalize_ticker(ticker)
    yf = _yf()
    obj = yf.Ticker(ticker)
    info: dict[str, Any] = {}
    try:
        info = obj.get_info() or {}
    except Exception:
        info = {}

    price = None
    try:
        fast = obj.fast_info
        last = fast.get("last_price") if hasattr(fast, "get") else None
        price = float(last) if last is not None else None
    except Exception:
        pass

    market_cap = info.get("marketCap")
    return TickerSnapshot(
        ticker=ticker,
        name=str(info.get("longName") or info.get("shortName") or ticker),
        currency=str(info.get("currency") or "N/A"),
        exchange=str(info.get("fullExchangeName") or info.get("exchange") or "N/A"),
        country=str(info.get("country") or "N/A"),
        sector=str(info.get("sector") or "N/A"),
        industry=str(info.get("industry") or "N/A"),
        price=float(price) if price is not None else None,
        market_cap=float(market_cap) if market_cap is not None else None,
        website=str(info.get("website") or ""),
    )
