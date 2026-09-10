from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd


def test_yahoo_adapter_with_fake_provider(monkeypatch):
    # This validates the adapter contract without needing outbound Internet in CI.
    from data import yahoo_finance

    idx = pd.date_range("2023-01-01", periods=10, freq="D")
    fake_df = pd.DataFrame({"Close": np.arange(10.0) + 100}, index=idx)

    class FakeTicker:
        def __init__(self, ticker):
            self.ticker = ticker
            self.fast_info = {"last_price": 109.0}

        def get_info(self):
            return {
                "longName": "Test Corporation",
                "currency": "USD",
                "exchange": "NASDAQ",
                "country": "United States",
                "sector": "Technology",
                "industry": "Software",
                "marketCap": 123_000_000,
                "website": "https://example.com",
            }

    fake_module = types.SimpleNamespace(Ticker=FakeTicker)
    fake_module.download = lambda *args, **kwargs: fake_df.copy()
    monkeypatch.setitem(sys.modules, "yfinance", fake_module)
    yahoo_finance.fetch_history.cache_clear()
    yahoo_finance.fetch_snapshot.cache_clear()

    out = yahoo_finance.fetch_history("TEST", period="1y")
    snap = yahoo_finance.fetch_snapshot("TEST")
    assert not out.empty
    assert out.columns.tolist() == ["Close"]
    assert snap.name == "Test Corporation"
    assert snap.market_cap == 123_000_000


def test_damodaran_parsers_from_local_bytes(tmp_path):
    from data.damodaran import _atomic_csv, _clean

    path = tmp_path / "test.csv"
    df = _clean(pd.DataFrame([[" United States ", "4.87%"], ["Colombia", "10.00%"]], columns=[" Country ", " Equity Risk Premium "]))
    _atomic_csv(df, path)
    read = pd.read_csv(path)
    assert "country" in read.columns
    assert "equity risk premium" in read.columns
    assert len(read) == 2
