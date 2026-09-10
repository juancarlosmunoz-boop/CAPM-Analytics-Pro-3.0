from __future__ import annotations

from io import StringIO
from functools import lru_cache

import pandas as pd
import requests

FRED_10Y = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"


@lru_cache(maxsize=1)
def fetch_us_10y() -> tuple[float, str]:
    r = requests.get(FRED_10Y, timeout=20, headers={"User-Agent": "CAPM-Analytics-Pro/2.0"})
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df["DGS10"] = pd.to_numeric(df["DGS10"], errors="coerce")
    latest = df.dropna(subset=["DGS10"])
    if latest.empty:
        raise ValueError("FRED returned no valid DGS10 observation.")
    row = latest.iloc[-1]
    return float(row["DGS10"]) / 100, str(row["observation_date"])
