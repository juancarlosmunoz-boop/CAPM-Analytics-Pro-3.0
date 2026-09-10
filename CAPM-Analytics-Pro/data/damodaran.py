from __future__ import annotations

import io
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests

BASE = "https://pages.stern.nyu.edu/adamodar/New_Home_Page"
URLS = {
    "country_xlsx": "https://www.stern.nyu.edu/~adamodar/pc/datasets/ctryprem.xlsx",
    "country_html": f"{BASE}/datafile/ctrypremtable.htm",
    "global_beta_xls": "https://www.stern.nyu.edu/~adamodar/pc/datasets/betaGlobal.xls",
    "global_beta_html": f"{BASE}/datafile/BetasGlobal.html",
}

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "datasets" / "damodaran"


def _get(url: str, timeout: int = 30) -> bytes:
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "CAPM-Analytics-Pro/2.1"})
    response.raise_for_status()
    return response.content


def _norm(s: object) -> str:
    return re.sub(r"\s+", " ", str(s).replace("\xa0", " ").strip().lower())


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [_norm(c) for c in out.columns]
    out = out.dropna(how="all").dropna(axis=1, how="all")
    return out.reset_index(drop=True)


def _find_header(raw: pd.DataFrame, required_terms: tuple[str, ...]) -> int:
    for i in range(min(len(raw), 30)):
        line = " | ".join(_norm(v) for v in raw.iloc[i].tolist())
        if all(term in line for term in required_terms):
            return i
    return 0


def _parse_xlsx_country(raw: bytes) -> pd.DataFrame:
    book = pd.ExcelFile(io.BytesIO(raw))
    for sheet in book.sheet_names:
        raw_df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet, header=None)
        header = _find_header(raw_df, ("country", "equity risk premium"))
        df = _clean(pd.read_excel(io.BytesIO(raw), sheet_name=sheet, header=header))
        if "country" in df.columns and any("equity risk premium" in c for c in df.columns):
            return df
    raise ValueError("Could not identify the country risk table in Damodaran workbook.")


def _parse_global_beta(raw: bytes) -> pd.DataFrame:
    raw_df = pd.read_excel(io.BytesIO(raw), header=None)
    header = _find_header(raw_df, ("industry name",))
    df = _clean(pd.read_excel(io.BytesIO(raw), header=header))
    if "industry name" not in df.columns:
        raise ValueError("Could not identify Industry Name in Damodaran beta workbook.")
    return df


def _parse_html_table(raw: bytes, required: tuple[str, ...]) -> pd.DataFrame:
    tables = pd.read_html(io.StringIO(raw.decode("utf-8", errors="ignore")))
    for table in sorted(tables, key=lambda x: x.shape[0], reverse=True):
        cleaned = _clean(table)
        if all(term in " | ".join(cleaned.columns) for term in required):
            return cleaned
    raise ValueError("Could not identify the required Damodaran HTML table.")


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


@lru_cache(maxsize=4)
def load_country_risk(refresh: bool = True) -> pd.DataFrame:
    path = CACHE / "country_risk.csv"
    if refresh:
        try:
            df = _parse_xlsx_country(_get(URLS["country_xlsx"]))
            _atomic_csv(df, path)
        except Exception:
            try:
                df = _parse_html_table(_get(URLS["country_html"]), ("country", "equity risk premium"))
                _atomic_csv(df, path)
            except Exception:
                pass
    if not path.exists():
        raise FileNotFoundError("No Damodaran country-risk cache is available.")
    df = pd.read_csv(path)
    return _clean(df)


@lru_cache(maxsize=4)
def load_global_beta(refresh: bool = True) -> pd.DataFrame:
    path = CACHE / "beta_global.csv"
    if refresh:
        try:
            df = _parse_global_beta(_get(URLS["global_beta_xls"]))
            _atomic_csv(df, path)
        except Exception:
            try:
                df = _parse_html_table(_get(URLS["global_beta_html"]), ("industry name",))
                _atomic_csv(df, path)
            except Exception:
                pass
    if not path.exists():
        raise FileNotFoundError("No Damodaran global-beta cache is available.")
    return _clean(pd.read_csv(path))


def find_column(df: pd.DataFrame, *terms: str) -> str | None:
    for column in df.columns:
        norm = _norm(column)
        if all(term in norm for term in terms):
            return column
    return None


def country_row(df: pd.DataFrame, country: str) -> pd.Series:
    country_col = find_column(df, "country") or "country"
    rows = df[df[country_col].astype(str).str.strip().str.lower() == country.strip().lower()]
    if rows.empty:
        raise KeyError(f"Country not found in Damodaran dataset: {country}")
    return rows.iloc[0]
