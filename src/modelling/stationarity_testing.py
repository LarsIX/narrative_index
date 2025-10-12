# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron


# ---------------- paths ---------------- #
def _project_paths() -> Tuple[Path, Path]:
    """Returns (variables_dir, tables_dir) and ensures they exist."""
    root = Path(__file__).resolve().parents[2]
    var_path = root / "data" / "processed" / "variables"
    table_path = root / "reports" / "tables"
    var_path.mkdir(parents=True, exist_ok=True)
    table_path.mkdir(parents=True, exist_ok=True)
    return var_path, table_path


def _financial_raw_dir() -> Path:
    """Project-relative raw financial data dir."""
    return Path(__file__).resolve().parents[2] / "data" / "raw" / "financial"


# ---------------- utilities ---------------- #
def _clean_series(x: pd.Series) -> pd.Series:
    """Numeric, finite, no NaNs."""
    return (
        pd.to_numeric(x, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )


def _coerce_datetime_column(df: pd.DataFrame, prefer: Iterable[str] = ("Date", "date")) -> pd.DataFrame:
    """Ensure there is a proper datetime 'Date' column, sorted and deduped."""
    df = df.copy()
    chosen = None
    for c in prefer:
        if c in df.columns:
            chosen = c
            break
    if chosen is None:
        for c in ("Datetime", "timestamp", "TIME", "TIME_STAMP"):
            if c in df.columns:
                chosen = c
                break
    if chosen is None:
        try:
            df["Date"] = pd.to_datetime(df.index, errors="coerce")
        except Exception as e:
            raise ValueError(f"Could not determine date column: {e}")
    else:
        df["Date"] = pd.to_datetime(df[chosen], errors="coerce")

    df = df.dropna(subset=["Date"]).sort_values("Date")
    df = df[~df["Date"].duplicated(keep="last")]
    return df


def _pick_close_like_column(df: pd.DataFrame) -> str:
    """
    Return a sensible price column for 'close':
    Preference: 'Adj Close', 'AdjClose', 'Adj_Close', 'Close', 'close', 'Adj Close*'.
    Fallback: numeric column with highest median (excluding Volume/Turnover).
    """
    for c in ("Adj Close", "AdjClose", "Adj_Close", "Close", "close", "Adj Close*"):
        if c in df.columns:
            return c
    num = df.select_dtypes(include=[np.number]).copy()
    num.drop(columns=[c for c in ("Volume", "volume", "Turnover") if c in num.columns],
             inplace=True, errors="ignore")
    if num.empty:
        raise ValueError("No numeric price-like column found.")
    return num.median().sort_values(ascending=False).index[0]


def _subset_periods(df: pd.DataFrame, date_col: str = "Date") -> dict:
    """Return dict of common period subsets based on a date column."""
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    return {
        "2023": d[d[date_col] < "2024-01-01"],
        "2024": d[(d[date_col] >= "2024-01-01") & (d[date_col] < "2025-01-01")],
        "2025": d[d[date_col] >= "2025-01-01"],
        "2023-2024": d[d[date_col] < "2025-01-01"],
        "2024-2025": d[d[date_col] >= "2024-01-01"],
        "2023-2025": d,
    }


def _run_adf_pp_kpss(
    series: pd.Series,
    kpss_regression: str = "c",
    kpss_nlags: str | int = "auto",
) -> Tuple[float, float, float, float, float, float]:
    """Return (adf_stat, adf_p, pp_stat, pp_p, kpss_stat, kpss_p)."""
    s = _clean_series(series)
    adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
    pp = PhillipsPerron(s)
    pp_stat = float(pp.stat)
    pp_p = float(pp.pvalue)
    try:
        kpss_stat, kpss_p, *_ = kpss(s, regression=kpss_regression, nlags=kpss_nlags)
    except Exception:
        # KPSS can fail on edge cases; record NA rather than explode
        kpss_stat, kpss_p = np.nan, np.nan
    return adf_stat, adf_p, pp_stat, pp_p, kpss_stat, kpss_p


def _agree_stationary(adf_p: float, pp_p: float, kpss_p: float, alpha: float) -> bool:
    """
    Conservative 'consensus' rule:
      ADF & PP reject unit root (p < alpha) AND KPSS fails to reject stationarity (p > alpha).
    """
    if any(pd.isna([adf_p, pp_p, kpss_p])):
        return False
    return (adf_p < alpha) and (pp_p < alpha) and (kpss_p > alpha)


# ---------------- loaders: indices ---------------- #
def _load_index_logreturns(csv_path: Path, ticker: str) -> pd.DataFrame:
    """
    Load an index CSV, pick a close-like column, compute LogReturn = diff(log(price)).
    Output: ['Date','Ticker','ClosePrice','LogReturn'] (first diff row dropped).
    """
    df = pd.read_csv(csv_path)
    df = _coerce_datetime_column(df)
    close_col = _pick_close_like_column(df)
    price = pd.to_numeric(df[close_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    price = price.where(price > 0)
    logret = np.log(price).diff()
    out = pd.DataFrame(
        {"Date": df["Date"], "Ticker": ticker, "ClosePrice": price, "LogReturn": logret}
    ).dropna(subset=["LogReturn"]).reset_index(drop=True)
    return out


def load_sox_gspc_logreturns() -> pd.DataFrame:
    """
    Load project-relative ^SOX and ^GSPC (full_2023_2025 CSVs) and return stacked log-returns.
    Expects:
      data/raw/financial/^SOX_full_2023_2025.csv
      data/raw/financial/^GSPC_full_2023_2025.csv
    """
    raw_dir = _financial_raw_dir()
    sox_csv = raw_dir / "^SOX_full_2023_2025.csv"
    gspc_csv = raw_dir / "^GSPC_full_2023_2025.csv"
    if not sox_csv.exists():
        raise FileNotFoundError(f"Missing file: {sox_csv}")
    if not gspc_csv.exists():
        raise FileNotFoundError(f"Missing file: {gspc_csv}")
    sox = _load_index_logreturns(sox_csv, "SOX")
    gspc = _load_index_logreturns(gspc_csv, "GSPC")
    return pd.concat([sox, gspc], ignore_index=True)


# ---------------- stationarity: financial log returns ---------------- #
def test_stationarity_fin_logreturns(
    fin_data: pd.DataFrame,
    alpha: float = 0.05,
    min_obs: int = 20,
    date_col: str = "Date",
    ticker_col: str = "Ticker",
    value_col: str = "LogReturn",
    label: str = "fin_logreturns",
    kpss_regression: str = "c",
    kpss_nlags: str | int = "auto",
) -> pd.DataFrame:
    """
    Run ADF, PP, KPSS on LogReturn (per Ticker, per Period subsets).
    Assumes fin_data has columns: [date_col, ticker_col, value_col].
    """
    var_path, table_path = _project_paths()

    df = fin_data.copy()
    for c in (date_col, ticker_col, value_col):
        if c not in df.columns:
            raise ValueError(f"fin_data missing required column: {c}")

    df["Date"] = pd.to_datetime(df[date_col])
    df = df.sort_values(["Date", ticker_col]).reset_index(drop=True)

    subsets = _subset_periods(df, date_col="Date")

    rows: List[dict] = []
    for period, dsub in subsets.items():
        for ticker, g in dsub.groupby(ticker_col, dropna=True):
            s = _clean_series(g[value_col])
            n = int(s.shape[0])
            if n < min_obs:
                rows.append({
                    "Period": period, "Ticker": ticker, "Variable": value_col, "N": n,
                    "ADF_stat": np.nan, "ADF_p": np.nan,
                    "PP_stat": np.nan, "PP_p": np.nan,
                    "KPSS_stat": np.nan, "KPSS_p": np.nan,
                    "agree_stationarity": False,
                    "error": f"too_few_obs(<{min_obs})"
                })
                continue
            try:
                adf_stat, adf_p, pp_stat, pp_p, kpss_stat, kpss_p = _run_adf_pp_kpss(
                    s, kpss_regression=kpss_regression, kpss_nlags=kpss_nlags
                )
                rows.append({
                    "Period": period, "Ticker": ticker, "Variable": value_col, "N": n,
                    "ADF_stat": adf_stat, "ADF_p": adf_p,
                    "PP_stat": pp_stat, "PP_p": pp_p,
                    "KPSS_stat": kpss_stat, "KPSS_p": kpss_p,
                    "agree_stationarity": _agree_stationary(adf_p, pp_p, kpss_p, alpha),
                    "error": ""
                })
            except Exception as e:
                rows.append({
                    "Period": period, "Ticker": ticker, "Variable": value_col, "N": n,
                    "ADF_stat": np.nan, "ADF_p": np.nan,
                    "PP_stat": np.nan, "PP_p": np.nan,
                    "KPSS_stat": np.nan, "KPSS_p": np.nan,
                    "agree_stationarity": False,
                    "error": str(e)
                })

    out = pd.DataFrame(rows).sort_values(["Period", "Ticker"]).reset_index(drop=True)

    csv_file = var_path / f"stationarity_tests_{label}.csv"
    html_file = table_path / f"stationarity_tests_{label}.html"
    out.to_csv(csv_file, index=False)
    out.to_html(html_file, index=False)
    print(f"Saved FIN LogReturn stationarity -> CSV: {csv_file} | HTML: {html_file}")
    return out


def run_stationarity_sox_gspc(
    alpha: float = 0.05,
    min_obs: int = 20,
    kpss_regression: str = "c",
    kpss_nlags: str | int = "auto",
    label: str = "sox_gspc_logrets",
) -> pd.DataFrame:
    """Load ^SOX and ^GSPC, compute log returns, test those (not prices)."""
    fin_df = load_sox_gspc_logreturns()
    return test_stationarity_fin_logreturns(
        fin_data=fin_df,
        alpha=alpha,
        min_obs=min_obs,
        date_col="Date",
        ticker_col="Ticker",
        value_col="LogReturn",
        label=label,
        kpss_regression=kpss_regression,
        kpss_nlags=kpss_nlags,
    )


# ---------------- stationarity: AINI variants ---------------- #
_AINI_LABEL_MAP = {
    "EMA_02": r"EMA_{0.2}",
    "EMA_08": r"EMA_{0.8}",
    "normalized_AINI": r"AINI^{\mathrm{norm}}",
    "normalized_AINI_z": r"AINI^{z}",
}


def _apply_aini_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "AINI_variant" in df.columns:
        df = df.copy()
        df["AINI_variant"] = df["AINI_variant"].astype(str).map(lambda v: _AINI_LABEL_MAP.get(v, v))
        df = df.rename(columns={"AINI_variant": r"AINI^{var}"})
    return df


def test_stationarity_aini_variants(
    aini_data: pd.DataFrame,
    aini_cols: Optional[Iterable[str]] = None,
    alpha: float = 0.05,
    min_obs: int = 20,
    kpss_regression: str = "c",
    kpss_nlags: str | int = "auto",
    label: str = "aini",
) -> pd.DataFrame:
    """
    Run ADF, PP, KPSS for AINI measures over standard periods.
    Expected columns: ['date', <AINI columns>]
    Default tests: EMA_02, EMA_08, normalized_AINI, normalized_AINI_z
    """
    var_path, table_path = _project_paths()

    if aini_cols is None:
        aini_cols = ["EMA_02", "EMA_08", "normalized_AINI", "normalized_AINI_z"]

    df = aini_data.copy()
    if "date" not in df.columns:
        raise ValueError("aini_data must have a 'date' column.")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    subsets = {
        "2023": df[df["date"] < "2024-01-01"],
        "2024": df[(df["date"] >= "2024-01-01") & (df["date"] < "2025-01-01")],
        "2025": df[df["date"] >= "2025-01-01"],
        "2023-2024": df[df["date"] < "2025-01-01"],
        "2024-2025": df[df["date"] >= "2024-01-01"],
        "2023-2025": df,
    }

    rows: List[dict] = []
    for period, dsub in subsets.items():
        for col in aini_cols:
            if col not in dsub.columns:
                rows.append({
                    "Period": period,
                    "AINI_variant": col,
                    "N": 0,
                    "ADF_stat": np.nan, "ADF_p": np.nan,
                    "PP_stat": np.nan, "PP_p": np.nan,
                    "KPSS_stat": np.nan, "KPSS_p": np.nan,
                    "agree_stationarity": False,
                    "error": "missing_column"
                })
                continue

            s = _clean_series(dsub[col])
            n = int(s.shape[0])
            if n < min_obs:
                rows.append({
                    "Period": period,
                    "AINI_variant": col,
                    "N": n,
                    "ADF_stat": np.nan, "ADF_p": np.nan,
                    "PP_stat": np.nan, "PP_p": np.nan,
                    "KPSS_stat": np.nan, "KPSS_p": np.nan,
                    "agree_stationarity": False,
                    "error": f"too_few_obs(<{min_obs})"
                })
                continue

            try:
                adf_stat, adf_p, pp_stat, pp_p, kpss_stat, kpss_p = _run_adf_pp_kpss(
                    s, kpss_regression=kpss_regression, kpss_nlags=kpss_nlags
                )
                rows.append({
                    "Period": period,
                    "AINI_variant": col,
                    "N": n,
                    "ADF_stat": adf_stat, "ADF_p": adf_p,
                    "PP_stat": pp_stat, "PP_p": pp_p,
                    "KPSS_stat": kpss_stat, "KPSS_p": kpss_p,
                    "agree_stationarity": _agree_stationary(adf_p, pp_p, kpss_p, alpha),
                    "error": ""
                })
            except Exception as e:
                rows.append({
                    "Period": period,
                    "AINI_variant": col,
                    "N": n,
                    "ADF_stat": np.nan, "ADF_p": np.nan,
                    "PP_stat": np.nan, "PP_p": np.nan,
                    "KPSS_stat": np.nan, "KPSS_p": np.nan,
                    "agree_stationarity": False,
                    "error": str(e)
                })

    out = pd.DataFrame(rows).sort_values(["Period", "AINI_variant"]).reset_index(drop=True)
    out = _apply_aini_labels(out)

    csv_file = var_path / f"stationarity_tests_{label}.csv"
    html_file = table_path / f"stationarity_tests_{label}.html"
    out.to_csv(csv_file, index=False)
    out.to_html(html_file, index=False)
    print(f"Saved AINI stationarity -> CSV: {csv_file} | HTML: {html_file}")
    return out


# ---------------- stationarity: VIX log growth ---------------- #
def test_stationarity_vix(
    vix_csv: Optional[Path] = None,
    date_col: str = "date",
    value_col: str = "log_growth_closed",
    alpha: float = 0.05,
    min_obs: int = 20,
    kpss_regression: str = "c",
    kpss_nlags: str | int = "auto",
    label: str = "vix_log_growth",
) -> pd.DataFrame:
    """
    Run stationarity tests on VIX log-growth series.
    Default expects: data/processed/variables/z_scores_VIX.csv with [date_col, value_col].
    """
    var_path, table_path = _project_paths()
    if vix_csv is None:
        vix_csv = var_path / "z_scores_VIX.csv"

    df = pd.read_csv(vix_csv)
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in {vix_csv}")
    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found in {vix_csv}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df[date_col])
    df = df.sort_values("Date").reset_index(drop=True)

    rows: List[dict] = []
    for period, dsub in _subset_periods(df, date_col="Date").items():
        s = _clean_series(dsub[value_col])
        n = int(s.shape[0])
        if n < min_obs:
            rows.append({
                "Period": period, "Variable": value_col, "N": n,
                "ADF_stat": np.nan, "ADF_p": np.nan,
                "PP_stat": np.nan, "PP_p": np.nan,
                "KPSS_stat": np.nan, "KPSS_p": np.nan,
                "agree_stationarity": False,
                "error": f"too_few_obs(<{min_obs})"
            })
            continue

        try:
            adf_stat, adf_p, pp_stat, pp_p, kpss_stat, kpss_p = _run_adf_pp_kpss(
                s, kpss_regression=kpss_regression, kpss_nlags=kpss_nlags
            )
            rows.append({
                "Period": period, "Variable": value_col, "N": n,
                "ADF_stat": adf_stat, "ADF_p": adf_p,
                "PP_stat": pp_stat, "PP_p": pp_p,
                "KPSS_stat": kpss_stat, "KPSS_p": kpss_p,
                "agree_stationarity": _agree_stationary(adf_p, pp_p, kpss_p, alpha),
                "error": ""
            })
        except Exception as e:
            rows.append({
                "Period": period, "Variable": value_col, "N": n,
                "ADF_stat": np.nan, "ADF_p": np.nan,
                "PP_stat": np.nan, "PP_p": np.nan,
                "KPSS_stat": np.nan, "KPSS_p": np.nan,
                "agree_stationarity": False,
                "error": str(e)
            })

    out = pd.DataFrame(rows).sort_values(["Period"]).reset_index(drop=True)

    csv_file = var_path / f"stationarity_tests_{label}.csv"
    html_file = table_path / f"stationarity_tests_{label}.html"
    out.to_csv(csv_file, index=False)
    out.to_html(html_file, index=False)
    print(f"Saved VIX stationarity -> CSV: {csv_file} | HTML: {html_file}")
    return out
