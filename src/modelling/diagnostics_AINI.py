"""
stationarity_testing.py

Assess stationarity of AINI indices and financial variables using ADF and Phillips–Perron.
Saves CSV/HTML to reports/tables and returns pandas DataFrames.
"""

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import PhillipsPerron


# ------------------------------ #
# Helpers
# ------------------------------ #

def _project_paths() -> Tuple[Path, Path]:
    root = Path(__file__).resolve().parents[2]
    var_path = root / "data" / "processed" / "variables"
    table_path = root / "reports" / "tables"
    var_path.mkdir(parents=True, exist_ok=True)
    table_path.mkdir(parents=True, exist_ok=True)
    return var_path, table_path


def _clean_series(x: pd.Series) -> pd.Series:
    # Ensure numeric and finite
    return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def _run_adf_pp(series: pd.Series) -> Tuple[float, float, float, float]:
    # ADF (autolag AIC), PP (default Bartlett kernel)
    adf_stat, adf_p, *_ = adfuller(series, autolag="AIC")
    pp = PhillipsPerron(series)
    return adf_stat, adf_p, float(pp.stat), float(pp.pvalue)



# ------------------------------ #
# AINI variants
# ------------------------------ #

def test_stationarity_aini_variants(
    aini_data: pd.DataFrame,
    variants: Optional[str] = None,
    window: Optional[int] = None,
    aini_cols: Optional[Iterable[str]] = None,
    alpha: float = 0.05,
    min_obs: int = 20,
) -> pd.DataFrame:
    """
    Run ADF & PP on selected AINI columns over multiple time windows.

    Parameters
    ----------
    aini_data : DataFrame with at least 'date' and the AINI columns.
    variants  : str label for the dataset (used in filenames).
    window    : None, 0, 1, 2 – only used in filenames/metadata.
    aini_cols : iterable of column names to test. Defaults to common AINI fields.
    alpha     : significance level for stationarity agreement.
    min_obs   : minimum observations to attempt a unit-root test.

    Returns
    -------
    DataFrame of results; also saved to CSV/HTML.
    """
    var_path, table_path = _project_paths()

    if aini_cols is None:
        aini_cols = ["normalized_AINI", "EMA_02", "EMA_08"]

    df = aini_data.copy()
    df["date"] = pd.to_datetime(df["date"])

    subsets = {
        "2023": df[df["date"] < "2024-01-01"],
        "2024": df[(df["date"] >= "2024-01-01") & (df["date"] < "2025-01-01")],
        "2025": df[df["date"] >= "2025-01-01"],
        "2023_24": df[df["date"] < "2025-01-01"],
        "2024_25": df[df["date"] >= "2024-01-01"],
        "2023_25": df,  # full
    }

    rows: List[dict] = []
    for period, dsub in subsets.items():
        for col in aini_cols:
            try:
                s = _clean_series(dsub[col])
                n = int(s.shape[0])
                if n < min_obs:
                    rows.append({
                        "Period": period,
                        "AINI_variant": col,
                        "Context_window": window,
                        "N": n,
                        "ADF_stat": np.nan, "ADF_p": np.nan,
                        "PP_stat": np.nan, "PP_p": np.nan,
                        "agree_stationarity": False,
                        "error": f"too_few_obs(<{min_obs})"
                    })
                    continue

                adf_stat, adf_p, pp_stat, pp_p = _run_adf_pp(s)
                rows.append({
                    "Period": period,
                    "AINI_variant": col,
                    "Context_window": window,
                    "N": n,
                    "ADF_stat": adf_stat, "ADF_p": adf_p,
                    "PP_stat": pp_stat, "PP_p": pp_p,
                    "agree_stationarity": (adf_p < alpha) and (pp_p < alpha),
                    "error": ""
                })

            except Exception as e:
                rows.append({
                    "Period": period,
                    "AINI_variant": col,
                    "Context_window": window,
                    "N": int(dsub.shape[0]),
                    "ADF_stat": np.nan, "ADF_p": np.nan,
                    "PP_stat": np.nan, "PP_p": np.nan,
                    "agree_stationarity": False,
                    "error": str(e)
                })

    out = pd.DataFrame(rows).sort_values(["Period", "AINI_variant"]).reset_index(drop=True)

    # Filenames 
    suffix = f"window{window}_" if (window is not None) else ""
    label = variants or "aini"
    csv_file = var_path / f"stationarity_tests_{suffix}{label}.csv"
    html_file = table_path / f"stationarity_tests_{suffix}{label}.html"

    out.to_csv(csv_file, index=False)
    out.to_html(html_file, index=False)
    print(f"Saved AINI stationarity results -> CSV: {csv_file} | HTML: {html_file}")
    return out


# ------------------------------ #
# Financial variables
# ------------------------------ #

def test_stationarity_fin_variables(
    fin_data: pd.DataFrame,
    fin_vars: Optional[Iterable[str]] = None,
    alpha: float = 0.05,
    min_obs: int = 20,
    date_col: str = "Date",
    ticker_col: str = "Ticker",
    label: str = "fin_var"
) -> pd.DataFrame:
    """
    Run ADF & PP on financial variables, grouped by ticker and time window.

    Parameters
    ----------
    fin_data  : DataFrame with date_col, ticker_col, and variables in fin_vars.
    fin_vars  : iterable of variables (e.g., ["Adj Close", "LogReturn"]).
    alpha     : significance level for agreement.
    min_obs   : minimum observations.
    date_col  : name of the date column in fin_data.
    ticker_col: name of the ticker column.
    label     : filename label.

    Returns
    -------
    DataFrame; also saved to CSV/HTML.
    """
    var_path, table_path = _project_paths()

    if fin_vars is None:
        fin_vars = ["Adj Close","LogReturn"]

    df = fin_data.copy()
    df["date"] = pd.to_datetime(df[date_col])

    subsets = {
        "2023": df[df["date"] < "2024-01-01"],
        "2024": df[(df["date"] >= "2024-01-01") & (df["date"] < "2025-01-01")],
        "2025": df[df["date"] >= "2025-01-01"],
        "2023_24": df[df["date"] < "2025-01-01"],
        "2024_25": df[df["date"] >= "2024-01-01"],
        "2023_25": df,
    }

    rows: List[dict] = []
    for period, dsub in subsets.items():
        for ticker, g in dsub.groupby(ticker_col, dropna=True):
            for var in fin_vars:
                try:
                    s = _clean_series(g[var])
                    n = int(s.shape[0])
                    if n < min_obs:
                        rows.append({
                            "Period": period, "Ticker": ticker, "Variable": var, "N": n,
                            "ADF_stat": np.nan, "ADF_p": np.nan,
                            "PP_stat": np.nan, "PP_p": np.nan,
                            "agree_stationarity": False,
                            "error": f"too_few_obs(<{min_obs})"
                        })
                        continue

                    adf_stat, adf_p, pp_stat, pp_p = _run_adf_pp(s)
                    rows.append({
                        "Period": period, "Ticker": ticker, "Variable": var, "N": n,
                        "ADF_stat": adf_stat, "ADF_p": adf_p,
                        "PP_stat": pp_stat, "PP_p": pp_p,
                        "agree_stationarity": (adf_p < alpha) and (pp_p < alpha),
                        "error": ""
                    })
                except Exception as e:
                    rows.append({
                        "Period": period, "Ticker": ticker, "Variable": var, "N": int(g.shape[0]),
                        "ADF_stat": np.nan, "ADF_p": np.nan,
                        "PP_stat": np.nan, "PP_p": np.nan,
                        "agree_stationarity": False,
                        "error": str(e)
                    })

    out = pd.DataFrame(rows).sort_values(["Period", "Ticker", "Variable"]).reset_index(drop=True)

    csv_file = var_path / f"stationarity_tests_{label}.csv"
    html_file = table_path / f"stationarity_tests_{label}.html"
    out.to_csv(csv_file, index=False)
    out.to_html(html_file, index=False)
    print(f"Saved FIN stationarity results -> CSV: {csv_file} | HTML: {html_file}")
    return out
