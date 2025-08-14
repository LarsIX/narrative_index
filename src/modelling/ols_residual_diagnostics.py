import csv
from pathlib import Path
from typing import Dict, Iterable, Tuple, Optional, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, acorr_breusch_godfrey, het_arch


# ------------------------------ #
# Helpers                        #
# ------------------------------ #

def _reports_tables_dir() -> Path:
    """Resolve reports/tables relative to this file."""
    filepath = Path(__file__).resolve()
    base = filepath.parents[2]
    out_dir = base / "reports" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def ensure_fin_formatting(fin_data: pd.DataFrame,
                          date_col: str = "Date",
                          ticker_col: str = "Ticker") -> pd.DataFrame:
    """Ensure datetime, sorting, and LogReturn."""
    df = fin_data.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values([ticker_col, date_col])
    if "LogReturn" not in df.columns:
        if "Adj Close" not in df.columns:
            raise ValueError("fin_data must have 'Adj Close' to compute LogReturn.")
        df["LogReturn"] = df.groupby(ticker_col)["Adj Close"].transform(
            lambda x: np.log(x) - np.log(x.shift(1))
        )
    return df


def _fit_and_diag(y: np.ndarray,
                  X: np.ndarray,
                  cov_type: str = "HC3",
                  cov_kwds: Optional[dict] = None,
                  lb_lags: Iterable[int] = (12, 24),
                  bg_maxlag: int = 12,
                  arch_lags: int = 12) -> Tuple[Dict[str, float], sm.regression.linear_model.RegressionResultsWrapper]:
    """Fit OLS and compute residual diagnostics."""
    y = np.asarray(y, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y = y[mask]
    X = X[mask, :]
    if y.size == 0:
        raise ValueError("No finite observations after cleaning.")

    Xc = sm.add_constant(X, has_constant="add")
    res = sm.OLS(y, Xc).fit(cov_type=cov_type, cov_kwds=(cov_kwds or {}))

    diags: Dict[str, float] = {
        "nobs": float(res.nobs),
        "r2": float(res.rsquared) if res.rsquared is not None else np.nan,
        "adj_r2": float(res.rsquared_adj) if res.rsquared_adj is not None else np.nan,
    }

    n = int(res.nobs)
    lb_lags = tuple(int(min(L, max(1, n - 1))) for L in lb_lags)
    bg_maxlag = int(min(bg_maxlag, max(1, n // 5)))
    arch_lags = int(min(arch_lags, max(1, n // 5)))

    for L in lb_lags:
        try:
            diags[f"lb_pval_L{L}"] = float(acorr_ljungbox(res.resid, lags=L, return_df=True)["lb_pvalue"].iloc[-1])
        except Exception:
            diags[f"lb_pval_L{L}"] = np.nan

    try:
        diags["bg_pval"] = float(acorr_breusch_godfrey(res, nlags=bg_maxlag)[3])
    except Exception:
        diags["bg_pval"] = np.nan

    try:
        diags["arch_lm_pval"] = float(het_arch(res.resid, maxlag=arch_lags)[1])
    except Exception:
        diags["arch_lm_pval"] = np.nan

    try:
        Lmax = max(lb_lags)
        diags[f"lb_sq_pval_L{Lmax}"] = float(acorr_ljungbox(res.resid ** 2, lags=Lmax, return_df=True)["lb_pvalue"].iloc[-1])
    except Exception:
        diags[f"lb_sq_pval_L{max(lb_lags)}"] = np.nan

    return diags, res


# ----------------------------------- #
# Main: Univariate OLS diagnostics    #
# ----------------------------------- #

def run_ols_residual_diagnostics(
    fin_df: pd.DataFrame,
    aini_df: pd.DataFrame,
    var: str,
    X_cols: Optional[List[str]] = None,
    y_col: str = "LogReturn",
    date_col_fin: str = "Date",
    ticker_col: str = "Ticker",
    date_col_aini: str = "date",
    by_ticker: bool = True,
    lb_lags: Iterable[int] = (12, 24),
    bg_maxlag: int = 12,
    arch_lags: int = 12,
    cov_type: str = "HC3",
    cov_kwds: Optional[dict] = None,
    write_csv: bool = True,
    write_html: bool = True,
) -> pd.DataFrame:
    """Merge FIN and AINI on date and run univariate OLS+diagnostics for each X_col."""
    if X_cols is None:
        X_cols = ["normalized_AINI", "EMA_08", "normalized_AINI_growth"]

    fin_df = ensure_fin_formatting(fin_df, date_col=date_col_fin, ticker_col=ticker_col).copy()
    aini = aini_df.copy()
    aini[date_col_aini] = pd.to_datetime(aini[date_col_aini], errors="coerce")

    missing = [c for c in X_cols if c not in aini.columns]
    if missing:
        raise ValueError(f"AINI DF missing columns: {missing}")

    merge_cols = [date_col_aini] + X_cols
    merged = fin_df.merge(aini[merge_cols], left_on=date_col_fin, right_on=date_col_aini, how="inner")
    if merged.empty:
        raise ValueError("Merge produced no rows; check date columns and ranges.")

    results_rows: List[Dict[str, float]] = []
    out_dir = _reports_tables_dir()

    groups = merged.groupby(ticker_col) if by_ticker else [(None, merged)]

    for tick, g in groups:
        for xcol in X_cols:
            cols_needed = [y_col, xcol]
            g2 = g[cols_needed].dropna()
            if g2.empty:
                continue
            y = g2[y_col].to_numpy()
            X = g2[[xcol]].to_numpy()
            diags, res = _fit_and_diag(y, X, cov_type=cov_type, cov_kwds=cov_kwds,
                                       lb_lags=lb_lags, bg_maxlag=bg_maxlag, arch_lags=arch_lags)
            row = {
                "Ticker": tick if tick is not None else "ALL",
                "model": f"UNI({xcol})",
                **diags
            }
            results_rows.append(row)

    results = pd.DataFrame(results_rows).sort_values(["Ticker", "model"]).reset_index(drop=True)

    if write_csv:
        results.to_csv(out_dir / f"ols_diag_{var}.csv", index=False)
    if write_html:
        results.to_html(out_dir / f"ols_diag_{var}.html", index=False)

    return results
