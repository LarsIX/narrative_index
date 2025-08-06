"""
Granger Causality (both directions) with HC3/HAC covariance, null-imposing
Moving Block Bootstrap p-values, and Benjamini–Hochberg FDR.

For each (Ticker, Year) and each AINI variant, test:
  1) AINI lags Granger-cause returns:      AINI → Return
  2) Return lags Granger-cause AINI:       Return → AINI

Per direction:
- Compute an analytic HC3/HAC Wald/F p-value for joint significance of the
  *causal* lags.
- Compute a null-imposing empirical p-value via Moving Block Bootstrap (MBB).
- Apply BH-FDR within (Ticker, Year, Direction).

Returns one row per (Ticker, Year, AINI_variant, Direction) with:
- F-statistic, df, analytic p, bootstrap p
- FDR-corrected p-values (both analytic and bootstrap)
- Model fit stats for the unrestricted model
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.random import default_rng
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _add_const(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add an intercept column named ``const`` to a design matrix (idempotent).
    """
    return sm.add_constant(X, has_constant="add")


def _fit_ols(
    y: pd.Series,
    X: pd.DataFrame,
    cov: str = "HAC",
    hac_lags: Optional[int] = None,
):
    """
    Fit OLS with either HC3 or HAC covariance.

    Parameters
    ----------
    y : pandas.Series
        Dependent variable (aligned to X).
    X : pandas.DataFrame
        Regressors (no intercept required).
    cov : {"HAC","HC3"}, default "HAC"
        Covariance estimator for inference.
    hac_lags : int, optional
        Newey–West maxlags when cov="HAC".

    Returns
    -------
    statsmodels.regression.linear_model.RegressionResultsWrapper
    """
    if cov.upper() == "HAC":
        return sm.OLS(y, _add_const(X)).fit(cov_type="HAC",
                                            cov_kwds={"maxlags": hac_lags or 1})
    return sm.OLS(y, _add_const(X)).fit(cov_type="HC3")


def _lag(df: pd.DataFrame, col: str, lags: int, prefix: Optional[str] = None) -> pd.DataFrame:
    """
    Create lagged columns for ``col`` with names ``{prefix}_lag1..lags``.
    """
    if lags < 1:
        return df
    base = col if prefix is None else prefix
    out = df.copy()
    for i in range(1, lags + 1):
        out[f"{base}_lag{i}"] = out[col].shift(i)
    return out


def _wald_F_for_zero_coefs(res, target_cols: List[str]) -> Tuple[float, float, int, int]:
    """
    HC3/HAC-robust Wald test (F-form) for joint zero restrictions on target_cols.
    """
    exog_names = res.model.exog_names
    k = len(target_cols)
    R = np.zeros((k, len(exog_names)))
    for i, name in enumerate(target_cols):
        if name not in exog_names:
            raise ValueError(f"Column '{name}' not in model exog.")
        R[i, exog_names.index(name)] = 1.0
    w = res.wald_test(R, use_f=True, scalar=True)  # <-- Fix here
    F_stat = float(w.fvalue)
    p_value = float(w.pvalue)
    df_num = int(w.df_num)
    df_den = int(w.df_denom)
    return F_stat, p_value, df_num, df_den



def _moving_block_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate indices for Moving Block Bootstrap (overlapping blocks).
    """
    if block_size < 1 or block_size > n:
        raise ValueError("block_size must be in [1, n]")
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n - block_size + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
    return idx


def _gc_bootstrap_null(
    y: pd.Series,
    X_restricted: pd.DataFrame,
    X_unrestricted: pd.DataFrame,
    tested_cols: List[str],
    n_boot: int = 1000,
    block_size: int = 5,
    seed: int = 42,
    cov: str = "HAC",
    hac_lags: Optional[int] = None,
) -> Tuple[float, int]:
    """
    Null-imposing MBB p-value for GC Wald F-test (right-tailed).

    Steps
    -----
    1) Fit Restricted model (R): get fitted values and residuals.
    2) Compute observed F from Unrestricted (U) with tested_cols.
    3) For b=1..B:
       - Resample residuals via MBB to e_b.
       - y_b = yhat_R + e_b  (simulation under H0)
       - Fit U on (y_b, XU_b) and record F_b.
    4) p = (#{F_b >= F_obs}+1)/(B_valid+1)

    Returns
    -------
    p_value : float
        Empirical right-tailed p-value.
    n_valid : int
        Number of valid bootstrap statistics used.
    """
    rng = default_rng(seed)


    # 1) Fit Restricted (cov choice irrelevant here)
    res_R = sm.OLS(y, _add_const(X_restricted)).fit()
    yhat_R = res_R.fittedvalues
    ehat_R = y - yhat_R

    # 2) Observed F from Unrestricted with chosen covariance
    res_U = _fit_ols(y, X_unrestricted, cov=cov, hac_lags=hac_lags)
    F_obs, _, _, _ = _wald_F_for_zero_coefs(res_U, tested_cols)

    # 3) Bootstrap under H0
    F_boot: List[float] = []
    n = len(y)
    for _ in range(n_boot):
        try:
            idx = _moving_block_indices(n, block_size, rng)
            e_b = ehat_R.iloc[idx].reset_index(drop=True)
            y_b = yhat_R.reset_index(drop=True) + e_b
            XU_b = X_unrestricted.iloc[idx].reset_index(drop=True)

            res_b = _fit_ols(y_b, XU_b, cov=cov, hac_lags=hac_lags)
            F_b, _, _, _ = _wald_F_for_zero_coefs(res_b, tested_cols)
            if np.isfinite(F_b):
                F_boot.append(float(F_b))
        except Exception:
            # Skip failed draws (singularity etc.)
            continue

    F_boot = np.asarray(F_boot, dtype=float)
    F_boot = F_boot[np.isfinite(F_boot)]
    n_valid = int(F_boot.size)
    if n_valid < max(50, int(0.2 * n_boot)):
        return float("nan"), n_valid

    p = (np.sum(F_boot >= F_obs) + 1) / (n_valid + 1)
    return float(p), n_valid


def _auto_hac_lags(T: int) -> int:
    """
    Newey–West rule-of-thumb for HAC maxlags: floor(4*(T/100)^(2/9)).
    """
    return int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def run_gc_mbboot_fdr(
    aini_df: pd.DataFrame,
    fin_data: pd.DataFrame,
    version: str,
    aini_variants: Optional[List[str]] = None,
    p_ret: int = 1,
    p_x: int = 3,
    min_obs: int = 60,
    block_size: int = 5,
    n_boot: int = 1000,
    seed: int = 42,
    fdr_alpha: float = 0.05,
    cov_for_analytic: str = "HAC",     # "HAC" (recommended) or "HC3"
    hac_lags: Optional[int] = None,    # If None and HAC, auto by T
    save_csv: bool = True,
    outdir: Optional[Path] = None,
    outname: Optional[str] = None,     # If None -> f"granger_causality_{version}.csv"
) -> pd.DataFrame:
    """
    Granger causality (AINI→Return and Return→AINI) with:
      - Analytic Wald/F p-values (HC3 or HAC)
      - Null-imposing MBB empirical p-values
      - Benjamini–Hochberg FDR per (Ticker, Year, Direction)

    Parameters
    ----------
    aini_df : pandas.DataFrame
        Must contain 'date' and one or more AINI variant columns.
    fin_data : pandas.DataFrame
       Contains financial data. Must contain Ticker, Date and closing_.
    version : str
        Suffix used in the output csv filename.
    aini_variants : list of str, optional
        AINI columns to test. Default: ['normalized_AINI','EMA_02','EMA_08','normalized_AINI_growth'].
    p_ret : int
        Return lags used in both directions.
    p_x : int
        AINI lags used in both directions.
    min_obs : int
        Minimum complete cases after lag construction.
    block_size : int
        Block size for MBB.
    n_boot : int
        Number of bootstrap replications.
    seed : int
        RNG seed.
    fdr_alpha : float
        BH-FDR level (applied within each (Ticker, Year, Direction)).
    cov_for_analytic : {"HAC","HC3"}
        Covariance for analytic Wald test p-values. HAC recommended for time series.
    hac_lags : int or None
        HAC maxlags. If None and cov_for_analytic=="HAC", uses NW rule.
    save_csv : bool
        If True, save results to CSV.
    outdir : pathlib.Path or None
        Output directory; if None, defaults to <repo>/data/processed/variables or CWD.
    outname : str or None
        Filename. If None: "granger_causality_{version}.csv". (Convention preserved.)

    Returns
    -------
    pandas.DataFrame
        Rows per (Ticker, Year, AINI_variant, Direction) with:
        - Direction ∈ {"AINI_to_RET","RET_to_AINI"}
        - p_ret, p_x, N_obs, block_size, N_boot, N_boot_valid
        - F_stat, df_num, df_den
        - Original_F_pval (analytic, HC3/HAC)
        - Empirical_F_pval (null-imposing MBB)
        - BH_corr_F_pval, BH_reject_F          (FDR on Empirical_F_pval)
        - BH_corr_F_pval_HC3, BH_reject_F_HC3  (FDR on Original_F_pval)
        - r2_u, adj_r2_u  (unrestricted model fit)
    """
    if aini_variants is None:
        aini_variants = ["normalized_AINI", "EMA_02", "EMA_08", "normalized_AINI_growth"]

    # load financial data
    fin_data['Date'] = pd.to_datetime(fin_data['Date'])

    # ensure sorting
    fin_data['Date'] = pd.to_datetime(fin_data['Date'])
    fin_data = fin_data.sort_values(['Ticker', 'Date'])

    # Calculate log returns by Ticker
    fin_data['log_return'] = fin_data.groupby('Ticker')['Adj Close'].transform(lambda x: np.log(x) - np.log(x.shift(1)))
    fin_data = fin_data.dropna(subset=['log_return'])

    
    # Ensure columns are datetime
    fin_data['date'] = pd.to_datetime(fin_data['Date'])

    # Define thresholds
    threshold_23 = pd.Timestamp('2023-12-31')
    threshold_24 = pd.Timestamp('2024-01-01')
    threshold_25 = pd.Timestamp('2025-01-01')

    # Filter data by year
    fin_data_23 = fin_data[fin_data['date'] < threshold_24]
    fin_data_24 = fin_data[(fin_data['date'] > threshold_23) & (fin_data['date'] < threshold_25)]
    fin_data_25 = fin_data[fin_data['date'] >= threshold_25]

    # overlapping
    fin_data_23_24 = fin_data[fin_data['date'] <= threshold_25]
    fin_data_24_25 = fin_data[fin_data['date'] > threshold_23]
    fin_data_23_24

    fin_data_by_year = {
        2023: fin_data_23,
        2024: fin_data_24,
        2025: fin_data_25,
        "2023_24": fin_data_23_24,
        "2024_25": fin_data_24_25,
        "2023_24_25": fin_data  
    }

    # Ensure datetime in aini data
    aini = aini_df.copy()
    aini["date"] = pd.to_datetime(aini["date"])

    rows = []
    rng = default_rng(seed)

    for year, fin in fin_data_by_year.items():
        f = fin.copy()
        f["date"] = pd.to_datetime(f["date"])
        f = f.rename(columns={"Ticker": "ticker", "LogReturn": "log_return"})
        f = f.sort_values(["ticker", "date"])

        # Merge contemporaneous AINI (lags built after, within ticker)
        merged = pd.merge(f, aini[["date"] + aini_variants], on="date", how="left")

        for ticker, d in merged.groupby("ticker"):
            d = d.sort_values("date").reset_index(drop=True)

            # Build return lags once
            df_lagged = _lag(d, "log_return", p_ret, prefix="ret")

            for var in aini_variants:
                # Build AINI lags
                df_v = _lag(df_lagged, var, p_x, prefix=var)

                ret_lag_cols = [f"ret_lag{i}" for i in range(1, p_ret + 1)]
                xlag_cols = [f"{var}_lag{i}" for i in range(1, p_x + 1)]

                # --------------------------
                # Direction: AINI → Return
                # --------------------------
                y_r = df_v["log_return"]
                XR = df_v[ret_lag_cols]
                XU = df_v[ret_lag_cols + xlag_cols]
                sub_r = pd.concat([y_r, XU], axis=1).dropna()

                # Guard: enough obs and df
                if len(sub_r) >= min_obs and (1 + len(ret_lag_cols) + len(xlag_cols)) < len(sub_r):
                    y_sub = sub_r["log_return"]
                    XR_sub = sub_r[ret_lag_cols]
                    XU_sub = sub_r[ret_lag_cols + xlag_cols]

                    # Analytic (HC3/HAC)
                    use_hac = (cov_for_analytic.upper() == "HAC")
                    _hac_lags = (hac_lags if use_hac and hac_lags is not None
                                 else (_auto_hac_lags(len(sub_r)) if use_hac else None))
                    try:
                        res_u = _fit_ols(y_sub, XU_sub,
                                         cov=("HAC" if use_hac else "HC3"),
                                         hac_lags=_hac_lags)
                        F_stat, p_analytic, df_num, df_den = _wald_F_for_zero_coefs(res_u, xlag_cols)
                    except Exception:
                        res_u = None
                        F_stat, p_analytic = np.nan, np.nan
                        df_num = len(xlag_cols)
                        df_den = max(1, len(sub_r) - (1 + len(ret_lag_cols) + len(xlag_cols)))

                    # Null-imposing MBB p
                    try:
                        p_boot, n_valid = _gc_bootstrap_null(
                            y=y_sub,
                            X_restricted=XR_sub,
                            X_unrestricted=XU_sub,
                            tested_cols=xlag_cols,
                            n_boot=n_boot,
                            block_size=block_size,
                            seed=int(rng.integers(0, 2**31 - 1)),
                            cov=("HAC" if use_hac else "HC3"),
                            hac_lags=_hac_lags,
                        )
                    except Exception:
                        p_boot, n_valid = np.nan, 0

                    rows.append({
                        "Ticker": ticker,
                        "AINI_variant": var,
                        "Year": year,
                        "Direction": "AINI_to_RET",
                        "p_ret": p_ret,
                        "p_x": p_x,
                        "N_obs": len(sub_r),
                        "block_size": block_size,
                        "N_boot": n_boot,
                        "N_boot_valid": n_valid,
                        "F_stat": F_stat,
                        "df_num": df_num,
                        "df_den": df_den,
                        "Original_F_pval": p_analytic,
                        "Empirical_F_pval": p_boot,
                        "r2_u": getattr(res_u, "rsquared", np.nan),
                        "adj_r2_u": getattr(res_u, "rsquared_adj", np.nan),
                    })

                # --------------------------
                # Direction: Return → AINI
                # --------------------------
                y_x = df_v[var]
                XR2 = df_v[xlag_cols] if xlag_cols else pd.DataFrame(index=df_v.index)
                XU2 = pd.concat([XR2, df_v[ret_lag_cols]], axis=1) if ret_lag_cols else XR2
                sub_x = pd.concat([y_x, XU2], axis=1).dropna()

                # Guard: enough obs and df
                if len(sub_x) >= min_obs and (1 + len(xlag_cols) + len(ret_lag_cols)) < len(sub_x):
                    y_sub = sub_x[var]
                    XR_sub = sub_x[xlag_cols] if xlag_cols else pd.DataFrame(index=sub_x.index)
                    # If XR_sub is empty, add a zero column to keep shapes valid in restricted fit
                    if XR_sub.shape[1] == 0:
                        XR_sub = pd.DataFrame({"_zero": np.zeros(len(sub_x), dtype=float)}, index=sub_x.index)
                    XU_sub = sub_x[(xlag_cols if xlag_cols else []) + ret_lag_cols]

                    use_hac = (cov_for_analytic.upper() == "HAC")
                    _hac_lags = (hac_lags if use_hac and hac_lags is not None
                                 else (_auto_hac_lags(len(sub_x)) if use_hac else None))
                    try:
                        res_u = _fit_ols(y_sub, XU_sub,
                                         cov=("HAC" if use_hac else "HC3"),
                                         hac_lags=_hac_lags)
                        F_stat, p_analytic, df_num, df_den = _wald_F_for_zero_coefs(res_u, ret_lag_cols)
                    except Exception:
                        res_u = None
                        F_stat, p_analytic = np.nan, np.nan
                        df_num = len(ret_lag_cols)
                        df_den = max(1, len(sub_x) - (1 + len(xlag_cols) + len(ret_lag_cols)))

                    try:
                        p_boot, n_valid = _gc_bootstrap_null(
                            y=y_sub,
                            X_restricted=XR_sub,
                            X_unrestricted=XU_sub,
                            tested_cols=ret_lag_cols,
                            n_boot=n_boot,
                            block_size=block_size,
                            seed=int(rng.integers(0, 2**31 - 1)),
                            cov=("HAC" if use_hac else "HC3"),
                            hac_lags=_hac_lags,
                        )
                    except Exception:
                        p_boot, n_valid = np.nan, 0

                    rows.append({
                        "Ticker": ticker,
                        "AINI_variant": var,
                        "Year": year,
                        "Direction": "RET_to_AINI",
                        "p_ret": p_ret,
                        "p_x": p_x,
                        "N_obs": len(sub_x),
                        "block_size": block_size,
                        "N_boot": n_boot,
                        "N_boot_valid": n_valid,
                        "F_stat": F_stat,
                        "df_num": df_num,
                        "df_den": df_den,
                        "Original_F_pval": p_analytic,
                        "Empirical_F_pval": p_boot,
                        "r2_u": getattr(res_u, "rsquared", np.nan),
                        "adj_r2_u": getattr(res_u, "rsquared_adj", np.nan),
                    })

    out = pd.DataFrame(rows).sort_values(["Year", "Ticker", "AINI_variant", "Direction"])

    # ----- FDR per (Ticker, Year, Direction) -----
    corrected = []
    for (ticker, year, direction), g in out.groupby(["Ticker", "Year", "Direction"], dropna=False):
        g = g.copy()

        # FDR on Empirical_F_pval (primary)
        p_emp = g["Empirical_F_pval"]
        if p_emp.notna().sum() >= 2:
            rej, p_corr = multipletests(p_emp, alpha=fdr_alpha, method="fdr_bh")[:2]
            g["BH_reject_F"] = rej
            g["BH_corr_F_pval"] = p_corr
        else:
            g["BH_reject_F"] = False
            g["BH_corr_F_pval"] = np.nan

        # FDR on analytic p-values (HC3/HAC)
        p_ana = g["Original_F_pval"]
        if p_ana.notna().sum() >= 2:
            rej2, p_corr2 = multipletests(p_ana, alpha=fdr_alpha, method="fdr_bh")[:2]
            g["BH_reject_F_HC3"] = rej2   # kept key name for compatibility
            g["BH_corr_F_pval_HC3"] = p_corr2
        else:
            g["BH_reject_F_HC3"] = False
            g["BH_corr_F_pval_HC3"] = np.nan

        corrected.append(g)

    out = pd.concat(corrected, ignore_index=True).sort_values(["Year", "Ticker", "AINI_variant", "Direction"])

    # ----- Save (filename convention preserved) -----
    if outname is None:
        outname = f"granger_causality_{version}.csv"

    if save_csv:
        if outdir is None:
            try:
                base = Path(__file__).resolve().parents[2]
            except NameError:
                base = Path.cwd()
            outdir = base / "data" / "processed" / "variables"
        outdir.mkdir(parents=True, exist_ok=True)
        out.to_csv(outdir / outname, index=False)

    return out
