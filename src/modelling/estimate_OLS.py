"""
Same-day OLS with HC3/HAC covariance, null-imposing residual MBB p-values,
and Benjamini–Hochberg FDR within (Ticker, Year).

Model (per Ticker, Year, AINI variant)
--------------------------------------
r_t = alpha + beta_aini * AINI_t + beta_ar * r_{t-1} + e_t

Outputs (per row)
-----------------
- alpha, beta_aini, beta_ar, r2, adj_r2, N_obs
- Original_coef_pval       : analytic p-value for beta_aini (HC3/HAC)
- Empirical_coef_pval      : null-imposing residual MBB p-value (two-sided)
- BH_corr_coef_pval, BH_reject_coef       : FDR on Empirical_coef_pval
- BH_corr_coef_pval_HC3, BH_reject_coef_HC3 : FDR on Original_coef_pval
- se_aini, t_aini          : SE and t-stat for beta_aini (analytic)
- block_size, N_boot, N_boot_valid

Notes
-----
- HAC (Newey–West) is preferable in time series; HC3 ignores autocorrelation.
- The residual MBB **imposes H0: beta_aini = 0** by simulating y* from the
  restricted AR(1) model and refitting the unrestricted model.
- FDR is applied per (Ticker, Year) across AINI variants.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.random import default_rng
from statsmodels.stats.multitest import multipletests


# ----------------------------
# Internal helpers
# ----------------------------

def _add_const(X: pd.DataFrame) -> pd.DataFrame:
    """Add an intercept column named 'const' (idempotent)."""
    return sm.add_constant(X, has_constant="add")


def _fit_ols(
    y: pd.Series,
    X: pd.DataFrame,
    cov: str = "HAC",
    hac_lags: Optional[int] = None,
):
    """
    Fit OLS with either HC3 or HAC covariance.
    """
    if cov.upper() == "HAC":
        return sm.OLS(y, _add_const(X)).fit(cov_type="HAC",
                                            cov_kwds={"maxlags": hac_lags or 1})
    return sm.OLS(y, _add_const(X)).fit(cov_type="HC3")


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


def _auto_hac_lags(T: int) -> int:
    """Newey–West rule-of-thumb: floor(4*(T/100)^(2/9))."""
    return int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))


def _coef_mbb_pvalue_null(
    y: pd.Series,
    X: pd.DataFrame,
    coef_name: str,
    n_boot: int = 1000,
    block_size: int = 5,
    seed: int = 42,
    cov: str = "HAC",
    hac_lags: Optional[int] = None,
) -> Tuple[float, int]:
    """
    Two-sided p-value for H0: beta_{coef_name}=0 using null-imposing residual MBB.
    Studentizes with the same covariance (HC3/HAC) as the analytic fit.

    Steps
    -----
    1) Restricted (R): drop coef_name from X to get XR; fit y ~ const + XR.
       Save fitted yhat_R and residuals ehat_R.
    2) Observed t from Unrestricted (U): fit y ~ const + X with cov; T_obs = t(coef_name).
    3) For b=1..B:
       - Draw block indices; e_b = ehat_R[idx].
       - Generate y_b = yhat_R + e_b  (data under H0).
       - Fit U on (y_b, X_b) with same cov; record t_b = t(coef_name).
    4) p = (#{|t_b| >= |T_obs|} + 1)/(B_valid + 1)

    Returns
    -------
    (p_value, n_valid_boot)
    """
    rng = default_rng(seed)
    if coef_name not in X.columns:
        return float("nan"), 0

    # Restricted regressors exclude the tested regressor
    XR = X.drop(columns=[coef_name])
    # 1) Fit Restricted to get yhat and residuals (cov irrelevant here)
    res_R = sm.OLS(y, _add_const(XR)).fit()
    yhat_R = res_R.fittedvalues
    ehat_R = y - yhat_R

    # 2) Observed studentized coefficient from Unrestricted with chosen covariance
    res_U = _fit_ols(y, X, cov=cov, hac_lags=hac_lags)
    if coef_name not in res_U.params.index:
        return float("nan"), 0
    T_obs = res_U.tvalues[coef_name]

    # 3) Bootstrap under H0
    n = len(y)
    T_boot: List[float] = []
    for _ in range(n_boot):
        try:
            idx = _moving_block_indices(n, block_size, rng)
            e_b = ehat_R.iloc[idx].reset_index(drop=True)
            y_b = yhat_R.reset_index(drop=True) + e_b
            X_b = X.iloc[idx].reset_index(drop=True)

            res_b = _fit_ols(y_b, X_b, cov=cov, hac_lags=hac_lags)
            t_b = res_b.tvalues.get(coef_name, np.nan)
            if np.isfinite(t_b):
                T_boot.append(float(t_b))
        except Exception:
            continue

    T_boot = np.asarray(T_boot, dtype=float)
    T_boot = T_boot[np.isfinite(T_boot)]
    n_valid = int(T_boot.size)
    if n_valid < max(50, int(0.2 * n_boot)):
        return float("nan"), n_valid

    p = (np.sum(np.abs(T_boot) >= np.abs(T_obs)) + 1) / (n_valid + 1)
    return float(p), n_valid


# ----------------------------
# Main entry point
# ----------------------------

def run_sameday_ols_mbboot_fdr(
    aini_df: pd.DataFrame,
    fin_data_by_year: Dict[int | str, pd.DataFrame],
    aini_variants: Optional[List[str]] = None,
    min_obs: int = 30,
    block_size: int = 5,
    n_boot: int = 1000,
    seed: int = 42,
    fdr_alpha: float = 0.05,
    cov_for_analytic: str = "HAC",      # "HAC" (recommended) or "HC3"
    hac_lags: Optional[int] = None,     # None -> auto by T if HAC
    standardize_aini: bool = False,     # z-score AINI within (Ticker, Year)
    save_csv: bool = True,
    outdir: Optional[Path] = None,
    outname: str = "ols_sameday_mbboot_fdr.csv",
) -> pd.DataFrame:
    """
    Fit same-day OLS by (Ticker, Year) with HC3/HAC and null-imposing MBB p-values, then apply FDR.

    Parameters
    ----------
    aini_df : pandas.DataFrame
        Columns: 'date' and one or more AINI variant columns.
    fin_data_by_year : dict[int|str, pandas.DataFrame]
        year -> DataFrame with columns ['date','Ticker','LogReturn'].
    aini_variants : list of str, optional
        If None: ['normalized_AINI','EMA_02','EMA_08','normalized_AINI_growth'].
    min_obs : int
        Minimum complete rows after merging and lagging.
    block_size : int
        MBB block length.
    n_boot : int
        Number of bootstrap replications.
    seed : int
        RNG seed.
    fdr_alpha : float
        BH-FDR level within (Ticker, Year).
    cov_for_analytic : {"HAC","HC3"}
        Covariance for analytic t-tests; HAC recommended.
    hac_lags : int or None
        HAC maxlags; if None and HAC, use Newey–West rule based on sample T.
    standardize_aini : bool
        If True, z-score each AINI variant within (Ticker, Year) before regression.
    save_csv : bool
        Save results to CSV.
    outdir : pathlib.Path or None
        Output directory; if None, defaults to <repo>/data/processed/variables or CWD.
    outname : str
        Filename (kept unchanged as requested).

    Returns
    -------
    pandas.DataFrame
        One row per (Ticker, Year, AINI_variant) with coefficients, p-values (analytic + bootstrap),
        FDR-adjusted p-values, fit stats, and bootstrap meta.
    """
    if aini_variants is None:
        aini_variants = ["normalized_AINI", "EMA_02", "EMA_08", "normalized_AINI_growth"]

    aini = aini_df.copy()
    aini["date"] = pd.to_datetime(aini["date"])

    rows = []
    master_rng = default_rng(seed)

    for year, fin in fin_data_by_year.items():
        f = fin.copy()
        f["date"] = pd.to_datetime(f["date"])
        f = f.rename(columns={"Ticker": "ticker", "LogReturn": "log_return"})
        f = f.sort_values(["ticker", "date"])
        f["ret_lag1"] = f.groupby("ticker")["log_return"].shift(1)

        merged = pd.merge(f, aini[["date"] + aini_variants], on="date", how="left")

        for ticker, d in merged.groupby("ticker"):
            d = d.sort_values("date").reset_index(drop=True)

            # Optional standardization within (Ticker, Year)
            d_std = d.copy()
            if standardize_aini:
                for var in aini_variants:
                    mu = d_std[var].mean(skipna=True)
                    sd = d_std[var].std(skipna=True)
                    if np.isfinite(sd) and sd > 0:
                        d_std[var] = (d_std[var] - mu) / sd

            for var in aini_variants:
                sub = d_std[["log_return", "ret_lag1", var]].dropna()
                if len(sub) < min_obs:
                    continue

                y = sub["log_return"]
                X = sub[["ret_lag1", var]]

                # Analytic fit (HC3/HAC)
                use_hac = (cov_for_analytic.upper() == "HAC")
                _hac = _auto_hac_lags(len(sub)) if (use_hac and hac_lags is None) else hac_lags
                try:
                    res = _fit_ols(y, X, cov=("HAC" if use_hac else "HC3"), hac_lags=_hac)
                    alpha = res.params.get("const", np.nan)
                    beta_aini = res.params.get(var, np.nan)
                    beta_ar = res.params.get("ret_lag1", np.nan)
                    se_aini = res.bse.get(var, np.nan)
                    t_aini = res.tvalues.get(var, np.nan)
                    p_ana = res.pvalues.get(var, np.nan)
                    r2 = res.rsquared
                    adj_r2 = res.rsquared_adj
                except Exception:
                    alpha = beta_aini = beta_ar = np.nan
                    se_aini = t_aini = p_ana = np.nan
                    r2 = adj_r2 = np.nan

                # Null-imposing residual MBB p-value for the AINI coefficient
                try:
                    p_boot, n_valid = _coef_mbb_pvalue_null(
                        y=y,
                        X=X,
                        coef_name=var,
                        n_boot=n_boot,
                        block_size=block_size,
                        seed=int(master_rng.integers(0, 2**31 - 1)),
                        cov=("HAC" if use_hac else "HC3"),
                        hac_lags=_hac,
                    )
                except Exception:
                    p_boot, n_valid = np.nan, 0

                rows.append({
                    "Ticker": ticker,
                    "AINI_variant": var,
                    "Year": year,
                    "N_obs": len(sub),
                    "block_size": block_size,
                    "N_boot": n_boot,
                    "N_boot_valid": n_valid,
                    # coefficients & inference
                    "alpha": alpha,
                    "beta_aini": beta_aini,
                    "se_aini": se_aini,
                    "t_aini": t_aini,
                    "Original_coef_pval": p_ana,      # analytic HC3/HAC p-value
                    "Empirical_coef_pval": p_boot,    # residual MBB p-value (two-sided)
                    "beta_ret_lag1": beta_ar,
                    "pval_ret_lag1": res.pvalues.get("ret_lag1", np.nan) if hasattr(res, "pvalues") else np.nan,
                    # fit
                    "r2": r2,
                    "adj_r2": adj_r2,
                })

    out = pd.DataFrame(rows).sort_values(["Year", "Ticker", "AINI_variant"])

    # ----- FDR per (Ticker, Year) -----
    corrected = []
    for (ticker, year), g in out.groupby(["Ticker", "Year"], dropna=False):
        g = g.copy()

        # FDR on Empirical_coef_pval (primary)
        p_emp = g["Empirical_coef_pval"]
        if p_emp.notna().sum() >= 2:
            rej, p_corr = multipletests(p_emp, alpha=fdr_alpha, method="fdr_bh")[:2]
            g["BH_reject_coef"] = rej
            g["BH_corr_coef_pval"] = p_corr
        else:
            g["BH_reject_coef"] = False
            g["BH_corr_coef_pval"] = np.nan

        # FDR on analytic p-values (HC3/HAC)
        p_ana = g["Original_coef_pval"]
        if p_ana.notna().sum() >= 2:
            rej2, p_corr2 = multipletests(p_ana, alpha=fdr_alpha, method="fdr_bh")[:2]
            g["BH_reject_coef_HC3"] = rej2   # keep key name for compatibility with your GC tables
            g["BH_corr_coef_pval_HC3"] = p_corr2
        else:
            g["BH_reject_coef_HC3"] = False
            g["BH_corr_coef_pval_HC3"] = np.nan

        corrected.append(g)

    out = pd.concat(corrected, ignore_index=True).sort_values(["Year", "Ticker", "AINI_variant"])

    # ----- Save -----
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
