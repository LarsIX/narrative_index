from __future__ import annotations

"""
Same-day OLS with HAC/HC3, null-imposing residual Moving-Block Bootstrap (MBB),
and Benjamini–Hochberg FDR correction within (Ticker, Year).

Data-science-style overview
---------------------------
- For each (Year) and each Ticker within that year:
  - Merge daily log returns with selected AINI variants on the date.
  - For each AINI variant:
      * Use the variant as-is (RAW).
      * If the variant is 'normalized_AINI', also create a per-(ticker,year)
        standardized version (z-score) and estimate a second model (STD).
  - Fit OLS with HAC/HC3 analytic SEs.
  - Compute a null-imposing residual Moving-Block Bootstrap (MBB) empirical p-value
    for the AINI coefficient, preserving serial dependence with blocks.
- Within each (Ticker, Year), apply BH-FDR to RAW and STD p-values separately.

Model (per Ticker, Year, AINI variant, Scale in {raw, std for normalized_AINI only})
------------------------------------------------------------------------------------
r_t = alpha + beta_aini * AINI_t + beta_ar * r_{t-1} + (optional controls) + e_t

Outputs (per row)
-----------------
- alpha_{scale}, beta_aini_{scale}, se_aini_{scale}, t_aini_{scale}
- r2_{scale}, adj_r2_{scale}, N_obs
- Original_coef_pval_{scale}   : analytic p-value (HC3 or HAC) for beta_aini
- Empirical_coef_pval_{scale}  : null-imposing residual MBB p-value (two-sided)
- N_boot_valid_{scale}         : number of valid bootstrap draws
- beta_ret_lag1_{scale}, pval_ret_lag1_{scale}
- (optional) beta_control_{scale}, pval_control_{scale} for each control
- BH_corr_* and BH_reject_* per scale (applied within (Ticker, Year))

Notes
-----
- Standardization is computed *in-sample* for the current (ticker, year) slice.
- If the standard deviation is 0 or not finite, the STD series becomes NaN and
  will be skipped by `min_obs` after dropna().
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.random import default_rng
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed


# ---------------------------
# Helpers: BH + OLS + Bootstrap
# ---------------------------

def bh_with_fallback(p: pd.Series, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benjamini–Hochberg FDR with robust fallbacks (keeps original shape).

    Why: Some (Ticker, Year) groups might have 0 or 1 valid p-values. BH normally
    expects ≥2. We keep consistent shapes and sensible behavior in edge cases.

    Returns
    -------
    p_corr : np.ndarray
        BH-corrected p-values, aligned to input; NaN where input was NaN.
    reject : np.ndarray
        Boolean BH decisions, aligned to input.
    """
    p = pd.to_numeric(p, errors="coerce")  # preserve length; NaN for non-numeric
    mask = p.notna()

    p_corr = np.full(len(p), np.nan, dtype=float)
    reject = np.zeros(len(p), dtype=bool)

    n = int(mask.sum())
    if n == 0:
        return p_corr, reject
    if n == 1:
        idx = np.flatnonzero(mask.values)[0]
        val = float(p.iloc[idx])
        p_corr[idx] = val
        reject[idx] = (val <= alpha)
        return p_corr, reject

    rej, p_adj = multipletests(p[mask].values, alpha=alpha, method="fdr_bh")[:2]
    p_corr[mask.values] = p_adj
    reject[mask.values] = rej
    return p_corr, reject


def _add_const(X: pd.DataFrame) -> pd.DataFrame:
    """Safely add an intercept column (avoids duplicating an existing constant)."""
    return sm.add_constant(X, has_constant="add")


def _fit_ols(
    y: pd.Series,
    X: pd.DataFrame,
    cov: str = "HAC",
    hac_lags: Optional[int] = None,
):
    """
    Fit OLS with either HAC (Newey–West) or HC3 covariance.

    Parameters
    ----------
    cov : {'HAC','HC3'}
        HAC handles serial correlation/heteroskedasticity; HC3 is a small-sample
        heteroskedasticity-robust option without autocorrelation correction.
    hac_lags : int or None
        Max lag for HAC. If None and cov='HAC', we auto-choose later.

    Returns
    -------
    statsmodels RegressionResultsWrapper
    """
    if cov.upper() == "HAC":
        return sm.OLS(y, _add_const(X)).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags or 1})
    return sm.OLS(y, _add_const(X)).fit(cov_type="HC3")


def _moving_block_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Moving-Block Bootstrap indices:
    - Sample starting positions uniformly.
    - Build consecutive blocks of length `block_size`.
    - Concatenate and truncate to length n.
    """
    if block_size < 1 or block_size > n:
        raise ValueError("block_size must be in [1, n]")
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n - block_size + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
    return idx


def _auto_hac_lags(T: int) -> int:
    """Andrews-style rule-of-thumb: floor(4 * (T/100)^(2/9))."""
    return int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))


def _boot_t_once(
    i: int,
    yhat_R: np.ndarray,
    ehat_R: np.ndarray,
    X: pd.DataFrame,
    coef_name: str,
    block_size: int,
    cov: str,
    hac_lags: Optional[int],
    base_seed: int,
) -> float:
    """
    One MBB iteration under the restricted DGP:
    - Resample restricted residuals in blocks.
    - Generate y^b = yhat_R + ehat_R^b.
    - Refit unrestricted model on (y^b, X).
    - Return t-stat of the target coefficient.
    """
    try:
        rng = np.random.default_rng(base_seed + i)
        n = yhat_R.shape[0]
        idx = _moving_block_indices(n, block_size, rng)
        y_b = pd.Series(yhat_R + ehat_R[idx])
        X_b = X.iloc[idx].reset_index(drop=True)
        res_b = _fit_ols(y_b, X_b, cov=cov, hac_lags=hac_lags)
        return float(res_b.tvalues.get(coef_name, np.nan))
    except Exception:
        # Fail-safe: treat as invalid draw
        return np.nan


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
    Empirical two-sided p-value for a coefficient via null-imposing residual MBB.

    Steps
    -----
    1) Fit restricted model (drop `coef_name`) → yhat_R, ehat_R.
    2) MBB on ehat_R → ehat_R^b; generate y^b = yhat_R + ehat_R^b.
    3) Refit unrestricted model on each bootstrap sample; collect t-stats.
    4) Compare |T_obs| to bootstrap |t| distribution, use +1 smoothing.

    Returns
    -------
    (p_value, n_valid_draws)
    """
    if coef_name not in X.columns:
        return float("nan"), 0

    # Restricted fit (impose H0 in the DGP)
    XR = X.drop(columns=[coef_name])
    res_R = sm.OLS(y, _add_const(XR)).fit()
    yhat_R = res_R.fittedvalues.to_numpy()
    ehat_R = (y - res_R.fittedvalues).to_numpy()

    # Observed unrestricted t-stat of the target coefficient
    res_U = _fit_ols(y, X, cov=cov, hac_lags=hac_lags)
    if coef_name not in res_U.params.index:
        return float("nan"), 0
    T_obs = float(res_U.tvalues[coef_name])

    # Parallel bootstrap
    base_seed = int(seed)
    t_vals = Parallel(n_jobs=-1, backend="loky", batch_size="auto")(
        delayed(_boot_t_once)(
            i, yhat_R, ehat_R, X, coef_name, block_size, cov, hac_lags, base_seed
        )
        for i in range(n_boot)
    )

    t_vals = np.asarray(t_vals, dtype=float)
    t_vals = t_vals[np.isfinite(t_vals)]
    n_valid = int(t_vals.size)

    # Avoid degenerate small-sample p-values
    if n_valid < max(50, int(0.2 * n_boot)):
        return float("nan"), n_valid

    # Two-sided empirical p-value with +1 smoothing
    p = (np.sum(np.abs(t_vals) >= abs(T_obs)) + 1) / (n_valid + 1)
    return float(p), n_valid


# -------------------------------------------------------
# Main entry: run_sameday_ols_mbboot_fdr (with checks)
# -------------------------------------------------------

def run_sameday_ols_mbboot_fdr(
    aini_df: pd.DataFrame,
    fin_data_by_year: Dict[int | str, pd.DataFrame],
    aini_variants: Optional[List[str]] = None,
    min_obs: int = 30,
    block_size: int = 5,
    n_boot: int = 1000,
    seed: int = 42,
    fdr_alpha: float = 0.05,
    cov_for_analytic: str = "HAC",
    hac_lags: Optional[int] = None,
    control_vars: Optional[List[str]] = None,
    save_csv: bool = True,
    outdir: Optional[Path] = None,
    outname: str = "ols_sameday_mbboot_fdr.csv",
    price_col: str = "Adj Close",
) -> pd.DataFrame:
    """
    Same-day OLS with analytic and MBB empirical p-values; BH-FDR within (Ticker, Year).

    Important behavior change
    -------------------------
    - Only 'normalized_AINI' gets a z-scored (STD) version per (ticker, year).
      All other variants are estimated only in RAW scale.

    Parameters
    ----------
    aini_df : DataFrame
        Must contain 'date' and the AINI variant columns you specify in `aini_variants`.
    fin_data_by_year : dict[int|str, DataFrame]
        For each year: DataFrame with at least ['Ticker', price_col, 'date'].
    aini_variants : list[str], optional
        AINI columns to use. Defaults to ['normalized_AINI', 'EMA_02', 'EMA_08'].
    min_obs : int
        Minimum observations required per (Ticker, Year, Var, Scale).
    block_size : int
        Moving block size for bootstrap resampling of residuals.
    n_boot : int
        Number of bootstrap draws.
    seed : int
        RNG seed base; per-group seeds are drawn from it.
    fdr_alpha : float
        Alpha for BH-FDR.
    cov_for_analytic : {'HAC','HC3'}
        Covariance estimator for analytic p-values.
    hac_lags : int, optional
        Maxlags for HAC. If None and cov='HAC', picked automatically.
    control_vars : list[str], optional
        Column names (in financial data) to include as controls.
    save_csv : bool
        If True, write CSV to outdir/outname.
    outdir : Path, optional
        Directory for output CSV. Defaults to project-like path or CWD.
    outname : str
        Output filename.
    price_col : str
        Price column to compute log returns (default 'Adj Close').

    Returns
    -------
    DataFrame
        One row per (Ticker, Year, AINI variant, Scale).
    """
    # Default variants if none provided
    if aini_variants is None:
        aini_variants = ["normalized_AINI", "EMA_02", "EMA_08"]

    # --- Validate and sanitize AINI data ---
    aini = aini_df.copy()
    if "date" not in aini.columns:
        raise KeyError("AINI data is missing required column: 'date'")
    aini["date"] = pd.to_datetime(aini["date"], errors="coerce")
    if aini["date"].isna().any():
        raise ValueError("AINI 'date' contains NaT after parsing; check input format.")

    missing_aini = set(aini_variants) - set(aini.columns)
    if missing_aini:
        raise KeyError(f"AINI data missing required variant columns: {missing_aini}")

    rows: List[Dict] = []
    master_rng = default_rng(seed)  # central RNG to derive per-group seeds

    # Iterate over yearly partitions of financial data
    for year, fin in fin_data_by_year.items():
        f = fin.copy()

        # --- Financial data checks ---
        required_fin_cols = {"Ticker", price_col, "date"}
        missing_fin = required_fin_cols - set(f.columns)
        if missing_fin:
            raise KeyError(f"Financial data missing columns: {missing_fin}")

        f["date"] = pd.to_datetime(f["date"], errors="coerce")
        if f["date"].isna().any():
            raise ValueError(f"Financial 'date' contains NaT after parsing for year={year}.")

        # Compute daily log returns by ticker; drop the first NA per ticker
        f["log_return"] = (
            f.groupby("Ticker")[price_col].transform(lambda x: np.log(x) - np.log(x.shift(1)))
        )
        f = f.dropna(subset=["log_return"]).copy()

        # Ticker housekeeping and AR(1) control
        f = f.rename(columns={"Ticker": "ticker"})
        f = f.sort_values(["ticker", "date"])
        f["ret_lag1"] = f.groupby("ticker")["log_return"].shift(1)

        # Controls existence check (fail early if requested but missing)
        if control_vars:
            missing_ctrl = set(control_vars) - set(f.columns)
            if missing_ctrl:
                raise KeyError(f"Financial data missing requested control variables: {missing_ctrl}")

        # Merge returns with AINI variants on the date
        merged = pd.merge(f, aini[["date"] + aini_variants], on="date", how="left")

        # Iterate tickers within the current year
        for ticker, d in merged.groupby("ticker"):
            d = d.sort_values("date").reset_index(drop=True)

            # Loop over variants: RAW for all; STD only for 'normalized_AINI'
            for var in aini_variants:
                # Parse to numeric to avoid dtype issues (strings -> NaN)
                raw_series = pd.to_numeric(d[var], errors="coerce")

                # Decide scales to estimate for this variant
                scales: List[Tuple[str, pd.Series]] = [("raw", raw_series)]

                # Compute per-(ticker,year) z-scores ONLY for 'normalized_AINI'
                if var == "normalized_AINI":
                    mu = raw_series.mean(skipna=True)
                    sd = raw_series.std(skipna=True)
                    if pd.notna(sd) and float(sd) > 0.0:
                        std_series = (raw_series - mu) / (sd if sd > 1e-12 else 1e-12)
                    else:
                        # Degenerate or undefined variance → no usable STD series
                        std_series = pd.Series(np.nan, index=raw_series.index)
                    scales.append(("std", std_series))

                # Estimate per scale (RAW always; STD only present for normalized_AINI)
                for scale_tag, series in scales:
                    # Build aligned design: y | X[var, lagged return, optional controls]
                    parts = [d["log_return"], d["ret_lag1"], series.rename(var)]
                    if control_vars:
                        parts.extend([d[c] for c in control_vars])

                    # Align and drop NA rows across y and X
                    sub = pd.concat(parts, axis=1).dropna()
                    if len(sub) < min_obs:
                        continue  # sample too small; skip this (variant, scale) for this ticker-year

                    y = sub["log_return"]
                    X = sub.drop(columns="log_return")

                    # HAC lag selection (auto if requested and not provided)
                    use_hac = (cov_for_analytic.upper() == "HAC")
                    _hac = _auto_hac_lags(len(sub)) if (use_hac and hac_lags is None) else hac_lags

                    # Analytic fit and summary stats
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
                        pval_ret_lag1 = res.pvalues.get("ret_lag1", np.nan)
                    except Exception:
                        # Defensive defaults on numerical issues
                        alpha = beta_aini = beta_ar = np.nan
                        se_aini = t_aini = p_ana = np.nan
                        r2 = adj_r2 = np.nan
                        pval_ret_lag1 = np.nan

                    # Empirical p-value via null-imposing MBB
                    try:
                        p_boot, n_valid = _coef_mbb_pvalue_null(
                            y=y,
                            X=X,
                            coef_name=var,
                            n_boot=n_boot,
                            block_size=block_size,
                            seed=int(master_rng.integers(0, 2**31 - 1)),  # unique seed per group
                            cov=("HAC" if use_hac else "HC3"),
                            hac_lags=_hac,
                        )
                    except Exception:
                        p_boot, n_valid = np.nan, 0

                    # Collect output row for this (ticker, year, variant, scale)
                    row = {
                        "Ticker": ticker,
                        "AINI_variant": var,
                        "Year": year,
                        "N_obs": len(sub),
                        "block_size": block_size,
                        "N_boot": n_boot,
                        f"N_boot_valid_{scale_tag}": n_valid,
                        # Coef & inference for AINI
                        f"alpha_{scale_tag}": alpha,
                        f"beta_aini_{scale_tag}": beta_aini,
                        f"se_aini_{scale_tag}": se_aini,
                        f"t_aini_{scale_tag}": t_aini,
                        f"Original_coef_pval_{scale_tag}": p_ana,
                        f"Empirical_coef_pval_{scale_tag}": p_boot,
                        # AR(1) and fit stats
                        f"beta_ret_lag1_{scale_tag}": beta_ar,
                        f"pval_ret_lag1_{scale_tag}": pval_ret_lag1,
                        f"r2_{scale_tag}": r2,
                        f"adj_r2_{scale_tag}": adj_r2,
                    }

                    # Optional controls: store betas & p-values
                    if control_vars and hasattr(res, "params"):
                        for c in control_vars:
                            row[f"beta_{c}_{scale_tag}"] = res.params.get(c, np.nan)
                            row[f"pval_{c}_{scale_tag}"] = res.pvalues.get(c, np.nan)

                    rows.append(row)

    # Combine all rows to a DataFrame
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(
            "No rows produced. Causes could be: all groups below min_obs, "
            "missing AINI variants after merge, or no usable STD series for normalized_AINI."
        )

    out = out.sort_values(["Year", "Ticker", "AINI_variant"]).reset_index(drop=True)

    # Ensure p-value columns exist for BH step even if no STD rows were produced at all
    for col in ["Empirical_coef_pval_raw", "Original_coef_pval_raw",
                "Empirical_coef_pval_std", "Original_coef_pval_std"]:
        if col not in out.columns:
            out[col] = np.nan

    # --------------------------------------------------------
    # FDR correction within (Ticker, Year) for RAW and STD
    # --------------------------------------------------------
    corrected = []
    for (ticker, year), g in out.groupby(["Ticker", "Year"], dropna=False):
        g = g.copy()

        # RAW empirical
        p_corr, rej = bh_with_fallback(g["Empirical_coef_pval_raw"], fdr_alpha)
        g["BH_corr_coef_pval_raw"] = p_corr
        g["BH_reject_coef_raw"] = rej

        # RAW analytic (legacy naming with HC3 suffix kept for compatibility)
        p_corr, rej = bh_with_fallback(g["Original_coef_pval_raw"], fdr_alpha)
        g["BH_corr_coef_pval_HC3_raw"] = p_corr
        g["BH_reject_coef_HC3_raw"] = rej

        # STD empirical (only populated where 'normalized_AINI' STD exists)
        p_corr, rej = bh_with_fallback(g["Empirical_coef_pval_std"], fdr_alpha)
        g["BH_corr_coef_pval_std"] = p_corr
        g["BH_reject_coef_std"] = rej

        # STD analytic
        p_corr, rej = bh_with_fallback(g["Original_coef_pval_std"], fdr_alpha)
        g["BH_corr_coef_pval_HC3_std"] = p_corr
        g["BH_reject_coef_HC3_std"] = rej

        corrected.append(g)

    out = pd.concat(corrected, ignore_index=True).sort_values(["Year", "Ticker", "AINI_variant"])

    # Resolve output path and persist if requested
    outname_final = "ols_sameday_mbboot_fdr_controlled.csv" if control_vars else outname
    if save_csv:
        if outdir is None:
            try:
                base = Path(__file__).resolve().parents[2]
            except NameError:
                base = Path.cwd()
            outdir = base / "data" / "processed" / "variables"
        outdir.mkdir(parents=True, exist_ok=True)
        out.to_csv(outdir / outname_final, index=False)

    return out
