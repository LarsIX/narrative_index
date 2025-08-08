from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.random import default_rng
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed


def _add_const(X: pd.DataFrame) -> pd.DataFrame:
    return sm.add_constant(X, has_constant="add")


def _fit_ols(y: pd.Series, X: pd.DataFrame, cov: str = "HAC", hac_lags: Optional[int] = None):
    if cov.upper() == "HAC":
        return sm.OLS(y, _add_const(X)).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags or 1})
    return sm.OLS(y, _add_const(X)).fit(cov_type="HC3")


def _moving_block_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if block_size < 1 or block_size > n:
        raise ValueError("block_size must be in [1, n]")
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n - block_size + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
    return idx


def _auto_hac_lags(T: int) -> int:
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
    try:
        rng = np.random.default_rng(base_seed + i)
        n = yhat_R.shape[0]
        idx = _moving_block_indices(n, block_size, rng)
        y_b = pd.Series(yhat_R + ehat_R[idx])
        X_b = X.iloc[idx].reset_index(drop=True)
        res_b = _fit_ols(y_b, X_b, cov=cov, hac_lags=hac_lags)
        return float(res_b.tvalues.get(coef_name, np.nan))
    except Exception:
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
    """Parallelized null-imposing residual MBB p-value for the target coefficient."""
    if coef_name not in X.columns:
        return float("nan"), 0

    # Restricted fit (drop coef -> impose H0 in data generation)
    XR = X.drop(columns=[coef_name])
    res_R = sm.OLS(y, _add_const(XR)).fit()
    yhat_R = res_R.fittedvalues.to_numpy()
    ehat_R = (y - res_R.fittedvalues).to_numpy()

    # Observed unrestricted t-stat
    res_U = _fit_ols(y, X, cov=cov, hac_lags=hac_lags)
    if coef_name not in res_U.params.index:
        return float("nan"), 0
    T_obs = float(res_U.tvalues[coef_name])

    # Parallel bootstrap (process-based)
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
    if n_valid < max(50, int(0.2 * n_boot)):
        return float("nan"), n_valid

    p = (np.sum(np.abs(t_vals) >= abs(T_obs)) + 1) / (n_valid + 1)
    return float(p), n_valid


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
) -> pd.DataFrame:

    if aini_variants is None:
        aini_variants = ["normalized_AINI", "EMA_02", "EMA_08", "normalized_AINI_growth"]

    aini = aini_df.copy()
    aini["date"] = pd.to_datetime(aini["date"])
    rows = []
    master_rng = default_rng(seed) # like np.RandomState, but more flexible

    for year, fin in fin_data_by_year.items():
        f = fin.copy()
        f["date"] = pd.to_datetime(f["date"])
        # Calculate log returns by Ticker
        f['log_return'] = f.groupby('Ticker')['Adj Close'].transform(lambda x: np.log(x) - np.log(x.shift(1)))
        f = f.dropna(subset=['log_return'])
        f = f.rename(columns={"Ticker": "ticker"})
        f = f.sort_values(["ticker", "date"])
        f["ret_lag1"] = f.groupby("ticker")["log_return"].shift(1)

        merged = pd.merge(f, aini[["date"] + aini_variants], on="date", how="left")

        for ticker, d in merged.groupby("ticker"):
            d = d.sort_values("date").reset_index(drop=True)

            for var in aini_variants:
                # Build raw and z-scored series (within this ticker-year)
                raw_series = d[var]
                mu = raw_series.mean(skipna=True)
                sd = raw_series.std(skipna=True)
                std_series = (raw_series - mu) / sd if np.isfinite(sd) and sd > 0 else raw_series

                for scale_tag, series in (("raw", raw_series), ("std", std_series)):
                    # Assemble sub-DataFrame with identical column name for predictor (var)
                    parts = [d["log_return"], d["ret_lag1"], series.rename(var)]
                    if control_vars:
                        parts.extend([d[c] for c in control_vars])
                    sub = pd.concat(parts, axis=1).dropna()

                    if len(sub) < min_obs:
                        continue

                    y = sub["log_return"]
                    X = sub.drop(columns="log_return")

                    use_hac = (cov_for_analytic.upper() == "HAC")
                    _hac = _auto_hac_lags(len(sub)) if (use_hac and hac_lags is None) else hac_lags

                    # Analytic fit
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
                        alpha = beta_aini = beta_ar = np.nan
                        se_aini = t_aini = p_ana = np.nan
                        r2 = adj_r2 = np.nan
                        pval_ret_lag1 = np.nan

                    # Bootstrap p-value (null-imposing residual MBB)
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

                    row = {
                        "Ticker": ticker,
                        "AINI_variant": var,
                        "Year": year,
                        "N_obs": len(sub),
                        "block_size": block_size,
                        "N_boot": n_boot,
                        "N_boot_valid_" + scale_tag: n_valid,
                        # Coefs & inference for AINI
                        f"alpha_{scale_tag}": alpha,
                        f"beta_aini_{scale_tag}": beta_aini,
                        f"se_aini_{scale_tag}": se_aini,
                        f"t_aini_{scale_tag}": t_aini,
                        f"Original_coef_pval_{scale_tag}": p_ana,
                        f"Empirical_coef_pval_{scale_tag}": p_boot,
                        # AR(1) and fit stats (per scale, since they can differ slightly)
                        f"beta_ret_lag1_{scale_tag}": beta_ar,
                        f"pval_ret_lag1_{scale_tag}": pval_ret_lag1,
                        f"r2_{scale_tag}": r2,
                        f"adj_r2_{scale_tag}": adj_r2,
                    }

                    # Controls (per scale)
                    if control_vars and hasattr(res, "params"):
                        for c in control_vars:
                            row[f"beta_{c}_{scale_tag}"] = res.params.get(c, np.nan)
                            row[f"pval_{c}_{scale_tag}"] = res.pvalues.get(c, np.nan)

                    rows.append(row)

    out = pd.DataFrame(rows).sort_values(["Year", "Ticker", "AINI_variant"])

    # Apply BH-FDR within (Ticker, Year) separately for RAW and STD sets
    corrected = []
    for (ticker, year), g in out.groupby(["Ticker", "Year"], dropna=False):
        g = g.copy()

        # RAW
        p_emp_raw = g["Empirical_coef_pval_raw"]
        if p_emp_raw.notna().sum() >= 2:
            rej, p_corr = multipletests(p_emp_raw, alpha=fdr_alpha, method="fdr_bh")[:2]
            g["BH_reject_coef_raw"] = rej
            g["BH_corr_coef_pval_raw"] = p_corr
        else:
            g["BH_reject_coef_raw"] = False
            g["BH_corr_coef_pval_raw"] = np.nan

        p_ana_raw = g["Original_coef_pval_raw"]
        if p_ana_raw.notna().sum() >= 2:
            rej2, p_corr2 = multipletests(p_ana_raw, alpha=fdr_alpha, method="fdr_bh")[:2]
            g["BH_reject_coef_HC3_raw"] = rej2
            g["BH_corr_coef_pval_HC3_raw"] = p_corr2
        else:
            g["BH_reject_coef_HC3_raw"] = False
            g["BH_corr_coef_pval_HC3_raw"] = np.nan

        # STD
        p_emp_std = g["Empirical_coef_pval_std"]
        if p_emp_std.notna().sum() >= 2:
            rej, p_corr = multipletests(p_emp_std, alpha=fdr_alpha, method="fdr_bh")[:2]
            g["BH_reject_coef_std"] = rej
            g["BH_corr_coef_pval_std"] = p_corr
        else:
            g["BH_reject_coef_std"] = False
            g["BH_corr_coef_pval_std"] = np.nan

        p_ana_std = g["Original_coef_pval_std"]
        if p_ana_std.notna().sum() >= 2:
            rej2, p_corr2 = multipletests(p_ana_std, alpha=fdr_alpha, method="fdr_bh")[:2]
            g["BH_reject_coef_HC3_std"] = rej2
            g["BH_corr_coef_pval_HC3_std"] = p_corr2
        else:
            g["BH_reject_coef_HC3_std"] = False
            g["BH_corr_coef_pval_HC3_std"] = np.nan

        corrected.append(g)

    out = pd.concat(corrected, ignore_index=True).sort_values(["Year", "Ticker", "AINI_variant"])
    outname = "ols_sameday_mbboot_fdr_controlled.csv" if control_vars else outname 

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
