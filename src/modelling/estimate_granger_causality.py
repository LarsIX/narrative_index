"""
Granger Causality (both directions) with HC3/HAC covariance for the
analytic F-test, and fast Null-imposing Wild Residual Bootstrap p-values
via RSS-difference F-statistics — with parallelization.

- Analytic (observed) F uses HAC/HC3 (robust, as you prefer).
- Optional block-wise wild weights to be conservative about short-run
  serial correlation (set block_size>1).

Public APIs preserved:
  - run_gc_mbboot_fdr(...)
  - run_gc_mbboot_fdr_controls(...)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.random import default_rng
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
import re

# ============================================================
# Utilities
# ============================================================

def _add_const(X: pd.DataFrame) -> pd.DataFrame:
    """Add an intercept column named ``const``."""
    return sm.add_constant(X, has_constant="add")

def _fit_ols(
    y: pd.Series,
    X: pd.DataFrame,
    cov: str = "HAC",
    hac_lags: Optional[int] = None,
):
    """
    Fit OLS with either HC3 or HAC covariance (for ANALYTIC/observed F only).
    """
    if cov.upper() == "HAC":
        return sm.OLS(y, _add_const(X)).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags or 1})
    return sm.OLS(y, _add_const(X)).fit(cov_type="HC3")

def _lag(df: pd.DataFrame, col: str, lags: int, prefix: Optional[str] = None) -> pd.DataFrame:
    """Create lagged columns for ``col`` with names ``{prefix}_lag1..lags``."""
    if lags < 1:
        return df
    base = col if prefix is None else prefix
    out = df.copy()
    for i in range(1, lags + 1):
        out[f"{base}_lag{i}"] = out[col].shift(i)
    return out

def _wald_F_for_zero_coefs(res, target_cols: List[str]) -> Tuple[float, float, int, int]:
    """
    HAC/HC3-robust Wald F-test for joint zero restrictions on target_cols.
    """
    exog_names = res.model.exog_names
    k = len(target_cols)
    R = np.zeros((k, len(exog_names)))
    for i, name in enumerate(target_cols):
        if name not in exog_names:
            raise ValueError(f"Column '{name}' not in model exog.")
        R[i, exog_names.index(name)] = 1.0
    w = res.wald_test(R, use_f=True, scalar=True)
    F_stat = float(w.fvalue)
    p_value = float(w.pvalue)
    df_num = int(w.df_num)
    df_den = int(w.df_denom)
    return F_stat, p_value, df_num, df_den

def _auto_hac_lags(T: int) -> int:
    """Newey–West rule-of-thumb for HAC maxlags: floor(4*(T/100)^(2/9))."""
    return int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))

# ============================================================
# bootstrap machinery 
# ============================================================

def _ols_rss(y: np.ndarray, X: np.ndarray) -> tuple[float, int]:
    """
    Plain OLS via lstsq to get RSS and number of parameters (incl. intercept).
    y: (n,), X: (n, k) without intercept column.
    """
    X1 = np.c_[np.ones((X.shape[0], 1)), X]
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    rss = float(resid.T @ resid)
    k = X1.shape[1]  # parameters incl. intercept
    return rss, k

def _f_stat_from_rss(rss_R: float, rss_U: float, k_R: int, k_U: int, n: int) -> float:
    """
    F = ((RSS_R - RSS_U)/q) / (RSS_U/(n - k_U)), where q = k_U - k_R.
    """
    q = k_U - k_R
    if q <= 0 or n <= k_U or rss_U <= 0:
        return float("nan")
    num = (rss_R - rss_U) / q
    den = rss_U / (n - k_U)
    return float(num / den) if den > 0 else float("nan")

def _wild_weights_iid(n: int, rng: np.random.Generator, dist: str = "rademacher") -> np.ndarray:
    d = dist.lower()
    if d == "rademacher":
        w = rng.integers(0, 2, size=n) * 2 - 1   # {+1,-1}
        return w.astype(float)
    elif d == "normal":
        return rng.standard_normal(n)
    else:
        raise ValueError(f"Unknown wild weight distribution: {dist}")

def _wild_weights_block(n: int, block_size: int, rng: np.random.Generator, dist: str = "rademacher") -> np.ndarray:
    """
    Optional block-wise wild weights: within each block the same weight, to be conservative w.r.t. short-run serial correlation.
    """
    if block_size <= 1:
        return _wild_weights_iid(n, rng, dist)
    n_blocks = int(np.ceil(n / block_size))
    if dist.lower() == "rademacher":
        block_w = rng.integers(0, 2, size=n_blocks) * 2 - 1
    elif dist.lower() == "normal":
        block_w = rng.standard_normal(n_blocks)
    else:
        raise ValueError(f"Unknown wild weight distribution: {dist}")
    w = np.repeat(block_w, block_size)[:n].astype(float)
    return w

def _gc_wild_bootstrap_null_fast(
    y: pd.Series,
    X_restricted: pd.DataFrame,
    X_unrestricted: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 42,
    weight_dist: str = "rademacher",
    block_size: int = 1,  # set >1 to be conservative about short-run autocorr
) -> tuple[float, int, float]:
    """
    Null-imposing **wild residual bootstrap** p-value for nested OLS GC test
    using RSS-difference F-statistic.
    Returns: (p_boot, n_valid, F_obs).
    """
    # -> numpy arrays, aligned
    yv = y.to_numpy(copy=False).astype(float)
    XR = X_restricted.to_numpy(copy=False).astype(float)
    XU = X_unrestricted.to_numpy(copy=False).astype(float)
    n = yv.shape[0]

    # Observed F via RSS difference (no robust cov here)
    rss_R, k_R = _ols_rss(yv, XR)
    rss_U, k_U = _ols_rss(yv, XU)
    F_obs = _f_stat_from_rss(rss_R, rss_U, k_R, k_U, n)

    # Restricted fit for H0 (OLS; bootstrap carries heteroskedasticity)
    X1_R = np.c_[np.ones((XR.shape[0], 1)), XR]
    beta_R, *_ = np.linalg.lstsq(X1_R, yv, rcond=None)
    yhat_R = X1_R @ beta_R
    e_R = yv - yhat_R

    rng = default_rng(seed)
    F_boot = []

    for _ in range(n_boot):
        w = _wild_weights_block(n, block_size, rng, dist=weight_dist)
        y_b = yhat_R + e_R * w
        rss_R_b, k_R_b = _ols_rss(y_b, XR)
        rss_U_b, k_U_b = _ols_rss(y_b, XU)
        F_b = _f_stat_from_rss(rss_R_b, rss_U_b, k_R_b, k_U_b, n)
        if np.isfinite(F_b):
            F_boot.append(float(F_b))

    F_boot = np.asarray(F_boot, dtype=float)
    F_boot = F_boot[np.isfinite(F_boot)]
    n_valid = int(F_boot.size)
    if n_valid < max(50, int(0.2 * n_boot)):
        return float("nan"), n_valid, F_obs

    # right-tailed p-value
    p_boot = (np.sum(F_boot >= F_obs) + 1.0) / (n_valid + 1.0)
    return float(p_boot), n_valid, F_obs

# ============================================================
# Column name canonicalization 
# ============================================================

def canonicalize_betas(df: pd.DataFrame, duplicate_beta0_ar: bool = False) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    rename: Dict[str, str] = {}
    for c in df.columns:
        if c in ("A2R_beta_const", "R2A_beta_const"):
            rename[c] = "β₀"
    role_maps = [
        dict(cross="x", ar="ret", prefix="A2R"),
        dict(cross="ret", ar="x", prefix="R2A"),
    ]
    cols = list(df.columns)
    for rm in role_maps:
        cross, ar, pref = rm["cross"], rm["ar"], rm["prefix"]
        pat_cross = re.compile(rf"^{pref}_beta_{cross}(?:_lag)?(\d+)$")
        pat_ar = re.compile(rf"^{pref}_beta_{ar}(?:_lag)?(\d+)$")
        pat_ctrl = re.compile(rf"^{pref}_beta_ctrl_(.+?)(?:_lag)?(\d+)$")
        for c in cols:
            if not isinstance(c, str) or c in rename:
                continue
            m = pat_cross.match(c)
            if m:
                rename[c] = f"β_x{m.group(1)}"; continue
            m = pat_ar.match(c)
            if m:
                rename[c] = f"β_x{m.group(1)}_ar"; continue
            m = pat_ctrl.match(c)
            if m:
                ctrl_name_raw, lag = m.group(1), m.group(2)
                safe = re.sub(r"[^A-Za-z0-9]+", "_", ctrl_name_raw).strip("_")
                rename[c] = f"β_ctrl_{safe}{lag}"; continue
    df = df.rename(columns=rename)
    if duplicate_beta0_ar and "β₀" in df.columns:
        df["β₀_ar"] = df["β₀"]
    return df

# ============================================================
# Main per-(ticker, variant) workers
# ============================================================

def _process_ticker_variant(
    d: pd.DataFrame,
    ticker: str,
    year_key,
    var: str,
    p_ret: int,
    p_x: int,
    min_obs: int,
    n_boot: int,
    base_seed: int,
    cov_for_analytic: str,
    hac_lags: Optional[int],
    weight_dist: str,
    block_size: int = 1,
) -> List[Dict]:
    out_rows: List[Dict] = []

    # Build return and AINI lags
    df_lagged = _lag(d, "log_return", p_ret, prefix="ret")
    df_v = _lag(df_lagged, var, p_x, prefix=var)

    ret_lag_cols = [f"ret_lag{i}" for i in range(1, p_ret + 1)]
    xlag_cols = [f"{var}_lag{i}" for i in range(1, p_x + 1)]

    # -------- AINI → Return --------
    y_r = df_v["log_return"]
    XR = df_v[ret_lag_cols]
    XU = df_v[ret_lag_cols + xlag_cols]
    sub_r = pd.concat([y_r, XU], axis=1).dropna()

    if len(sub_r) >= min_obs and (1 + len(ret_lag_cols) + len(xlag_cols)) < len(sub_r):
        y_sub = sub_r["log_return"]
        XR_sub = sub_r[ret_lag_cols]
        XU_sub = sub_r[ret_lag_cols + xlag_cols]

        use_hac = (cov_for_analytic.upper() == "HAC")
        _hac_lags = (hac_lags if use_hac and hac_lags is not None
                     else (_auto_hac_lags(len(sub_r)) if use_hac else None))

        # ANALYTIC observed F with HAC/HC3 (robust)
        try:
            res_u = _fit_ols(y_sub, XU_sub,
                             cov=("HAC" if use_hac else "HC3"),
                             hac_lags=_hac_lags)
            F_stat, p_analytic, df_num, df_den = _wald_F_for_zero_coefs(res_u, xlag_cols)
            coef_dict = {}
            for k, v in res_u.params.to_dict().items():
                if k.startswith(var + "_lag"):
                    lagnum = k.split("lag")[-1]; coef_dict[f"A2R_beta_x_lag{lagnum}"] = v
                elif k.startswith("ret_lag"):
                    lagnum = k.split("lag")[-1]; coef_dict[f"A2R_beta_ret_lag{lagnum}"] = v
                elif k == "const":
                    coef_dict["A2R_beta_const"] = v
                else:
                    coef_dict[f"A2R_beta_{k}"] = v
        except Exception:
            res_u = None
            F_stat = p_analytic = np.nan
            df_num = len(xlag_cols)
            df_den = max(1, len(sub_r) - (1 + len(ret_lag_cols) + len(xlag_cols)))
            coef_dict = {}

        #  bootstrap p-value 
        seed = int((abs(hash((str(year_key), str(ticker), var, "A2R"))) % (2**31 - 1)) + base_seed)
        try:
            p_boot, n_valid, F_obs = _gc_wild_bootstrap_null_fast(
                y=y_sub, X_restricted=XR_sub, X_unrestricted=XU_sub,
                n_boot=n_boot, seed=seed, weight_dist=weight_dist, block_size=block_size,
            )
        except Exception:
            p_boot, n_valid, F_obs = np.nan, 0, np.nan

        out_rows.append({
            "Ticker": ticker,
            "AINI_variant": var,
            "Year": year_key,
            "Direction": "AINI_to_RET",
            **coef_dict,
            "p_x": p_x,
            "N_obs": len(sub_r),
            "N_boot": n_boot,
            "N_boot_valid": n_valid,
            "F_stat": F_stat,                 # analytic robust F
            "F_stat_obs_RSS": F_obs,          # observed F via RSS 
            "Original_F_pval": p_analytic,    # analytic p
            "Empirical_F_pval": p_boot,       # bootstrap p
            "r2_u": getattr(res_u, "rsquared", np.nan),
            "adj_r2_u": getattr(res_u, "rsquared_adj", np.nan),
        })

    # -------- Return → AINI --------
    y_x = df_v[var]
    XR2 = df_v[xlag_cols] if xlag_cols else pd.DataFrame(index=df_v.index)
    XU2 = pd.concat([XR2, df_v[ret_lag_cols]], axis=1) if ret_lag_cols else XR2
    sub_x = pd.concat([y_x, XU2], axis=1).dropna()

    if len(sub_x) >= min_obs and (1 + len(xlag_cols) + len(ret_lag_cols)) < len(sub_x):
        y_sub = sub_x[var]
        XR_sub = sub_x[xlag_cols] if xlag_cols else pd.DataFrame(index=sub_x.index)
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
            coef_dict = {}
            for k, v in res_u.params.to_dict().items():
                if k.startswith(var + "_lag"):
                    lagnum = k.split("lag")[-1]; coef_dict[f"R2A_beta_x_lag{lagnum}"] = v
                elif k.startswith("ret_lag"):
                    lagnum = k.split("lag")[-1]; coef_dict[f"R2A_beta_ret_lag{lagnum}"] = v
                elif k == "const":
                    coef_dict["R2A_beta_const"] = v
                else:
                    coef_dict[f"R2A_beta_{k}"] = v
        except Exception:
            res_u = None
            F_stat = p_analytic = np.nan
            df_num = len(ret_lag_cols)
            df_den = max(1, len(sub_x) - (1 + (len(xlag_cols) + len(ret_lag_cols))))
            coef_dict = {}

        seed = int((abs(hash((str(year_key), str(ticker), var, "R2A"))) % (2**31 - 1)) + base_seed)
        try:
            p_boot, n_valid, F_obs = _gc_wild_bootstrap_null_fast(
                y=y_sub, X_restricted=XR_sub, X_unrestricted=XU_sub,
                n_boot=n_boot, seed=seed, weight_dist=weight_dist, block_size=block_size,
            )
        except Exception:
            p_boot, n_valid, F_obs = np.nan, 0, np.nan

        out_rows.append({
            "Ticker": ticker,
            "AINI_variant": var,
            "Year": year_key,
            "Direction": "RET_to_AINI",
            **coef_dict,
            "p_x": p_x,
            "N_obs": len(sub_x),
            "N_boot": n_boot,
            "N_boot_valid": n_valid,
            "F_stat": F_stat,
            "F_stat_obs_RSS": F_obs,
            "Original_F_pval": p_analytic,
            "Empirical_F_pval": p_boot,
            "r2_u": getattr(res_u, "rsquared", np.nan),
            "adj_r2_u": getattr(res_u, "rsquared_adj", np.nan),
        })

    return out_rows

def _merge_controls_base(frame: pd.DataFrame, controls_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if controls_df is None:
        return frame
    c = controls_df.copy()
    if "date" not in c.columns:
        raise ValueError("controls_df must contain a 'date' column.")
    c["date"] = pd.to_datetime(c["date"], errors="coerce")
    c.columns = [("ticker" if col.lower() == "ticker" else col) for col in c.columns]
    f = frame.copy()
    f.columns = [("ticker" if col == "Ticker" else col) for col in f.columns]
    if "ticker" in c.columns:
        return pd.merge(f, c, on=["date", "ticker"], how="left")
    return pd.merge(f, c, on=["date"], how="left")

def _build_control_lags_inline(d: pd.DataFrame, controls_lags: Optional[Dict[str, int]]) -> Tuple[pd.DataFrame, List[str]]:
    if not controls_lags:
        return d, []
    out = d.copy()
    ctrl_lag_cols: List[str] = []
    for ctrl_name, p_c in controls_lags.items():
        if p_c is None or p_c <= 0 or ctrl_name not in out.columns:
            continue
        out = _lag(out, ctrl_name, p_c, prefix=ctrl_name)
        for i in range(1, p_c + 1):
            col = f"{ctrl_name}_lag{i}"
            if col in out.columns:
                ctrl_lag_cols.append(col)
    return out, ctrl_lag_cols

def _process_ticker_variant_with_controls(
    d: pd.DataFrame,
    ticker: str,
    year_key,
    var: str,
    p_ret: int,
    p_x: int,
    controls_lags: Optional[Dict[str, int]],
    min_obs: int,
    n_boot: int,
    base_seed: int,
    cov_for_analytic: str,
    hac_lags: Optional[int],
    weight_dist: str,
    block_size: int = 1,
) -> List[Dict]:
    out_rows: List[Dict] = []

    df_lagged = _lag(d, "log_return", p_ret, prefix="ret")
    df_v = _lag(df_lagged, var, p_x, prefix=var)
    df_v, ctrl_lag_cols = _build_control_lags_inline(df_v, controls_lags)

    ret_lag_cols = [f"ret_lag{i}" for i in range(1, p_ret + 1)]
    xlag_cols = [f"{var}_lag{i}" for i in range(1, p_x + 1)]

    # AINI → Return
    y_r = df_v["log_return"]
    XR = df_v[ret_lag_cols + ctrl_lag_cols] if ctrl_lag_cols else df_v[ret_lag_cols]
    XU = df_v[ret_lag_cols + ctrl_lag_cols + xlag_cols] if ctrl_lag_cols else df_v[ret_lag_cols + xlag_cols]
    sub_r = pd.concat([y_r, XU], axis=1).dropna()

    if len(sub_r) >= min_obs and (1 + XU.shape[1]) < len(sub_r) + 1:
        y_sub = sub_r["log_return"]; XR_sub = sub_r[XR.columns]; XU_sub = sub_r[XU.columns]
        use_hac = (cov_for_analytic.upper() == "HAC")
        _hac = (hac_lags if use_hac and hac_lags is not None else (_auto_hac_lags(len(sub_r)) if use_hac else None))
        try:
            res_u = _fit_ols(y_sub, XU_sub, cov=("HAC" if use_hac else "HC3"), hac_lags=_hac)
            F_stat, p_analytic, df_num, df_den = _wald_F_for_zero_coefs(res_u, xlag_cols)
            coef_dict = {}
            for k, v in res_u.params.to_dict().items():
                if k in xlag_cols: coef_dict[f"A2R_beta_x_{k.split('lag')[-1]}"] = v
                elif k in ret_lag_cols: coef_dict[f"A2R_beta_ret_{k.split('lag')[-1]}"] = v
                elif k in ctrl_lag_cols: coef_dict[f"A2R_beta_ctrl_{k}"] = v
                elif k == "const": coef_dict["A2R_beta_const"] = v
        except Exception:
            res_u = None; F_stat = p_analytic = np.nan
            df_num = len(xlag_cols); df_den = max(1, len(sub_r) - (1 + XU.shape[1])); coef_dict = {}
        seed = int((abs(hash((str(year_key), str(ticker), var, "A2R"))) % (2**31 - 1)) + base_seed)
        try:
            p_boot, n_valid, F_obs = _gc_wild_bootstrap_null_fast(
                y=y_sub, X_restricted=XR_sub, X_unrestricted=XU_sub,
                n_boot=n_boot, seed=seed, weight_dist=weight_dist, block_size=block_size,
            )
        except Exception:
            p_boot, n_valid, F_obs = np.nan, 0, np.nan

        out_rows.append({
            "Ticker": ticker, "AINI_variant": var, "Year": year_key, "Direction": "AINI_to_RET",
            **coef_dict, "p_x": p_x, "N_obs": len(sub_r),
            "N_boot": n_boot, "N_boot_valid": n_valid,
            "F_stat": F_stat, "F_stat_obs_RSS": F_obs,
            "Original_F_pval": p_analytic, "Empirical_F_pval": p_boot,
            "r2_u": getattr(res_u, "rsquared", np.nan), "adj_r2_u": getattr(res_u, "rsquared_adj", np.nan),
        })

    # Return → AINI
    y_x = df_v[var]
    XR2_base = df_v[xlag_cols] if xlag_cols else pd.DataFrame(index=df_v.index)
    XR2 = pd.concat([XR2_base, df_v[ctrl_lag_cols]], axis=1) if ctrl_lag_cols else XR2_base
    XU2 = pd.concat([XR2, df_v[ret_lag_cols]], axis=1) if ret_lag_cols else XR2
    sub_x = pd.concat([y_x, XU2], axis=1).dropna()

    if len(sub_x) >= min_obs and (1 + XU2.shape[1]) < len(sub_x) + 1:
        y_sub = sub_x[var]; XR_sub = sub_x[XR2.columns]; XU_sub = sub_x[XU2.columns]
        if XR_sub.shape[1] == 0:
            XR_sub = pd.DataFrame({"_zero": np.zeros(len(sub_x), dtype=float)}, index=sub_x.index)
        use_hac = (cov_for_analytic.upper() == "HAC")
        _hac = (hac_lags if use_hac and hac_lags is not None else (_auto_hac_lags(len(sub_x)) if use_hac else None))
        try:
            res_u = _fit_ols(y_sub, XU_sub, cov=("HAC" if use_hac else "HC3"), hac_lags=_hac)
            F_stat, p_analytic, df_num, df_den = _wald_F_for_zero_coefs(res_u, [c for c in ret_lag_cols])
            coef_dict = {}
            for k, v in res_u.params.to_dict().items():
                if k in xlag_cols: coef_dict[f"R2A_beta_x_{k.split('lag')[-1]}"] = v
                elif k in ret_lag_cols: coef_dict[f"R2A_beta_ret_{k.split('lag')[-1]}"] = v
                elif k in ctrl_lag_cols: coef_dict[f"R2A_beta_ctrl_{k}"] = v
                elif k == "const": coef_dict["R2A_beta_const"] = v
        except Exception:
            res_u = None; F_stat = p_analytic = np.nan
            df_num = len(ret_lag_cols); df_den = max(1, len(sub_x) - (1 + XU2.shape[1])); coef_dict = {}
        seed = int((abs(hash((str(year_key), str(ticker), var, "R2A"))) % (2**31 - 1)) + base_seed)
        try:
            p_boot, n_valid, F_obs = _gc_wild_bootstrap_null_fast(
                y=y_sub, X_restricted=XR_sub, X_unrestricted=XU_sub,
                n_boot=n_boot, seed=seed, weight_dist=weight_dist, block_size=block_size,
            )
        except Exception:
            p_boot, n_valid, F_obs = np.nan, 0, np.nan

        out_rows.append({
            "Ticker": ticker, "AINI_variant": var, "Year": year_key, "Direction": "RET_to_AINI",
            **coef_dict, "p_x": p_x, "N_obs": len(sub_x),
            "N_boot": n_boot, "N_boot_valid": n_valid,
            "F_stat": F_stat, "F_stat_obs_RSS": F_obs,
            "Original_F_pval": p_analytic, "Empirical_F_pval": p_boot,
            "r2_u": getattr(res_u, "rsquared", np.nan), "adj_r2_u": getattr(res_u, "rsquared_adj", np.nan),
        })
    return out_rows

# ============================================================
# Public APIs
# ============================================================

def run_gc_mbboot_fdr(
    aini_df: pd.DataFrame,
    fin_data: pd.DataFrame,
    version: str,
    aini_variants: Optional[List[str]] = None,
    p_ret: int = 1,
    p_x: int = 3,
    min_obs: int = 60,
    n_boot: int = 1000,
    seed: int = 42,
    fdr_alpha: float = 0.05,
    cov_for_analytic: str = "HAC",     # "HAC" or "HC3" for observed F
    hac_lags: Optional[int] = None,    # If None and HAC, auto by T
    weight_dist: str = "rademacher",   # 'rademacher' | 'normal'
    save_csv: bool = True,
    outdir: Optional[Path] = None,
    outname: Optional[str] = None,     # If None -> f"granger_causality_{version}_pret{p_ret}_px{p_x}_.csv"
    n_jobs: int = -1,                  # parallel workers (joblib)
    block_size: int = 1,               # >1 for block-wise wild bootstrap
) -> pd.DataFrame:
    if aini_variants is None:
        aini_variants = ["normalized_AINI", "EMA_02", "EMA_08"]

    # --- Financial data prep ---
    fin = fin_data.copy()
    fin["Date"] = pd.to_datetime(fin["Date"])
    fin = fin.sort_values(["Ticker", "Date"])
    fin["log_return"] = fin.groupby("Ticker")["Adj Close"].transform(lambda x: np.log(x) - np.log(x.shift(1)))
    fin = fin.dropna(subset=["log_return"])
    fin["date"] = pd.to_datetime(fin["Date"])

    threshold_24 = pd.Timestamp("2024-01-01")
    threshold_25 = pd.Timestamp("2025-01-01")
    fin_data_by_year = {
        2023: fin[fin["date"] < threshold_24],
        2024: fin[(fin["date"] >= threshold_24) & (fin["date"] < threshold_25)],
        2025: fin[fin["date"] >= threshold_25],
        "2023_24": fin[fin["date"] < threshold_25],
        "2024_25": fin[fin["date"] >= threshold_24],
        "2023_24_25": fin,
    }

    aini = aini_df.copy()
    aini["date"] = pd.to_datetime(aini["date"])

    all_rows: List[Dict] = []
    base_seed = int(seed)

    for year_key, fin_y in fin_data_by_year.items():
        f = fin_y.copy().rename(columns={"Ticker": "ticker"}).sort_values(["ticker", "date"])

        merged = pd.merge(f, aini[["date"] + aini_variants], on="date", how="left")

        tasks: List[Tuple[pd.DataFrame, str, object, str]] = []
        for ticker, d in merged.groupby("ticker"):
            d = d.sort_values("date").reset_index(drop=True)
            for var in aini_variants:
                if var in d.columns:
                    tasks.append((d, str(ticker), year_key, var))

        results_nested: List[List[Dict]] = Parallel(n_jobs=n_jobs, backend="loky", batch_size="auto")(
            delayed(_process_ticker_variant)(
                d=td, ticker=tkr, year_key=yrk, var=vr,
                p_ret=p_ret, p_x=p_x, min_obs=min_obs,
                n_boot=n_boot, base_seed=base_seed,
                cov_for_analytic=cov_for_analytic, hac_lags=hac_lags,
                weight_dist=weight_dist, block_size=block_size,
            )
            for (td, tkr, yrk, vr) in tasks
        )

        for rows in results_nested:
            all_rows.extend(rows)

    out = pd.DataFrame(all_rows).sort_values(["Year", "Ticker", "AINI_variant", "Direction"])

    # ----- FDR per (Ticker, Year, Direction) -----
    if not out.empty:
        corrected = []
        for (ticker, year, direction), g in out.groupby(["Ticker", "Year", "Direction"], dropna=False):
            g = g.copy()
            p_emp = pd.to_numeric(g["Empirical_F_pval"], errors="coerce")
            if p_emp.notna().sum() >= 2:
                rej, p_corr = multipletests(p_emp, alpha=fdr_alpha, method="fdr_bh")[:2]
                g["BH_reject_F"] = rej; g["BH_corr_F_pval"] = p_corr
            else:
                g["BH_reject_F"] = False; g["BH_corr_F_pval"] = np.nan

            p_ana = pd.to_numeric(g["Original_F_pval"], errors="coerce")
            if p_ana.notna().sum() >= 2:
                rej2, p_corr2 = multipletests(p_ana, alpha=fdr_alpha, method="fdr_bh")[:2]
                g["BH_reject_F_HC3"] = rej2; g["BH_corr_F_pval_HC3"] = p_corr2
            else:
                g["BH_reject_F_HC3"] = False; g["BH_corr_F_pval_HC3"] = np.nan

            corrected.append(g)
        out = pd.concat(corrected, ignore_index=True).sort_values(["Year", "Ticker", "AINI_variant", "Direction"])

    # ----- Save  -----
    if outname is None:
        suffix = f"pret{p_ret}_px{p_x}_"
        outname = f"granger_causality_{version}_{suffix}.csv"
    if not out.empty and save_csv:
        if outdir is None:
            try:
                base = Path(__file__).resolve().parents[2]
            except NameError:
                base = Path.cwd()
            outdir = base / "data" / "processed" / "variables"
        outdir.mkdir(parents=True, exist_ok=True)
        out.to_csv(outdir / outname, index=False)

    return out

def run_gc_mbboot_fdr_controls(
    aini_df: pd.DataFrame,
    fin_data: pd.DataFrame,
    version: str,
    control_var: str,
    controls_df: Optional[pd.DataFrame] = None,
    controls_lags: Optional[Dict[str, int]] = None,
    aini_variants: Optional[List[str]] = None,
    p_ret: int = 1,
    p_x: int = 3,
    min_obs: int = 60,
    n_boot: int = 1000,
    seed: int = 42,
    fdr_alpha: float = 0.05,
    cov_for_analytic: str = "HAC",
    hac_lags: Optional[int] = None,
    weight_dist: str = "rademacher",
    save_csv: bool = True,
    outdir: Optional[Path] = None,
    outname: Optional[str] = None,
    n_jobs: int = -1,
    block_size: int = 1,
) -> pd.DataFrame:
    if aini_variants is None:
        aini_variants = ["normalized_AINI", "EMA_02", "EMA_08"]

    fin = fin_data.copy()
    fin["Date"] = pd.to_datetime(fin["Date"])
    fin = fin.sort_values(["Ticker", "Date"])
    fin["log_return"] = fin.groupby("Ticker")["Adj Close"].transform(lambda x: np.log(x) - np.log(x.shift(1)))
    fin = fin.dropna(subset=["log_return"])
    fin["date"] = pd.to_datetime(fin["Date"])

    threshold_24 = pd.Timestamp("2024-01-01")
    threshold_25 = pd.Timestamp("2025-01-01")
    fin_by_year = {
        2023: fin[fin["date"] < threshold_24],
        2024: fin[(fin["date"] >= threshold_24) & (fin["date"] < threshold_25)],
        2025: fin[fin["date"] >= threshold_25],
        "2023_24": fin[fin["date"] < threshold_25],
        "2024_25": fin[fin["date"] >= threshold_24],
        "2023_24_25": fin,
    }

    aini = aini_df.copy()
    aini["date"] = pd.to_datetime(aini["date"])

    all_rows: List[Dict] = []
    base_seed = int(seed)

    for year_key, fin_y in fin_by_year.items():
        f = fin_y.copy().rename(columns={"Ticker": "ticker"}).sort_values(["ticker", "date"])
        merged = pd.merge(f, aini[["date"] + aini_variants], on="date", how="left")
        if controls_df is not None:
            merged = _merge_controls_base(merged, controls_df)

        tasks: List[Tuple[pd.DataFrame, str, object, str]] = []
        for ticker, d in merged.groupby("ticker"):
            d = d.sort_values("date").reset_index(drop=True)
            for var in aini_variants:
                if var in d.columns:
                    tasks.append((d, str(ticker), year_key, var))

        results_nested: List[List[Dict]] = Parallel(n_jobs=n_jobs, backend="loky", batch_size="auto")(
            delayed(_process_ticker_variant_with_controls)(
                d=td, ticker=tkr, year_key=yrk, var=vr,
                p_ret=p_ret, p_x=p_x,
                controls_lags=controls_lags,
                min_obs=min_obs, n_boot=n_boot, base_seed=base_seed,
                cov_for_analytic=cov_for_analytic, hac_lags=hac_lags,
                weight_dist=weight_dist, block_size=block_size,
            )
            for (td, tkr, yrk, vr) in tasks
        )
        for rows in results_nested:
            all_rows.extend(rows)

    out = pd.DataFrame(all_rows).sort_values(["Year", "Ticker", "AINI_variant", "Direction"])

    if not out.empty:
        corrected = []
        for (ticker, year, direction), g in out.groupby(["Ticker", "Year", "Direction"], dropna=False):
            g = g.copy()
            p_emp = pd.to_numeric(g["Empirical_F_pval"], errors="coerce")
            if p_emp.notna().sum() >= 2:
                rej, p_corr = multipletests(p_emp, alpha=fdr_alpha, method="fdr_bh")[:2]
                g["BH_reject_F"] = rej; g["BH_corr_F_pval"] = p_corr
            else:
                g["BH_reject_F"] = False; g["BH_corr_F_pval"] = np.nan
            p_ana = pd.to_numeric(g["Original_F_pval"], errors="coerce")
            if p_ana.notna().sum() >= 2:
                rej2, p_corr2 = multipletests(p_ana, alpha=fdr_alpha, method="fdr_bh")[:2]
                g["BH_reject_F_HC3"] = rej2; g["BH_corr_F_pval_HC3"] = p_corr2
            else:
                g["BH_reject_F_HC3"] = False; g["BH_corr_F_pval_HC3"] = np.nan
            corrected.append(g)
        out = pd.concat(corrected, ignore_index=True).sort_values(["Year", "Ticker", "AINI_variant", "Direction"])

    if outname is None:
        ctrl_tag = control_var
        if controls_lags:
            ctrl_tag = f"{control_var}_lags" + "".join(f"{k}{v}" for k, v in sorted(controls_lags.items()))
        suffix = f"pret{p_ret}_px{p_x}_"
        outname = f"granger_causality_{version}_{ctrl_tag}_{suffix}.csv"

    if not out.empty and save_csv:
        if outdir is None:
            try:
                base = Path(__file__).resolve().parents[2]
            except NameError:
                base = Path.cwd()
            outdir = base / "data" / "processed" / "variables"
        outdir.mkdir(parents=True, exist_ok=True)
        out.to_csv(outdir / outname, index=False)
    return out

# ============================================================
# VIX endogeneity / collider precondition tests
#   - RET -> VIX  
#   - AINI -> VIX 
# ============================================================

def _process_ticker_vix_tests(
    d: pd.DataFrame,
    ticker: str,
    year_key,
    aini_var: str,
    vix_col: str,
    p_ret: int,
    p_x: int,
    p_vix: int,
    min_obs: int,
    n_boot: int,
    base_seed: int,
    cov_for_analytic: str,
    hac_lags: Optional[int],
    weight_dist: str,
    block_size: int = 1,
) -> List[Dict]:
    """
    Build nested models with VIX as y:
      y = VIX_t;  R: VIX lags only;  U: VIX lags + block (RET lags) or (AINI lags).
    Returns two rows (RET_to_VIX, AINI_to_VIX) if feasible.
    """
    out_rows: List[Dict] = []

    if vix_col not in d.columns:
        return out_rows

    # Lags
    df1 = _lag(d, vix_col, p_vix, prefix="vix")
    df2 = _lag(df1, "log_return", p_ret, prefix="ret")
    df3 = _lag(df2, aini_var, p_x, prefix=aini_var)

    vix_lag_cols = [f"vix_lag{i}" for i in range(1, p_vix + 1)]
    ret_lag_cols = [f"ret_lag{i}" for i in range(1, p_ret + 1)]
    aini_lag_cols = [f"{aini_var}_lag{i}" for i in range(1, p_x + 1)]

    # ---------- RET -> VIX ----------
    y = df3[vix_col]
    XR = df3[vix_lag_cols]  # restricted: VIX AR only
    XU = df3[vix_lag_cols + ret_lag_cols]  # unrestricted: + return lags
    sub = pd.concat([y, XU], axis=1).dropna()

    if len(sub) >= min_obs and (1 + len(vix_lag_cols) + len(ret_lag_cols)) < len(sub):
        y_sub = sub[vix_col]
        XR_sub = sub[vix_lag_cols]
        XU_sub = sub[vix_lag_cols + ret_lag_cols]

        use_hac = (cov_for_analytic.upper() == "HAC")
        _hac = (hac_lags if use_hac and hac_lags is not None else (_auto_hac_lags(len(sub)) if use_hac else None))
        try:
            res_u = _fit_ols(y_sub, XU_sub, cov=("HAC" if use_hac else "HC3"), hac_lags=_hac)
            F_stat, p_analytic, df_num, df_den = _wald_F_for_zero_coefs(res_u, ret_lag_cols)
            coef_dict = {}
            for k, v in res_u.params.to_dict().items():
                if k in vix_lag_cols: coef_dict[f"VIX_ar_{k.split('lag')[-1]}"] = v
                elif k in ret_lag_cols: coef_dict[f"RET_{k.split('lag')[-1]}"] = v
                elif k == "const": coef_dict["β₀"] = v
        except Exception:
            res_u = None; F_stat = p_analytic = np.nan
            df_num = len(ret_lag_cols); df_den = max(1, len(sub) - (1 + len(vix_lag_cols) + len(ret_lag_cols)))
            coef_dict = {}

        seed = int((abs(hash((str(year_key), str(ticker), aini_var, "RET2VIX"))) % (2**31 - 1)) + base_seed)
        try:
            p_boot, n_valid, F_obs = _gc_wild_bootstrap_null_fast(
                y=y_sub, X_restricted=XR_sub, X_unrestricted=XU_sub,
                n_boot=n_boot, seed=seed, weight_dist=weight_dist, block_size=block_size,
            )
        except Exception:
            p_boot, n_valid, F_obs = np.nan, 0, np.nan

        out_rows.append({
            "Ticker": ticker, "AINI_variant": aini_var, "Year": year_key,
            "Direction": "RET_to_VIX",
            **coef_dict,
            "p_vix": p_vix, "p_ret": p_ret, "p_x": p_x,
            "N_obs": len(sub), "N_boot": n_boot, "N_boot_valid": n_valid,
            "F_stat": F_stat, "F_stat_obs_RSS": F_obs,
            "Original_F_pval": p_analytic, "Empirical_F_pval": p_boot,
            "r2_u": getattr(res_u, "rsquared", np.nan), "adj_r2_u": getattr(res_u, "rsquared_adj", np.nan),
        })

    # ---------- AINI -> VIX ----------
    y = df3[vix_col]
    XR = df3[vix_lag_cols]  # restricted: VIX AR only
    XU = df3[vix_lag_cols + aini_lag_cols]  # unrestricted: + AINI lags
    sub = pd.concat([y, XU], axis=1).dropna()

    if len(sub) >= min_obs and (1 + len(vix_lag_cols) + len(aini_lag_cols)) < len(sub):
        y_sub = sub[vix_col]
        XR_sub = sub[vix_lag_cols]
        XU_sub = sub[vix_lag_cols + aini_lag_cols]

        use_hac = (cov_for_analytic.upper() == "HAC")
        _hac = (hac_lags if use_hac and hac_lags is not None else (_auto_hac_lags(len(sub)) if use_hac else None))
        try:
            res_u = _fit_ols(y_sub, XU_sub, cov=("HAC" if use_hac else "HC3"), hac_lags=_hac)
            F_stat, p_analytic, df_num, df_den = _wald_F_for_zero_coefs(res_u, aini_lag_cols)
            coef_dict = {}
            for k, v in res_u.params.to_dict().items():
                if k in vix_lag_cols: coef_dict[f"VIX_ar_{k.split('lag')[-1]}"] = v
                elif k in aini_lag_cols: coef_dict[f"AINI_{k.split('lag')[-1]}"] = v
                elif k == "const": coef_dict["β₀"] = v
        except Exception:
            res_u = None; F_stat = p_analytic = np.nan
            df_num = len(aini_lag_cols); df_den = max(1, len(sub) - (1 + len(vix_lag_cols) + len(aini_lag_cols)))
            coef_dict = {}

        seed = int((abs(hash((str(year_key), str(ticker), aini_var, "AINI2VIX"))) % (2**31 - 1)) + base_seed)
        try:
            p_boot, n_valid, F_obs = _gc_wild_bootstrap_null_fast(
                y=y_sub, X_restricted=XR_sub, X_unrestricted=XU_sub,
                n_boot=n_boot, seed=seed, weight_dist=weight_dist, block_size=block_size,
            )
        except Exception:
            p_boot, n_valid, F_obs = np.nan, 0, np.nan

        out_rows.append({
            "Ticker": ticker, "AINI_variant": aini_var, "Year": year_key,
            "Direction": "AINI_to_VIX",
            **coef_dict,
            "p_vix": p_vix, "p_ret": p_ret, "p_x": p_x,
            "N_obs": len(sub), "N_boot": n_boot, "N_boot_valid": n_valid,
            "F_stat": F_stat, "F_stat_obs_RSS": F_obs,
            "Original_F_pval": p_analytic, "Empirical_F_pval": p_boot,
            "r2_u": getattr(res_u, "rsquared", np.nan), "adj_r2_u": getattr(res_u, "rsquared_adj", np.nan),
        })

    return out_rows


def run_vix_causality_tests(
    aini_df: pd.DataFrame,
    fin_data: pd.DataFrame,
    version: str,
    vix_col: str = "VIX",             # column in fin_data with VIX level 
    aini_variants: Optional[List[str]] = None,
    p_vix: int = 3,                    # VIX AR order
    p_ret: int = 1,                    # return lags
    p_x: int = 3,                      # AINI lags
    min_obs: int = 60,
    n_boot: int = 1000,
    seed: int = 42,
    fdr_alpha: float = 0.05,
    cov_for_analytic: str = "HAC",
    hac_lags: Optional[int] = None,
    weight_dist: str = "rademacher",
    save_csv: bool = True,
    outdir: Optional[Path] = None,
    outname: Optional[str] = None,     # default: f"vix_causality_{version}_pvix{p_vix}_pret{p_ret}_px{p_x}.csv"
    n_jobs: int = -1,
    block_size: int = 1,
) -> pd.DataFrame:
    """
    Tests RET->VIX and AINI->VIX using the same analytic Wald-F (HAC/HC3) and wild bootstrap
    machinery as main GC. Output is FDR-adjusted per (Ticker, Year, Direction).
    """
    if aini_variants is None:
        aini_variants = ["normalized_AINI", "EMA_02", "EMA_08"]

    # --- Prep fin data (must include vix_col) ---
    fin = fin_data.copy()
    fin["Date"] = pd.to_datetime(fin["Date"])
    fin = fin.sort_values(["Ticker", "Date"])
    fin["log_return"] = fin.groupby("Ticker")["Adj Close"].transform(lambda x: np.log(x) - np.log(x.shift(1)))
    if vix_col not in fin.columns:
        raise ValueError(f"fin_data must contain a '{vix_col}' column (VIX level or transformation).")
    fin = fin.dropna(subset=["log_return", vix_col])
    fin["date"] = pd.to_datetime(fin["Date"])

    # Year splits (mirrors main API)
    threshold_24 = pd.Timestamp("2024-01-01")
    threshold_25 = pd.Timestamp("2025-01-01")
    fin_by_year = {
        2023: fin[fin["date"] < threshold_24],
        2024: fin[(fin["date"] >= threshold_24) & (fin["date"] < threshold_25)],
        2025: fin[fin["date"] >= threshold_25],
        "2023_24": fin[fin["date"] < threshold_25],
        "2024_25": fin[fin["date"] >= threshold_24],
        "2023_24_25": fin,
    }

    # AINI data
    aini = aini_df.copy()
    aini["date"] = pd.to_datetime(aini["date"])

    all_rows: List[Dict] = []
    base_seed = int(seed)

    for year_key, fin_y in fin_by_year.items():
        f = fin_y.copy().rename(columns={"Ticker": "ticker"}).sort_values(["ticker", "date"])
        # Merge in AINI variants by date (VIX assumed present in fin_data per row)
        merged = pd.merge(f, aini[["date"] + aini_variants], on="date", how="left")

        tasks: List[Tuple[pd.DataFrame, str, object, str]] = []
        for ticker, d in merged.groupby("ticker"):
            d = d.sort_values("date").reset_index(drop=True)
            for var in aini_variants:
                if var in d.columns:
                    tasks.append((d, str(ticker), year_key, var))

        results_nested: List[List[Dict]] = Parallel(n_jobs=n_jobs, backend="loky", batch_size="auto")(
            delayed(_process_ticker_vix_tests)(
                d=td, ticker=tkr, year_key=yrk, aini_var=vr, vix_col=vix_col,
                p_ret=p_ret, p_x=p_x, p_vix=p_vix,
                min_obs=min_obs, n_boot=n_boot, base_seed=base_seed,
                cov_for_analytic=cov_for_analytic, hac_lags=hac_lags,
                weight_dist=weight_dist, block_size=block_size,
            )
            for (td, tkr, yrk, vr) in tasks
        )
        for rows in results_nested:
            all_rows.extend(rows)

    out = pd.DataFrame(all_rows).sort_values(["Year", "Ticker", "AINI_variant", "Direction"])

    # ----- FDR within (Ticker, Year, Direction) -----
    if not out.empty:
        corrected = []
        for (ticker, year, direction), g in out.groupby(["Ticker", "Year", "Direction"], dropna=False):
            g = g.copy()
            p_emp = pd.to_numeric(g["Empirical_F_pval"], errors="coerce")
            if p_emp.notna().sum() >= 2:
                rej, p_corr = multipletests(p_emp, alpha=fdr_alpha, method="fdr_bh")[:2]
                g["BH_reject_F"] = rej; g["BH_corr_F_pval"] = p_corr
            else:
                g["BH_reject_F"] = False; g["BH_corr_F_pval"] = np.nan

            p_ana = pd.to_numeric(g["Original_F_pval"], errors="coerce")
            if p_ana.notna().sum() >= 2:
                rej2, p_corr2 = multipletests(p_ana, alpha=fdr_alpha, method="fdr_bh")[:2]
                g["BH_reject_F_HC3"] = rej2; g["BH_corr_F_pval_HC3"] = p_corr2
            else:
                g["BH_reject_F_HC3"] = False; g["BH_corr_F_pval_HC3"] = np.nan

            corrected.append(g)
        out = pd.concat(corrected, ignore_index=True).sort_values(["Year", "Ticker", "AINI_variant", "Direction"])

    # ----- Save -----
    if outname is None:
        outname = f"vix_causality_{version}_pvix{p_vix}_pret{p_ret}_px{p_x}.csv"
    if not out.empty and save_csv:
        if outdir is None:
            try:
                base = Path(__file__).resolve().parents[2]
            except NameError:
                base = Path.cwd()
            outdir = base / "data" / "processed" / "variables"
        outdir.mkdir(parents=True, exist_ok=True)
        out.to_csv(outdir / outname, index=False)

    return out
