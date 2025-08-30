"""
Granger Causality (both directions) with HC3/HAC covariance, null-imposing
Wild Residual Bootstrap p-values, and Benjamini–Hochberg FDR — with parallelization.

Differences vs previous MBB version:
- Uses WILD bootstrap on restricted residuals (Rademacher by default).
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


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

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
    Fit OLS with either HC3 or HAC covariance.
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
    """HC3/HAC-robust Wald F-test for joint zero restrictions on target_cols."""
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


# -------------------------
# Wild bootstrap utilities
# -------------------------

def _wild_weights(n: int, rng: np.random.Generator, dist: str = "rademacher") -> np.ndarray:
    """
    Generate i.i.d. wild bootstrap weights with mean 0 and variance 1.
    Options:
      - 'rademacher': P(+1)=P(-1)=0.5  (robust, standard choice)
      - 'normal': standard normal N(0,1)
    """
    d = dist.lower()
    if d == "rademacher":
        w = rng.integers(0, 2, size=n) * 2 - 1   # {+1,-1}
        return w.astype(float) 
    elif d == "normal":
        return rng.standard_normal(n)
    else:
        raise ValueError(f"Unknown wild weight distribution: {dist}")


def _gc_wild_bootstrap_null(
    y: pd.Series,
    X_restricted: pd.DataFrame,
    X_unrestricted: pd.DataFrame,
    tested_cols: List[str],
    n_boot: int = 1000,
    seed: int = 42,
    weight_dist: str = "rademacher",
    cov: str = "HAC",
    hac_lags: Optional[int] = None,
) -> Tuple[float, int]:
    """
    Null-imposing **wild residual bootstrap** p-value for GC Wald F-test (right-tailed).

    Steps
    -----
    1) Fit Restricted model (R): get fitted values yhat_R and residuals e_R.
    2) Compute observed F from Unrestricted (U) with tested_cols.
    3) For b=1..B:
       - Draw i.i.d. weights w_t with E[w]=0, Var[w]=1 (e.g., Rademacher).
       - e_b = e_R * w   (element-wise)
       - y_b = yhat_R + e_b   (simulation under H0 with heteroskedasticity preserved)
       - Fit U on (y_b, XU) and record F_b.
    4) p = (#{F_b >= F_obs}+1)/(B_valid+1)
    """
    rng = default_rng(seed)

    # Restricted fit (no tested_cols)
    res_R = sm.OLS(y, _add_const(X_restricted)).fit()
    yhat_R = res_R.fittedvalues
    e_R = y - yhat_R

    # Observed unrestricted F
    res_U = _fit_ols(y, X_unrestricted, cov=cov, hac_lags=hac_lags)
    F_obs, _, _, _ = _wald_F_for_zero_coefs(res_U, tested_cols)

    F_boot: List[float] = []
    n = len(y)

    for _ in range(n_boot):
        try:
            w = _wild_weights(n, rng, dist=weight_dist)        # i.i.d., mean 0, var 1
            y_b = yhat_R.reset_index(drop=True) + (e_R.reset_index(drop=True) * w)
            XU_b = X_unrestricted.reset_index(drop=True)       # same design matrix order
            res_b = _fit_ols(y_b, XU_b, cov=cov, hac_lags=hac_lags)
            F_b, _, _, _ = _wald_F_for_zero_coefs(res_b, tested_cols)
            if np.isfinite(F_b):
                F_boot.append(float(F_b))
        except Exception:
            continue

    F_boot = np.asarray(F_boot, dtype=float)
    F_boot = F_boot[np.isfinite(F_boot)]
    n_valid = int(F_boot.size)
    if n_valid < max(50, int(0.2 * n_boot)):
        return float("nan"), n_valid

    p = (np.sum(F_boot >= F_obs) + 1) / (n_valid + 1)
    return float(p), n_valid


def _auto_hac_lags(T: int) -> int:
    """Newey–West rule-of-thumb for HAC maxlags: floor(4*(T/100)^(2/9))."""
    return int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))


# -------------------------
# Parallel job per (ticker, variant)
# -------------------------

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
) -> List[Dict]:
    """
    Run both directions for a single (ticker, variant) slice and return row dicts.
    Expands regression coefficients into separate columns with direction prefixes:
      - A2R_beta_* for AINI → Return
      - R2A_beta_* for Return → AINI
    """
    out_rows: List[Dict] = []

    # Build return lags once
    df_lagged = _lag(d, "log_return", p_ret, prefix="ret")

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

    if len(sub_r) >= min_obs and (1 + len(ret_lag_cols) + len(xlag_cols)) < len(sub_r):
        y_sub = sub_r["log_return"]
        XR_sub = sub_r[ret_lag_cols]
        XU_sub = sub_r[ret_lag_cols + xlag_cols]

        use_hac = (cov_for_analytic.upper() == "HAC")
        _hac_lags = (hac_lags if use_hac and hac_lags is not None
                     else (_auto_hac_lags(len(sub_r)) if use_hac else None))
        try:
            res_u = _fit_ols(y_sub, XU_sub,
                             cov=("HAC" if use_hac else "HC3"),
                             hac_lags=_hac_lags)
            F_stat, p_analytic, df_num, df_den = _wald_F_for_zero_coefs(res_u, xlag_cols)
            coef_dict = {}
            for k, v in res_u.params.to_dict().items():
                if k.startswith(var + "_lag"):           # AINI lags
                    lagnum = k.split("lag")[-1]
                    coef_dict[f"A2R_beta_x_lag{lagnum}"] = v
                elif k.startswith("ret_lag"):            # return lags
                    lagnum = k.split("lag")[-1]
                    coef_dict[f"A2R_beta_ret_lag{lagnum}"] = v
                elif k == "const":
                    coef_dict["A2R_beta_const"] = v
                else:
                    coef_dict[f"A2R_beta_{k}"] = v
        except Exception:
            res_u = None
            F_stat, p_analytic = np.nan, np.nan
            df_num = len(xlag_cols)
            df_den = max(1, len(sub_r) - (1 + len(ret_lag_cols) + len(xlag_cols)))
            coef_dict = {}

        seed = int((abs(hash((str(year_key), str(ticker), var, "A2R"))) % (2**31 - 1)) + base_seed)
        try:
            p_boot, n_valid = _gc_wild_bootstrap_null(
                y=y_sub,
                X_restricted=XR_sub,
                X_unrestricted=XU_sub,
                tested_cols=xlag_cols,
                n_boot=n_boot,
                seed=seed,
                weight_dist=weight_dist,
                cov=("HAC" if use_hac else "HC3"),
                hac_lags=_hac_lags,
            )
        except Exception:
            p_boot, n_valid = np.nan, 0

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
                if k.startswith(var + "_lag"):           # AINI lags
                    lagnum = k.split("lag")[-1]
                    coef_dict[f"R2A_beta_x_lag{lagnum}"] = v
                elif k.startswith("ret_lag"):            # return lags
                    lagnum = k.split("lag")[-1]
                    coef_dict[f"R2A_beta_ret_lag{lagnum}"] = v
                elif k == "const":
                    coef_dict["R2A_beta_const"] = v
                else:
                    coef_dict[f"R2A_beta_{k}"] = v
        except Exception:
            res_u = None
            F_stat, p_analytic = np.nan, np.nan
            df_num = len(ret_lag_cols)
            df_den = max(1, len(sub_x) - (1 + len(xlag_cols) + len(ret_lag_cols)))
            coef_dict = {}

        seed = int((abs(hash((str(year_key), str(ticker), var, "R2A"))) % (2**31 - 1)) + base_seed)
        try:
            p_boot, n_valid = _gc_wild_bootstrap_null(
                y=y_sub,
                X_restricted=XR_sub,
                X_unrestricted=XU_sub,
                tested_cols=ret_lag_cols,
                n_boot=n_boot,
                seed=seed,
                weight_dist=weight_dist,
                cov=("HAC" if use_hac else "HC3"),
                hac_lags=_hac_lags,
            )
        except Exception:
            p_boot, n_valid = np.nan, 0

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
            "df_num": df_num,
            "df_den": df_den,
            "Original_F_pval": p_analytic,
            "Empirical_F_pval": p_boot,
            "r2_u": getattr(res_u, "rsquared", np.nan),
            "adj_r2_u": getattr(res_u, "rsquared_adj", np.nan),
        })

    return out_rows


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def run_gc_mbboot_fdr(   # keep name for backwards compatibility
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
    cov_for_analytic: str = "HAC",     # "HAC" or "HC3"
    hac_lags: Optional[int] = None,    # If None and HAC, auto by T
    weight_dist: str = "rademacher",   # 'rademacher' | 'normal'
    save_csv: bool = True,
    outdir: Optional[Path] = None,
    outname: Optional[str] = None,     # If None -> f"granger_causality_{version}.csv"
    n_jobs: int = -1,                  # parallel workers (joblib)
) -> pd.DataFrame:
    """
    Granger causality (AINI→Return and Return→AINI) with:
      - Analytic Wald/F p-values (HC3 or HAC)
      - Null-imposing **wild bootstrap** empirical p-values
      - Benjamini–Hochberg FDR per (Ticker, Year, Direction)
      - Parallel estimation across (ticker, variant) jobs

    Also computes per-(ticker, year) z-scores for `normalized_AINI` → `normalized_AINI_z`
    and includes it as an additional tested variant (only if std>0).

    Defaults
    --------
    aini_variants: ["normalized_AINI", "EMA_02", "EMA_08"]
    """
    if aini_variants is None:
        aini_variants = ["normalized_AINI", "EMA_02", "EMA_08"]

    # --- Financial data prep ---
    fin_data = fin_data.copy()
    fin_data["Date"] = pd.to_datetime(fin_data["Date"])
    fin_data = fin_data.sort_values(["Ticker", "Date"])

    fin_data["log_return"] = fin_data.groupby("Ticker")["Adj Close"].transform(
        lambda x: np.log(x) - np.log(x.shift(1))
    )
    fin_data = fin_data.dropna(subset=["log_return"])
    fin_data["date"] = pd.to_datetime(fin_data["Date"])

    # Year splits
    threshold_24 = pd.Timestamp("2024-01-01")
    threshold_25 = pd.Timestamp("2025-01-01")
    fin_data_by_year = {
        2023: fin_data[fin_data["date"] < threshold_24],
        2024: fin_data[(fin_data["date"] >= threshold_24) & (fin_data["date"] < threshold_25)],
        2025: fin_data[fin_data["date"] >= threshold_25],
        "2023_24": fin_data[fin_data["date"] < threshold_25],
        "2024_25": fin_data[fin_data["date"] >= threshold_24],
        "2023_24_25": fin_data,
    }

    # --- AINI data prep ---
    aini = aini_df.copy()
    aini["date"] = pd.to_datetime(aini["date"])

    all_rows: List[Dict] = []
    base_seed = int(seed)

    # Iterate year partitions
    for year_key, fin in fin_data_by_year.items():
        f = fin.copy().rename(columns={"Ticker": "ticker"})
        f = f.sort_values(["ticker", "date"])

        # Merge contemporaneous AINI columns
        merged = pd.merge(f, aini[["date"] + aini_variants], on="date", how="left")

        # Build per-ticker tasks
        tasks: List[Tuple[pd.DataFrame, str, object, str]] = []

        for ticker, d in merged.groupby("ticker"):
            d = d.sort_values("date").reset_index(drop=True)

            # On-the-fly z for normalized_AINI
            eff_variants = list(aini_variants)
            if "normalized_AINI" in d.columns:
                s = pd.to_numeric(d["normalized_AINI"], errors="coerce")
                mu, sd = s.mean(skipna=True), s.std(skipna=True)
                if pd.notna(sd) and float(sd) > 0.0:
                    d = d.copy()
                    d["normalized_AINI_z"] = (s - mu) / (sd if sd > 1e-12 else 1e-12)
                    eff_variants.append("normalized_AINI_z")

            for var in eff_variants:
                if var not in d.columns:
                    continue
                tasks.append((d, str(ticker), year_key, var))

        # Run tasks in parallel for this year partition
        results_nested: List[List[Dict]] = Parallel(n_jobs=n_jobs, backend="loky", batch_size="auto")(
            delayed(_process_ticker_variant)(
                d=td,
                ticker=tkr,
                year_key=yrk,
                var=vr,
                p_ret=p_ret,
                p_x=p_x,
                min_obs=min_obs,
                n_boot=n_boot,
                base_seed=base_seed,
                cov_for_analytic=cov_for_analytic,
                hac_lags=hac_lags,
                weight_dist=weight_dist,
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

            # FDR on Empirical_F_pval (primary, wild bootstrap)
            p_emp = pd.to_numeric(g["Empirical_F_pval"], errors="coerce")
            if p_emp.notna().sum() >= 2:
                rej, p_corr = multipletests(p_emp, alpha=fdr_alpha, method="fdr_bh")[:2]
                g["BH_reject_F"] = rej
                g["BH_corr_F_pval"] = p_corr
            else:
                g["BH_reject_F"] = False
                g["BH_corr_F_pval"] = np.nan

            # FDR on analytic p-values (HC3/HAC)
            p_ana = pd.to_numeric(g["Original_F_pval"], errors="coerce")
            if p_ana.notna().sum() >= 2:
                rej2, p_corr2 = multipletests(p_ana, alpha=fdr_alpha, method="fdr_bh")[:2]
                g["BH_reject_F_HC3"] = rej2   # name kept for compatibility
                g["BH_corr_F_pval_HC3"] = p_corr2
            else:
                g["BH_reject_F_HC3"] = False
                g["BH_corr_F_pval_HC3"] = np.nan

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
# ==============================
# Controls-enabled GC (add-ons)
# ==============================

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
            p_boot, n_valid = _gc_wild_bootstrap_null(y=y_sub, X_restricted=XR_sub, X_unrestricted=XU_sub,
                                                      tested_cols=xlag_cols, n_boot=n_boot, seed=seed,
                                                      weight_dist=weight_dist, cov=("HAC" if use_hac else "HC3"),
                                                      hac_lags=_hac)
        except Exception:
            p_boot, n_valid = np.nan, 0
        out_rows.append({
            "Ticker": ticker, "AINI_variant": var, "Year": year_key, "Direction": "AINI_to_RET",
            **coef_dict, "p_x": p_x, "N_obs": len(sub_r), "N_boot": n_boot, "N_boot_valid": n_valid,
            "F_stat": F_stat, "df_num": df_num, "df_den": df_den,
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
            p_boot, n_valid = _gc_wild_bootstrap_null(y=y_sub, X_restricted=XR_sub, X_unrestricted=XU_sub,
                                                      tested_cols=ret_lag_cols, n_boot=n_boot, seed=seed,
                                                      weight_dist=weight_dist, cov=("HAC" if use_hac else "HC3"),
                                                      hac_lags=_hac)
        except Exception:
            p_boot, n_valid = np.nan, 0
        out_rows.append({
            "Ticker": ticker, "AINI_variant": var, "Year": year_key, "Direction": "RET_to_AINI",
            **coef_dict, "p_x": p_x, "N_obs": len(sub_x), "N_boot": n_boot, "N_boot_valid": n_valid,
            "F_stat": F_stat, "df_num": df_num, "df_den": df_den,
            "Original_F_pval": p_analytic, "Empirical_F_pval": p_boot,
            "r2_u": getattr(res_u, "rsquared", np.nan), "adj_r2_u": getattr(res_u, "rsquared_adj", np.nan),
        })
    return out_rows


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
            eff_variants = list(aini_variants)
            if "normalized_AINI" in d.columns:
                s = pd.to_numeric(d["normalized_AINI"], errors="coerce")
                mu, sd = s.mean(skipna=True), s.std(skipna=True)
                if pd.notna(sd) and float(sd) > 0.0:
                    d = d.copy()
                    d["normalized_AINI_z"] = (s - mu) / (sd if sd > 1e-12 else 1e-12)
                    eff_variants.append("normalized_AINI_z")
            for var in eff_variants:
                if var in d.columns:
                    tasks.append((d, str(ticker), year_key, var))

        results_nested: List[List[Dict]] = Parallel(n_jobs=n_jobs, backend="loky", batch_size="auto")(
            delayed(_process_ticker_variant_with_controls)(
                d=td, ticker=tkr, year_key=yrk, var=vr,
                p_ret=p_ret, p_x=p_x,
                controls_lags=controls_lags,
                min_obs=min_obs, n_boot=n_boot, base_seed=base_seed,
                cov_for_analytic=cov_for_analytic, hac_lags=hac_lags, weight_dist=weight_dist,
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
        ctrl_tag = f"{control_var}_lags" + "".join(f"{k}{v}" for k,v in sorted((controls_lags or {}).items()))
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


