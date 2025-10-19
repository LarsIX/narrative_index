# ============================================================
# VIX endogeneity / collider precondition tests
#   - RET -> VIX  (do return lags predict VIX beyond VIX's own lags?)
#   - AINI -> VIX (do AINI lags predict VIX beyond VIX's own lags?)
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
    vix_col: str = "VIX",             # column in fin_data with VIX level (you can switch to ΔVIX upstream if you prefer)
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
