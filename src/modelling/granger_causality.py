"""
Granger Causality Estimation Module

This module estimates Granger causality between AI Narrative Indices (AINI) and the log of financial returns
using bootstrapping and heteroskedasticity diagnostics. It supports:

- bootstrapped empirical p-values with FDR correction
- Breusch-Pagan and White tests for residual heteroskedasticity
- flexible directionality: AINI → returns or returns → AINI

"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.tools.tools import add_constant
from tqdm import tqdm

# Local import (adjust if structure changes)
sys.path.append(str(Path(__file__).resolve().parent.parent / "modelling"))
from format_te_gc_inputs import get_ticker_for_granger


def estimate_bootstraped_gc(
    aini_data,
    fin_data_by_year,
    aini_variants=None,
    lag_range=range(1, 11),
    n_bootstrap=1000,
    alpha=0.05,
    reverse=False,
    window=None,
    seed=42
):
    """
    Estimate Granger causality with bootstrapped F-tests, FDR correction,
    and heteroskedasticity diagnostics on model residuals.

    Parameters
    ----------
    aini_data : pd.DataFrame
        AINI time series with datetime index.
    fin_data_by_year : dict
        Mapping from year to financial returns DataFrame.
    aini_variants : list of str, optional
        List of AINI variable names to test.
    lag_range : range
        Lags to test for Granger causality.
    n_bootstrap : int
        Number of bootstrap samples.
    alpha : float
        Significance level for FDR correction.
    reverse : bool
        If True, tests returns → AINI.
    window : int or None
        Optional identifier to tag the result file.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Full results with F-test p-values, bootstrap p-values, heteroskedasticity stats, and model info.
    """
    np.random.seed(seed)

    if aini_variants is None:
        aini_variants = [
            "normalized_AINI", "EMA_02", "EMA_08", "normalized_AINI_growth"
        ]

    root = Path(__file__).resolve().parents[2]
    var_path = root / "data" / "processed" / "variables"
    var_path.mkdir(parents=True, exist_ok=True)

    direction_flag = 'return_to_AINI' if reverse else 'AINI_to_return'
    file_name = f"gc_bootstrap_F_all_lags_groupwise_{direction_flag}"
    if window is not None:
        file_name += f"_w{window}"
    file_name += ".csv"
    output_path = var_path / file_name

    results = []

    for var in tqdm(aini_variants):
        for year, fin_data in fin_data_by_year.items():
            ticker_arrays = get_ticker_for_granger(fin_data, aini_data, var)

            for ticker, data in ticker_arrays.items():
                try:
                    clean_data = data[~np.isnan(data).any(axis=1)]
                    aini_raw = clean_data[:, 0]
                    return_raw = clean_data[:, 1]
                    min_len = min(len(aini_raw), len(return_raw))

                    y = return_raw[-min_len:]
                    x = aini_raw[-min_len:]
                    base_input = np.column_stack([y, x]) if reverse else np.column_stack([x, y])

                    for lag in lag_range:
                        try:
                            orig_test = grangercausalitytests(base_input, maxlag=lag, verbose=False)
                            orig_f_p = orig_test[lag][0]['ssr_ftest'][1]
                            model = orig_test[lag][1][0]
                            aic = model.aic
                            bic = model.bic
                            coef_names = model.model.exog_names
                            coefs = model.params

                            # Residual heteroskedasticity tests
                            resid = model.resid
                            exog = add_constant(model.model.exog)
                            bp_stat, bp_pval, _, _ = het_breuschpagan(resid, exog)
                            white_stat, white_pval, _, _ = het_white(resid, exog)

                            boot_f_pvals = []
                            for _ in range(n_bootstrap):
                                idx = np.random.choice(len(base_input), len(base_input), replace=True)
                                boot_data = base_input[idx]
                                try:
                                    test = grangercausalitytests(boot_data, maxlag=lag, verbose=False)
                                    boot_f_pvals.append(test[lag][0]['ssr_ftest'][1])
                                except:
                                    continue

                            emp_f_p = (np.sum(np.array(boot_f_pvals) <= orig_f_p) + 1) / (len(boot_f_pvals) + 1)

                            result = {
                                "Ticker": ticker,
                                "AINI_variant": var,
                                "Year": year,
                                "Lag": lag,
                                "Original_F_p": orig_f_p,
                                "Empirical_F_p": emp_f_p,
                                "AIC": aic,
                                "BIC": bic,
                                "N_boot": len(boot_f_pvals),
                                "bp_stat" : bp_stat,
                                "BP_pval": bp_pval,
                                "white_stat" : white_stat,
                                "White_pval": white_pval,
                                "BP_reject": bp_pval < 0.05,
                                "White_reject": white_pval < 0.05,
                                "Reverse": reverse,
                                "window": window
                            }

                            for name, val in zip(coef_names, coefs):
                                result[f"coef_{name}"] = val

                            results.append(result)

                        except Exception as e:
                            print(f"Lag {lag} failed for {ticker} ({var}, {year}): {e}")

                except Exception as e:
                    print(f"Ticker {ticker} failed for ({var}, {year}): {e}")

    gc_df = pd.DataFrame(results)

    # Apply FDR correction per (Ticker, Year)
    corrected = []
    for (ticker, year), group in gc_df.groupby(["Ticker", "Year"]):
        try:
            reject, pvals_corr = multipletests(group["Empirical_F_p"], alpha=alpha, method="fdr_bh")[:2]
            group["BH_reject_F"] = reject
            group["BH_corr_F"] = pvals_corr
        except Exception as e:
            print(f"BH correction failed for {ticker}, {year}: {e}")
            group["BH_reject_F"] = False
            group["BH_corr_F"] = np.nan
        corrected.append(group)

    gc_df = pd.concat(corrected, ignore_index=True)
    gc_df.to_csv(output_path, index=False)
    print(f"Bootstrapped Granger results saved to: {output_path}")

    return gc_df
