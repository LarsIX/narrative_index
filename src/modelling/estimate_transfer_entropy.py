"""
Transfer Entropy Estimation Module

This module estimates Transfer Entropy (TE) from AI Narrative Indices (AINI) to financial return series
for a set of tickers, using the Kraskov estimator via IDTxl.

It supports:
- multiple AINI variants,
- different k-nearest neighbor settings,
- automatic NaN handling and logging,
- saving results per `k` and as combined CSVs.

Author: [Your Name]
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from idtxl.data import Data
from idtxl.bivariate_te import BivariateTE
import sys
import math
from fnmatch import fnmatch

import jpype
from pathlib import Path

# Resolve base project path
project_root = Path(__file__).resolve().parents[2]

# Add path for input formatter module
modelling_path = project_root / "src" / "modelling"
sys.path.append(str(modelling_path))

# load custom formatter module
from format_te_gc_inputs import get_ticker_for_TE

# Add path for jvm helper module
jvm_path = project_root / "src" / "scripts"
sys.path.append(str(jvm_path))

# Import and start JVM
from init_jvm import start_jvm
start_jvm()

# Monkey-patch for IDTxl compatibility
np.math = math

# dfeinve var_path 
var_path = project_root / "data" / "processed" / "variables"

def estimate_te(
    year,
    fin_data,
    aini_data,
    aini_vars=None,
    target_name="LogReturn",
    var_path=var_path,
    max_lag_sources=2,
    n_perm=200,
    k_list=[2, 3, 4],
    save=False,
    window=None
):
    """
    Estimate Transfer Entropy (TE) from AI Narrative Index (AINI) variants to financial returns.

    Parameters
    ----------
    year : int or str
        Label for output filenames (e.g., 2024).
    fin_data : pd.DataFrame
        DataFrame with columns ['Date', 'LogReturn', 'Ticker'] for financial variables.
    aini_data : pd.DataFrame
        DataFrame with columns ['date', ...] where ... are AINI variants.
    aini_vars : list of str or None
        List of AINI variables to evaluate. If None, selects all columns matching '*AINI*'.
    target_name : str
        Name of the target financial variable, used for saving/output labels.
    var_path : Path
        Directory for saving result CSVs.
    max_lag_sources : int
        Maximum lag to include for TE source variables (min lag is 1).
    n_perm : int
        Number of permutations for FDR-based significance testing.
    k_list : list of int
        List of `k` values (nearest neighbors) for Kraskov estimator.
    save : bool
        Whether to save CSVs for each k and a combined benchmark file.
    window : int or None
        Window size used in AINI preprocessing, passed for logging/saving.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all TE results across k-values and AINI variants.
    """

    all_results = []

    # Automatically detect AINI variables if none provided
    if aini_vars is None:
        aini_vars = [col for col in aini_data.columns if fnmatch(col, "*AINI*")]
    print(f"Detected AINI variants: {aini_vars}")

    for aini_var in aini_vars:
        print(f"\n Estimating TE for AINI variable: {aini_var}")

        ticker_arrays = get_ticker_for_TE(
            fin_data=fin_data,
            aini_data=aini_data,
            aini_var=aini_var,
        )

        for k_val in k_list:
            print(f"\n Running TE estimation with k = {k_val}")

            results_list = []

            for ticker in tqdm(ticker_arrays, desc=f"k={k_val}"):
                array = ticker_arrays[ticker]

                # Drop leading NaNs
                nan_mask = ~np.isnan(array).any(axis=0)
                first_valid_index = np.argmax(nan_mask)
                array_cleaned = array[:, first_valid_index:]

                if np.isnan(array_cleaned).any():
                    print(f"{ticker}: skipped – residual NaNs")
                    continue

                try:
                    data = Data(array_cleaned, dim_order='ps')
                    analysis = BivariateTE()
                    settings = {
                        "cmi_estimator": "JidtKraskovCMI",
                        "min_lag_sources": 1,
                        "max_lag_sources": max_lag_sources,
                        "n_perm": n_perm,
                        "k": k_val,
                        "target": 0,
                        "sources": [1]
                    }

                    result = analysis.analyse_single_target(
                        settings=settings,
                        data=data,
                        target=0
                    )
                    
                    te_val = result.get_single_target(0, fdr=True)
                    te_score = te_val["selected_sources_te"][0]
                    lag = result.get_target_delays(target=0, criterion='max_te', fdr=True)[0]

                    results_list.append({
                        "Ticker": ticker,
                        "Source": aini_var,
                        "Target": target_name,
                        "Source → Target": f"{aini_var} → {target_name}",
                        "Max TE Lag": lag,
                        "Transfer Entropy": te_score,
                        "Self-Link": False,
                        "FDR-filtered": True,
                        "k_neighbors": k_val,
                        "p-value threshold": 0.05,
                        "Year": year,
                        "AINI Variant": aini_var,
                        "Max Lag": max_lag_sources,
                        "n_perm": n_perm,
                        "window": window
                    })

                except Exception as e:
                    print(f"{ticker}: TE analysis failed → {e}")

            df_k = pd.DataFrame(results_list)
            all_results.append(df_k)

            if save:
                filename = f"{aini_var}_to_{target_name}_te_k{k_val}_{year}.csv"
                df_k.to_csv(var_path / filename, index=False)
                print(f"Saved results for k={k_val} to {filename}")

    df_all = pd.concat(all_results, ignore_index=True)

    if save:
        combined_name = f"te_benchmark_{year}_window{window}.csv" if window else f"te_benchmark_{year}.csv"
        df_all.to_csv(var_path / combined_name, index=False)
        print(f"\n Combined TE results saved to {combined_name}")

    return df_all
