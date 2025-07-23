"""
stationarity_testing.py

Provides functions to assess the stationarity of AINI sentiment indices and financial time series
using Augmented Dickey-Fuller (ADF) and Phillips-Perron (PP) tests.

Functions
---------
- test_stationarity_aini_variants: Runs ADF and PP tests on multiple AINI variants across yearly subsets.
- test_stationarity_fin_variables: Applies ADF and PP tests to financial variables grouped by ticker and time period.

Results are saved as CSV files and returned as pandas DataFrames.

Dependencies
------------
- pandas
- pathlib
- statsmodels
- arch
"""

import pandas as pd
from pathlib import Path
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import PhillipsPerron


def test_stationarity_aini_variants(aini_data, variants=None, window = None):
    """
    Perform stationarity tests (ADF & Phillips-Perron) on AINI index variants over different time windows.

    Parameters
    ----------
    aini_data : DataFrame
        AINI dataset containing date and all target variants as columns.
    variants : list of str, optional
        Specific AINI columns to test. If None, a default list is used.
    window : int or str, optional
        Context window size for documentation (not used in logic).

    Returns
    -------
    DataFrame
        Results of stationarity tests including statistics and p-values, saved as CSV.
    """

    root = Path(__file__).resolve().parents[2]
    var_path = root / "data" / "processed" / "variables"
    var_path.mkdir(parents=True, exist_ok=True)

    output_file = var_path / (
        f"stationarity_tests_aini_window{window}_var.csv" if window else "stationarity_tests_aini_var.csv"
    )

    if variants is None:
        variants = [
            "normalized_AINI", "MA_7", "MA_30", "EMA_02", "EMA_04", "EMA_06",
            "EMA_08", "normalized_AINI_growth", "relative_AINI_weekly", "relative_AINI_month"
        ]

    aini_data['date'] = pd.to_datetime(aini_data['date'])

    # Subsets by time period
    data_subsets = {
        "2023": aini_data[aini_data['date'] < "2024-01-01"],
        "2024": aini_data[(aini_data['date'] >= "2024-01-01") & (aini_data['date'] < "2025-01-01")],
        "2025": aini_data[aini_data['date'] >= "2025-01-01"],
        "2023_24": aini_data[aini_data['date'] < "2025-01-01"],
        "2024_25": aini_data[aini_data['date'] >= "2024-01-01"],
        "2023_25": aini_data.copy(),
    }

    results = []

    for period, df in data_subsets.items():
        for variant in variants:
            try:
                series = df[variant].dropna()

                adf_stat, adf_p, *_ = adfuller(series, autolag='AIC')
                pp_test = PhillipsPerron(series)
                pp_stat = pp_test.stat
                pp_p = pp_test.pvalue
                agree = adf_p < 0.05 and pp_p < 0.05

                results.append({
                    "Period": period,
                    "AINI_variant": variant,
                    "Context_window": window,
                    "ADF_stat": adf_stat,
                    "ADF_p": adf_p,
                    "PP_stat": pp_stat,
                    "PP_p": pp_p,
                    "agree_stationarity": agree
                })

            except Exception as e:
                print(f"⚠️ Stationarity test failed for {variant} ({period}): {e}")

    # Save final results
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    print(f"✅ Stationarity test results saved to: {output_file}")

    # return df for further analysis
    return df_results

def test_stationarity_fin_variables(fin_data, fin_var=None):
    """
    Run ADF & PP stationarity tests on financial time series, grouped by ticker and time period.

    Parameters
    ----------
    fin_data : DataFrame
        Financial dataset with 'Date', 'Ticker', and target variables.
    fin_var : list of str, optional
        List of variable names to test (e.g., 'Adj Close', 'LogReturn').

    Returns
    -------
    DataFrame
        Results of stationarity tests saved as CSV and returned as DataFrame.
    """

    root = Path(__file__).resolve().parents[2]
    var_path = root / "data" / "processed" / "variables"
    var_path.mkdir(parents=True, exist_ok=True)

    output_file = var_path / "stationarity_tests_fin_var.csv"

    if fin_var is None:
        fin_var = ["Adj Close", "Volume", "LogReturn"]

    fin_data['date'] = pd.to_datetime(fin_data['Date'])

    # Define year-based subsets
    data_subsets = {
        "2023": fin_data[fin_data['date'] < "2024-01-01"],
        "2024": fin_data[(fin_data['date'] >= "2024-01-01") & (fin_data['date'] < "2025-01-01")],
        "2025": fin_data[fin_data['date'] >= "2025-01-01"],
        "2023_24": fin_data[fin_data['date'] < "2025-01-01"],
        "2024_25": fin_data[fin_data['date'] >= "2024-01-01"],
        "2023_25": fin_data.copy(),
    }

    results = []

    for period, df_period in data_subsets.items():
        for ticker, df_group in df_period.groupby("Ticker"):
            for variable in fin_var:
                try:
                    series = df_group[variable].dropna()

                    adf_stat, adf_p, *_ = adfuller(series, autolag='AIC')
                    pp_test = PhillipsPerron(series)
                    pp_stat = pp_test.stat
                    pp_p = pp_test.pvalue
                    agree = adf_p < 0.05 and pp_p < 0.05


                    results.append({
                        "Period": period,
                        "Ticker": ticker,
                        "Financial_variable": f"{ticker}_{variable}",
                        "ADF_stat": adf_stat,
                        "ADF_p": adf_p,
                        "PP_stat": pp_stat,
                        "PP_p": pp_p,
                        "agree_stationarity": agree
                    })

                except Exception as e:
                    print(f"⚠️ Stationarity test failed for {ticker} / {variable} ({period}): {e}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    print(f"✅ Stationarity test results saved to: {output_file}")
    return df_results