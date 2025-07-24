"""
Module to construct AI Narrative Index (AINI) variables from raw model predictions.

This script concatenates prediction outputs from different time periods, filters them by date,
computes normalized hype scores, exponential moving averages (EMAs), and growth metrics,
and saves the result as a CSV file for further analysis.
"""

import pandas as pd
import numpy as np
import typer
from pathlib import Path
import datetime as dt

def build_df(df_1, df_2, df_3=None, df_4=None, cutoff_min=None, cutoff_max=None, vers="binary"):
    """
    Concatenates raw AINI predictions for different intervals with min and max cutoffs and outputs EMAs, relative frequencies, and growth rates.

    Parameters
    ----------
    df_1, df_2 : pd.DataFrame
        Required DataFrames containing 'date' and 'predicted_label' columns.
    df_3, df_4 : pd.DataFrame, optional
        Additional DataFrames to include.
    cutoff_min, cutoff_max : datetime, optional
        Min/max date bounds for filtering.
    vers : str
        Version label used in output filename.

    Returns
    -------
    pd.DataFrame
        Final DataFrame with AINI metrics.
    """

    # Validate and preprocess
    df_list = [df for df in [df_1, df_2, df_3, df_4] if df is not None]

    for df in df_list:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        else:
            typer.echo(f"DataFrame is missing 'date' column:\n{df.head()}")
            return None

    cutoff_min = dt.datetime(2023, 3, 31) if cutoff_min is None else cutoff_min
    cutoff_max = dt.datetime(2030, 3, 31) if cutoff_max is None else cutoff_max

    # Concatenate inputs
    df_concat = pd.concat(df_list, ignore_index=True)
    df_cut = df_concat[(df_concat["date"] > cutoff_min) & (df_concat["date"] < cutoff_max)]

    # Rename and aggregate
    df_cut = df_cut.rename(columns={"predicted_label": "hype_score"})

    simple_AINI = df_cut.groupby("date")["hype_score"].sum()
    daily_count = df_cut.groupby("date")["hype_score"].count()
    normalized_AINI = simple_AINI / daily_count

    final_df = normalized_AINI.reset_index().rename(columns={"hype_score": "normalized_AINI"})
    final_df["simple_AINI"] = final_df["date"].map(simple_AINI)

    final_df["EMA_02"] = final_df["normalized_AINI"].ewm(alpha=0.2, adjust=False).mean()
    final_df["EMA_08"] = final_df["normalized_AINI"].ewm(alpha=0.8, adjust=False).mean()

    final_df = final_df.sort_values("date").set_index("date")
    final_df["normalized_AINI_growth"] = final_df["normalized_AINI"].diff()
    final_df = final_df.reset_index()

    typer.echo(f"Created dataframe with {len(final_df)} observations.")

    root = Path(__file__).resolve().parents[2]
    output_path = root / "data" / "processed" / "variables" / f"{vers}_AINI_variables.csv"
    final_df.to_csv(output_path, index=False)

    typer.echo(f"Saved {vers}_AINI_variables.csv to processed/variables")

    return final_df
