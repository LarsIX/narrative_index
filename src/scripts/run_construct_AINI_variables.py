"""
CLI app to construct AINI variables from FinBERT prediction files.

Usage:
    python run_construct_AINI_variables.py `
        FinBERT_AINI_prediction_2023_windsize_0.csv `
        FinBERT_AINI_prediction_2024_windsize_0.csv `
        --file3 FinBERT_AINI_prediction_2025_windsize_0.csv `
        --vers w0
"""

import typer
import pandas as pd
from pathlib import Path
import sys

# define path for files & function
project_root = Path(__file__).resolve().parents[2]
func_path = project_root / "src" / "modelling"
sys.path.append(str(func_path))

from construct_AINI_variables import build_df

app = typer.Typer()


@app.command()
def run(
    file1: str = typer.Argument(..., help="Filename in data/processed/variables/"),
    file2: str = typer.Argument(..., help="Second filename"),
    file3: str = typer.Option(None, help="Optional third filename"),
    file4: str = typer.Option(None, help="Optional fourth filename"),
    cutoff_min: str = typer.Option("2023-03-31", help="Lower date bound (YYYY-MM-DD)"),
    cutoff_max: str = typer.Option("2030-03-31", help="Upper date bound (YYYY-MM-DD)"),
    vers: str = typer.Option("binary", help="Version tag for output CSV name")
):
    """
    Construct AI Narrative Index (AINI) variables from raw FinBERT prediction files.

    This function loads up to four CSV files from `data/processed/variables/`,
    applies temporal filtering, computes hype scores and moving averages,
    and saves the transformed DataFrame as a versioned CSV.

    Parameters
    ----------
    file1 : str
        First CSV filename (mandatory).
    file2 : str
        Second CSV filename (mandatory).
    file3 : str, optional
        Third CSV filename.
    file4 : str, optional
        Fourth CSV filename.
    cutoff_min : str
        Lower bound for date filtering (YYYY-MM-DD).
    cutoff_max : str
        Upper bound for date filtering (YYYY-MM-DD).
    vers : str
        Version string used in output file naming.
    """

    data_dir = project_root / "data" / "processed" / "variables"

    def resolve(f): return data_dir / f if f else None

    df1 = pd.read_csv(resolve(file1))
    df2 = pd.read_csv(resolve(file2))
    df3 = pd.read_csv(resolve(file3)) if file3 else None
    df4 = pd.read_csv(resolve(file4)) if file4 else None

    build_df(
        df_1=df1,
        df_2=df2,
        df_3=df3,
        df_4=df4,
        cutoff_min=pd.to_datetime(cutoff_min),
        cutoff_max=pd.to_datetime(cutoff_max),
        vers=vers
    )

if __name__ == "__main__":
    app()
