"""
Combine Article Tables from WSJ Databases (2023–2025)

This module provides functions to load the 'article' table from multiple SQLite
databases and combine them into a single pandas DataFrame.

Intended for use in preprocessing pipelines that require unified access
to Wall Street Journal article data across years.
"""

import sqlite3
import pandas as pd
from pathlib import Path


def load_article_table(db_path: Path) -> pd.DataFrame:
    """
    Load the 'article' table from a SQLite database.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite .db file.

    Returns
    -------
    pd.DataFrame
        Table named 'article' from the given database.
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql("SELECT * FROM article", conn)


def combine_article_databases() -> pd.DataFrame:
    """
    Loads and concatenates the 'article' tables from the WSJ databases
    for 2023, 2024, and 2025.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all article records from the 3 years.
    """
    base_dir = Path(__file__).resolve().parents[2] / "data" / "processed" / "articles"

    db_files = [
        base_dir / "articlesWSJ_clean_2023.db",
        base_dir / "articlesWSJ_clean_2024.db",
        base_dir / "articlesWSJ_clean_2025.db"
    ]

    dfs = []
    for db_path in db_files:
        if db_path.exists():
            df = load_article_table(db_path)
            dfs.append(df)
        else:
            print(f"Warning: File not found – {db_path}")

    if not dfs:
        raise FileNotFoundError("No database files were found or loaded.")

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def save_combined_csv(df: pd.DataFrame, filename: str = "combined_articles_2023_2025.csv") -> None:
    """
    Save the combined article DataFrame as a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        The combined article records.

    filename : str
        Output CSV filename, stored in data/processed/articles/
    """
    output_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "articles" / filename
    df.to_csv(output_path, index=False)
