"""
CLI app to combine 'article' tables from WSJ databases (2023–2025) into a single CSV.

Usage:
    python run_combine_article_dbs.py --output combined_articles_2023_2025.csv
"""

import typer
from pathlib import Path
import sys

# Set up import path
project_root = Path(__file__).resolve().parents[2]
preprocessing_path = project_root / "src" / "preprocessing"
sys.path.append(str(preprocessing_path))

from combine_article_dbs import combine_article_databases, save_combined_csv

app = typer.Typer()


@app.command()
def run(
    output: str = typer.Option("combined_articles_2023_2025.csv", help="Name of the output CSV file")
):
    """
    Combines article data from 2023–2025 WSJ SQLite databases and saves as a single CSV.
    """
    df = combine_article_databases()
    save_combined_csv(df, filename=output)
    print(f"✅ Combined file written to: data/processed/articles/{output}")


if __name__ == "__main__":
    app()
