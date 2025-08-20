'''
CLI tool to clean database by removing duplicates, NAs & redunant topics.
Usage:
    python run_clean_database.py
'''

import typer
import sys
from pathlib import Path

root = Path(__file__).parent.parent
sys.path.append(str(root / "preprocessing"))
from clean_database import clean_database

def main(
    year: int = typer.Option(None, help="Target year (2023, 2024, 2025). Leave empty to run all.")
):
    years = [2023, 2024, 2025] if year is None else [year]
    for y in years:
        typer.echo(f"\n=== Cleaning started for {y} ===")
        results = clean_database(year=y)
        typer.echo(f"Cleaned articles saved to: {results['db_path']}")
        typer.echo(f"Removed {results['n_dropped_articles']} noisy/duplicate articles")
        typer.echo(f"Removed articles log saved to: {results['removed_path']}")

if __name__ == "__main__":
    typer.run(main)
