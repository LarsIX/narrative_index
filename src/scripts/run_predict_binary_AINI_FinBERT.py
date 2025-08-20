"""
CLI tool to Label WSJ articles with presence of AI-related narrative using fintetuned FinBERT model.

Usage:
    # single year
    python run_predict_binary_AINI_FinBERT.py run-bert --year 2024 --batch-size 16

    # range of years (defaults 2023..2025)
    python run_predict_binary_AINI_FinBERT.py run-bert-all --start-year 2023 --end-year 2025 --batch-size 16

    # explicit list (defaults [2023, 2024, 2025])
    python run_predict_binary_AINI_FinBERT.py run-bert-years --years 2023 --years 2024 --years 2025 --batch-size 16
"""
import sys
from pathlib import Path
from typing import List
import typer
# Add 'src/modelling' to Python path
root = Path(__file__).parent  # = src/scripts
modelling_path = (root.parent / 'modelling').resolve()
sys.path.append(str(modelling_path))

from predict_binary_AINI_FinBERT import label_data

app = typer.Typer(help="Predict AINI labels with FinBERT.")

@app.command(name="run-bert")
def run_bert(
    year: int = typer.Option(..., help="Enter year to label"),
    batch_size: int = typer.Option(16, help="Batch size (default=16)")
):
    """Run FinBERT-based inference for a single year."""
    typer.echo(f"[INFO] Starting FinBERT for {year}")
    result = label_data(year=year, batch_size=batch_size)
    typer.echo(f"[DONE] {year} → {result['n_labeled_articles']} articles | Saved: {result['output_path']}")

@app.command(name="run-bert-all")
def run_bert_all(
    start_year: int = typer.Option(2023, help="Start year (default=2023)"),
    end_year: int = typer.Option(2025, help="End year inclusive (default=2025)"),
    batch_size: int = typer.Option(16, help="Batch size (default=16)")
):
    """Run FinBERT-based inference for a continuous year range."""
    for y in range(start_year, end_year + 1):
        typer.echo(f"[INFO] Starting FinBERT for {y}")
        result = label_data(year=y, batch_size=batch_size)
        typer.echo(f"[DONE] {y} → {result['n_labeled_articles']} articles | Saved: {result['output_path']}")
    typer.echo("[ALL DONE] run-bert-all complete.")

@app.command(name="run-bert-years")
def run_bert_years(
    years: List[int] = typer.Option([2023, 2024, 2025], "--years", "-y", help="Years to run"),
    batch_size: int = typer.Option(16, help="Batch size (default=16)")
):
    """Run FinBERT-based inference for an explicit list of years."""
    for y in years:
        typer.echo(f"[INFO] Starting FinBERT for {y}")
        result = label_data(year=y, batch_size=batch_size)
        typer.echo(f"[DONE] {y} → {result['n_labeled_articles']} articles | Saved: {result['output_path']}")
    typer.echo("[ALL DONE] run-bert-years complete.")

if __name__ == "__main__":
    app()
