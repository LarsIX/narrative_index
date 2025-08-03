"""
CLI tool to run AINI prediction via FinBERT utilizing context windows.

Usage:
    python run_predict_AINI_FinBERT_window.py run --year 2023 --batch-size 4 --context-window 1 --max-tokens 512
    python run_predict_AINI_FinBERT_window.py run-all --start-year 2023 --end-year 2025 --context-window 2
"""

from pathlib import Path
import sys
import typer
import pandas as pd

# Add 'src/modelling' to Python path
root = Path(__file__).parent  # = src/scripts
modelling_path = (root.parent / 'modelling').resolve()
sys.path.append(str(modelling_path))

# Import prediction function
from predict_AINI_FinBERT_window import predict_sentiment

# Initialize CLI app
app = typer.Typer(help="Run AINI-prediction (-1 to +1) via FinBERT using contextual windows.")

@app.command()
def run(
    year: int = typer.Option(..., help="Target year (e.g., 2023)"),
    batch_size: int = typer.Option(4, help="Batch size for FinBERT"),
    context_window: int = typer.Option(1, help="Number of surrounding sentences to include"),
    max_tokens: int = typer.Option(512, help="Max token length (<= 512 for BERT)")
):
    typer.echo(f"FinBERT starting for {year}...")
    result = predict_sentiment(
        year=year,
        batch_size=batch_size,
        context_window=context_window,
        max_tokens=max_tokens
    )
    typer.echo(f"Prediction done. Saved to: {result['output_path']}")
    typer.echo(f"Used device: {result['device']}")

@app.command()
def run_all(
    start_year: int = typer.Option(..., help="Start year (e.g., 2023)"),
    end_year: int = typer.Option(..., help="End year (e.g., 2025)"),
    batch_size: int = typer.Option(4, help="Batch size for FinBERT"),
    context_window: int = typer.Option(1, help="Context window size"),
    max_tokens: int = typer.Option(512, help="Max token length for BERT")
):
    for year in range(start_year, end_year + 1):
        typer.echo(f"\n[INFO] Running FinBERT for {year}")
        result = predict_sentiment(
            year=year,
            batch_size=batch_size,
            context_window=context_window,
            max_tokens=max_tokens
        )
        typer.echo(f"[DONE] {year} → Saved to: {result['output_path']} | Device: {result['device']}")


if __name__ == "__main__":
    app()
