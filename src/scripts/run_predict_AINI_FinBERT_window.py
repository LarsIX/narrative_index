"""
CLI tool to run AINI prediction via FinBERT utilizing context windows.

Usage examples:
    # single year, all windows
    python run_predict_AINI_FinBERT_window.py run --year 2023 --context-window 0 --context-window 1 --context-window 2

    # single year, default runs 0,1,2
    python run_predict_AINI_FinBERT_window.py run --year 2023

    # multi-year, explicit windows
    python run_predict_AINI_FinBERT_window.py run-all --start-year 2023 --end-year 2025 --context-window 0 --context-window 1 --context-window 2

"""

from pathlib import Path
import sys
from typing import List
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
    context_window: List[int] = typer.Option(
        [0, 1, 2],
        "--context-window",
        "-w",
        help="Context window size(s). Pass multiple times to run several (e.g., -w 0 -w 1 -w 2)."
    ),
    max_tokens: int = typer.Option(512, help="Max token length (<= 512 for BERT)")
):
    typer.echo(f"[INFO] FinBERT starting for {year}...")
    for cw in context_window:
        typer.echo(f"  - Running context window = {cw}")
        result = predict_sentiment(
            year=year,
            batch_size=batch_size,
            context_window=cw,
            max_tokens=max_tokens
        )
        typer.echo(f"    [DONE] Saved to: {result['output_path']} | Device: {result['device']}")
    typer.echo(f"[ALL DONE] Year {year}")

@app.command()
def run_all(
    start_year: int = typer.Option(..., help="Start year (e.g., 2023)"),
    end_year: int = typer.Option(..., help="End year (e.g., 2025)"),
    batch_size: int = typer.Option(4, help="Batch size for FinBERT"),
    context_window: List[int] = typer.Option(
        [0, 1, 2],
        "--context-window",
        "-w",
        help="Context window size(s). Pass multiple times to run several (e.g., -w 0 -w 1 -w 2)."
    ),
    max_tokens: int = typer.Option(512, help="Max token length for BERT")
):
    for year in range(start_year, end_year + 1):
        typer.echo(f"\n[INFO] Running FinBERT for {year}")
        for cw in context_window:
            typer.echo(f"  - Context window = {cw}")
            result = predict_sentiment(
                year=year,
                batch_size=batch_size,
                context_window=cw,
                max_tokens=max_tokens
            )
            typer.echo(f"    [DONE] {year}, cw={cw} → {result['output_path']} | Device: {result['device']}")
    typer.echo("[ALL DONE] run-all complete.")

if __name__ == "__main__":
    app()
