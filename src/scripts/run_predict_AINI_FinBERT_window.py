"""
CLI tool to run AINI prediction via FinBERT utilizing context windows.

Usage:
    python run_predict_AINI_FinBERT_window.py --year 2023 --batch-size 4 --context-window 1 --max-tokens 512
"""

from pathlib import Path
import sys
import typer

# Add 'src/modelling' to Python path
root = Path(__file__).parent  # = src/scripts
modelling_path = (root.parent / 'modelling').resolve()
sys.path.append(str(modelling_path))

# import function
from predict_AINI_FinBERT_window import predict_sentiment

# Initialize CLI app
app = typer.Typer(help="Run AINI-prediction (-1 to +1) via FinBERT using contextual windows.")

@app.command()
def run(
    year: int = typer.Option(..., help="Target year (e.g., 2023)"),
    batch_size: int = typer.Option(4, help="Batch size for FinBERT (default: 4)"),
    context_window: int = typer.Option(1, help="Number of surrounding sentences to include (default: 1)"),
    max_tokens: int = typer.Option(512, help="Max token length (<= 512 for BERT)")
):
    typer.echo(f"FinBERT starting")

    result = predict_sentiment(
        year=year,
        batch_size=batch_size,
        context_window=context_window,
        max_tokens=max_tokens
    )
    
    typer.echo(f"Prediction successful. Results saved to: {result['output_path']}")
    typer.echo(f"Used device: {result['device']}")

# Run the app
if __name__ == "__main__":
    app()
