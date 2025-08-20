# scripts/run_predict_FinBERT_sentiment.py
import typer
from pathlib import Path
from typing import Optional
import sys
src = Path(__file__).resolve().parents[1] 
sys.path.append(str(src))
app = typer.Typer(help="Run FinBERT sentiment predictions on WSJ AI-relevant articles.")
from modelling.predict_AINI_FinBERT_prelabeled_fin import predict_sentiment

app = typer.Typer(help="Run FinBERT sentiment predictions on WSJ AI-relevant articles.")

@app.command()
def run(
    year: Optional[int] = typer.Option(
        None,
        help="Year to run prediction for. If omitted, runs for all predefined years."
    ),
    batch_size: int = typer.Option(
        16,
        help="Batch size for inference."
    ),
    model_name: str = typer.Option(
        "ProsusAI/finbert",
        help="HuggingFace model name."
    )
):
    """
    Run FinBERT sentiment prediction for AI-relevant WSJ articles.

    Parameters
    ----------
    year : int, optional
        Specific year to run sentiment predictions for.
        If ``None``, predictions will be run for all predefined years (2023, 2024, 2025).
    batch_size : int, default=16
        Number of articles to process in parallel during inference.
    model_name : str, default="ProsusAI/finbert"
        HuggingFace model identifier for the FinBERT model.

    Notes
    -----
    - Results are saved as CSV files in the `data/processed/variables/` directory.

    Examples
    --------
    Run for a specific year:
    python scripts/run_predict_AINI_FinBERT_prelabeled_fin.py run year 2024

    Run for all predefined years:   
    python run_predict_AINI_FinBERT_prelabeled_fin.py run
    """
    if year is None:
        years = [2023, 2024, 2025]
    else:
        years = [year]

    for y in years:
        typer.echo(f"Running sentiment prediction for {y}...")
        result = predict_sentiment(year=y, batch_size=batch_size, model_name=model_name)
        typer.echo(f"Saved results to: {result['output_path']} (device: {result['device']})")

if __name__ == "__main__":
    app()
