"""
CLI tool to predict investor sentiment using FinBERT.

Usage:
    python run_predict_investor_sentiment.py run --year 2024 --batch-size 32
"""


import typer
from src.modelling.predict_investor_sentiment import predict_sentiment

app = typer.Typer()

@app.command()
def run(
    year: int = typer.Argument(..., help="Enter year to start inference"),
    batch_size: int = typer.Option(16, help="Batch size for model inference")
):
    """
    Run FinBERT sentiment inference on WSJ articles for a given year.

    - Loads cleaned articles from SQLite database.
    - Combines title and corpus for prediction.
    - Applies pretrained FinBERT model to classify sentiment (Positive, Negative, Neutral).
    - Saves results to CSV in `data/processed/variables/`.

    Uses GPU if available.
    """  
    typer.echo("🚀 Starting sentiment prediction...")

    result = predict_sentiment(year, batch_size)
    typer.echo(f"🖥️ Used device: {result['device']}")
    typer.echo(f"✅ Saved sentiment predictions to: {result['output_path']}")

if __name__ == "__main__":
    app()
