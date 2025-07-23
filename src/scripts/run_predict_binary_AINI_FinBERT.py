"""
CLI tool to predict AINI values.

Usage:
    python run_predict_AINI.py run-bert --year 2024 --batch-size 16
"""

import typer
from src.modelling.laber_articles_trained_BERT import label_data

app = typer.Typer()

@app.command(name="run-bert")
def run_bert(
    year: int = typer.Option(..., help="Enter year to label"),
    batch_size: int = typer.Option(16, help="Enter batch size, default = 16")
):
    """
    Run FinBERT-based inference to label WSJ articles for AI-related narrative presence.
    """
    typer.echo("🚀 Script started")

    result = label_data(year=year, batch_size=batch_size)

    typer.echo(f"✅ Predictions saved to: {result['output_path']}")
    typer.echo(f"🔢 Total articles labeled: {result['n_labeled_articles']}")

if __name__ == "__main__":
    app()
