'''
CLI script to subset the database for sentiment prediction.
Usage:
    python run_reduce_db_for_sentiment.py --year 2023
'''

import typer
from preprocessing.reduce_db_for_sentiment import subset_database

# Initialize CLI
app = typer.Typer()

@app.command()
def run(year: int = typer.Option(..., help="Insert year to subset.")):

    results = subset_database(year = year)

    typer.echo(f"💾 Saved {results['n_articles']} articles to {results['db_path']}")

# CLI entry point
if __name__ == "__main__":
    app()
