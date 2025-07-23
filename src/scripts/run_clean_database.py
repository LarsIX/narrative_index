'''
CLI tool to clean database by removing duplicates, NAs & redunant topics.
Usage:
    -python run_clean_database.py --year 2023
'''

import typer
from preprocessing.clean_database import clean_database

# Define Typer app
app = typer.Typer(help="Clean and filter raw WSJ article databases by year.")

@app.command()
def run(year: int = typer.Option(..., help="Target year (e.g., 2023, 2024, 2025)")): 
    typer.echo("🚀 Cleaning started...")

    results = clean_database(year=year)

    typer.echo(f"✅ Cleaned articles saved to: {results['db_path']}")
    typer.echo(f"🧹 Removed {results['n_dropped_articles']} noisy/duplicate articles")
    typer.echo(f"📁 Removed articles log saved to: {results['removed_path']}")

# CLI Entry Point
if __name__ == "__main__":
    app()
