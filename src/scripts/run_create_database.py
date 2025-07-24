"""
CLI-App to create SQLite Database.

Usage:
    python run_create_database.py --year 2024
"""

from pathlib import Path
import sys
import typer

# Append scripts folder to sys.path
root = Path(__file__).parent.parent
scripts_path = root / "databases"
sys.path.append(str(scripts_path.resolve()))

# Load function
import create_database

# Create CLI app
app = typer.Typer()

@app.command()
def run(
    year: int = typer.Option(..., help="Year of fetching & scraping"),
    folder: str = typer.Option(None, help="Folder where DB is created (default: data/raw/articles)")
):
    """
    Create a new WSJ article database.
    """
    typer.echo("Creating Database...")

    results = create_database.create_db(year=year, folder=folder)

    typer.echo(f"New database for {results['year']} created at: {results['output_path']}")

# CLI entry point
if __name__ == "__main__":
    app()
