"""
CLI tool to run WSJ crawler and load metadata.

Usage:
    python run_wsj_crawler.py run --year 2024 --batch-size 32
"""

import os
import typer
from src.fetch_data.wsj_archive_crawler import search_year

app = typer.Typer()

@app.command()
def run(
    year: int = typer.Option(..., help="Year to crawl (e.g., 2024)"),
    wait: int = typer.Option(5, help="Delay between page requests in seconds")
):
    """
    Crawl WSJ article metadata for a given year and store it in a yearly SQLite database.
    """
    db_path = os.path.join("data", "raw", "articles", f"articlesWSJ_{year}.db")
    typer.echo(f"Starting crawl for {year} with {wait}s delay — DB: {db_path}")
    search_year(year=year, wait=wait, db_path=db_path)

if __name__ == "__main__":
    app()
