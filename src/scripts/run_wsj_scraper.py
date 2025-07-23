"""
CLI tool to scrape full WSJ articles from previously stored links in a local SQLite database.

Usage:
    python run_wsj_scraper.py scrape --db-path "path/to/db.sqlite" --year 2024 --limit 30

Note:
    Requires a Chrome browser running in remote-debugging mode (e.g., `chrome.exe --remote-debugging-port=9222`)
"""

import typer
from pathlib import Path
from src.fetch_data.wsj_archive_scraper import WSJScraper

app = typer.Typer()

@app.command()
def scrape(
    db_path: Path = typer.Option(..., exists=True, help="Path to SQLite database containing WSJ article links."),
    year: int = typer.Option(None, help="Optional year filter for article date."),
    month: int = typer.Option(None, help="Optional month filter for article date."),
    day: int = typer.Option(None, help="Optional day filter for article date."),
    limit: int = typer.Option(None, help="Maximum number of articles to scrape."),
):
    """
    Scrape WSJ article content by date or in batch from a local database.
    """
    typer.echo(f"🔍 Starting WSJ scraping with limit={limit} and date={year}-{month}-{day}")
    
    scraper = WSJScraper(str(db_path))
    articles = scraper.get_article_links(limit=limit, year=year, month=month, day=day)

    if not articles:
        typer.echo("⚠️ No unscreened articles found for the given filters.")
        scraper.close()
        raise typer.Exit(code=0)

    for article_id, url in articles:
        scraper.scrape_article(article_id, url)

    scraper.close()
    typer.echo("✅ Finished scraping WSJ articles.")

if __name__ == "__main__":
    app()
