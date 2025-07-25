"""
CLI tool to download stock market data.

Usage:
    python run_load_financial_data.py --start-date 2023-01-01 --end-date 2023-12-31
"""

from pathlib import Path
import typer
import sys

# append fetch_data path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# load fetching function
from fetch_data.load_financial_data import fetch_and_save_data

# initialize CLI app
app = typer.Typer()

@app.command()
def run(
    start_date: str = typer.Option(..., help="Start date in yyyy-mm-dd format"),
    end_date: str = typer.Option(..., help="End date in yyyy-mm-dd format"),
    output_dir: Path = typer.Option("data/raw/financial", help="Directory to save CSV files")
):
    """
    Downloads financial stock data for predefined tickers between the given dates.
    """
    save_dir = Path(__file__).resolve().parents[2] / output_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    file_name, saved_path = fetch_and_save_data(start_date=start_date, end_date=end_date, save_dir=save_dir)

    typer.echo(f"Saved merged financial data as: {saved_path / file_name}")

if __name__ == "__main__":
    app()