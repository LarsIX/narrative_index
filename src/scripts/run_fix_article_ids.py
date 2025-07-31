"""
CLI app to fix non-unique article_id values in WSJ article databases by prefixing them with the corresponding year.

This prevents ID collisions across multiple years (e.g., 2023 and 2024 both having article_id = 1234),
which can cause data integrity issues when merging or analyzing articles.

Each article_id will be transformed from:
    1234 → 20231234 (if from 2023)
    5678 → 20245678 (if from 2024)

Usage:
    python run_fix_article_ids.py --years 2023,2024,2025
"""

import typer
import sys
from pathlib import Path

# Resolve project root and add src/databases to path
project_root = Path(__file__).resolve().parents[2]
db_module_path = project_root / "src" / "databases"
sys.path.append(str(db_module_path))

from fix_article_ids_in_db import fix_article_ids_in_db

app = typer.Typer()

@app.command()
def main(
    years: str = typer.Option(..., help="Comma-separated list of years, e.g. '2023,2024,2025'")
):
    """
    Prefix article_id fields in WSJ databases with their year to make them globally unique.
    """
    year_list = [int(y.strip()) for y in years.split(",")]
    fix_article_ids_in_db(year_list)

if __name__ == "__main__":
    app()
