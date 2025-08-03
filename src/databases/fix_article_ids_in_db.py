import sqlite3
import pandas as pd
from pathlib import Path

def fix_article_ids_in_db(years):
    """
    Fixes article_id in WSJ SQLite databases by prefixing the year.

    Parameters
    ----------
    years : list of int
        Years to process, e.g., [2023, 2024, 2025]
    """

    root = Path(__file__).resolve().parents[2]
    db_dir = root / "data" / "processed" / "articles"

    for year in years:
        db_file = db_dir / f"articlesWSJ_clean_{year}.db"
        if not db_file.exists():
            print(f"Skipping {year}: File not found: {db_file}")
            continue

        print(f"Fixing article_id in: {db_file.name}")

        conn = sqlite3.connect(db_file)
        df = pd.read_sql("SELECT * FROM article", conn)

        # Prefix article_id with year (as int to avoid double-prefixing if already done)
        if df["article_id"].astype(str).str.startswith(str(year)).all():
            print(f"Already fixed, Skipping year {year}")
            conn.close()
            continue

        df["article_id"] = df["article_id"].astype(str).apply(lambda x: f"{year}{x}")

        # Overwrite old table
        df.to_sql("article", conn, if_exists="replace", index=False)
        conn.close()

        print(f"Done fixing year {year}")

