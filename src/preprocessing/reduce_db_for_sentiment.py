'''
Function to subset the database for sentiment prediction.
'''

import sqlite3
import pandas as pd
from pathlib import Path

# Custom filter for sentiment-relevant sections
from src.preprocessing.section_filtering import filter_article_for_sentiment

def subset_database(year):
    """
    Load WSJ articles for a given year, remove duplicates, filter by relevant sections,
    and save a reduced version of the database.

    Parameters
    ----------
    year : int
        Year of the source database (e.g., 2024 for 'articlesWSJ_clean_2024.db').

    This script expects:
    - An existing database in: data/processed/articles/articlesWSJ_clean_<year>.db
    - Tables: 'article' (with article content), 'articles_index' (with metadata).
    """

    # Define project root relative to script
    root = Path(__file__).resolve().parents[2]

    # Define source and target database paths
    articles_path = root / "data" / "processed" / "articles"
    source_path = articles_path / f"articlesWSJ_clean_{year}.db"
    target_path = articles_path / f"articlesWSJ_clean_reduced_{year}.db"

    # Load data from database
    conn = sqlite3.connect(source_path)
    df = pd.read_sql_query("SELECT * FROM article;", conn)
    df_index = pd.read_sql_query("SELECT * FROM articles_index;", conn)
    conn.close()

    # Remove articles with duplicate text
    duplicates = df[df.duplicated(subset=["corpus"], keep=False)]
    df = df.drop_duplicates(subset=["corpus"], keep="first")

    # Apply custom filtering based on section/topic/title
    df_filtered = filter_article_for_sentiment(df=df, df_index=df_index)

    # Save filtered DataFrame to new SQLite database
    conn_out = sqlite3.connect(target_path)
    df_filtered.to_sql("article", conn_out, index=False, if_exists="replace")
    conn_out.close()

    return {
        "n_articles" : str(len(df_filtered)),
        "db_path" : str(target_path)
    }
    

