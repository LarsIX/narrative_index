"""
Function tool to clean and filter WSJ article databases by year.
Applies section filtering, exclusion-based filtering, and advanced corpus cleaning.
"""

import os
import re
import sqlite3
import pandas as pd
import typer
from src.article_preprocessing.corpus_cleaner import clean_article_corpus
from src.article_preprocessing.section_filtering import filter_article_by_section


def clean_database(year: int = typer.Option(..., help="Target year (e.g., 2023, 2024, 2025)")):
    """
    Cleans and processes the raw WSJ article database for a given year:
    - Loads article and index data from SQLite
    - Filters irrelevant sections and known noisy articles
    - Applies custom corpus cleaning
    - Stores the cleaned dataset to a new SQLite database

    Parameters:
    - year: Year of scraped articles, e.g. 2023 / 2024 / 2025.

    Returns: 
    - articlesWSJ_clean_{year}.db: SQLite Database with cleaned corpora.
    """

    # Paths and filenames
    repo_root = os.getcwd()
    raw_base = os.path.join(repo_root, "data", "raw", "articles")
    processed_base = os.path.join(repo_root, "data", "processed", "articles")

    db_filenames = {
        2023: "articlesWSJ_2023.db",
        2024: "articlesWSJ_2024.db",
        2025: "articlesWSJ_2025.db",
    }

    if year not in db_filenames:
        raise ValueError(f"Unsupported year: {year}. Supported years: {list(db_filenames.keys())}")

    db_path = os.path.join(raw_base, db_filenames[year])
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at: {db_path}")

    # Load article data
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM article;", conn)
    df_index = pd.read_sql_query("SELECT * FROM articles_index;", conn)
    conn.close()

    # Drop duplicates
    duplicates = df[df.duplicated(subset=['corpus'], keep=False)]
    df = df.drop_duplicates(subset=['corpus'])

    # Add parsed date column
    df_index['date'] = pd.to_datetime(
        df_index[['year', 'month', 'day']].astype(str).agg('-'.join, axis=1),
        format='%Y-%m-%d'
    )
    print("📅 Date column added to index")

    # Filter by relevant sections 
    df = filter_article_by_section(df=df, df_index=df_index)

    # Exclude known noisy articles based on phrases
    phrases = [
        "MIT says it no longer stands behind AI research paper",
        "how to use generative ai tools for everyday tasks",
        "ai can be a force for deregulation",
        "ai helped heal my chronic pain",
        "massive ai chip deal",
    ]
    pattern = "|".join([re.escape(p) for p in phrases])
    mask = df["corpus"].str.contains(pattern, case=False, na=False)

    removed_articles = df[mask]
    df = df[~mask]

    # Save removed articles to review log
    removed_dir = os.path.join(repo_root, "data", "logs", "filtered_out")
    os.makedirs(removed_dir, exist_ok=True)
    removed_csv_path = os.path.join(removed_dir, f"removed_articles_{year}.csv")
    removed_articles.to_csv(removed_csv_path, index=False)

    #  Clean corpus text
    df['cleaned_corpus'] = df['corpus'].apply(clean_article_corpus)

    # === Write cleaned data to new database ===
    clean_filenames = {
        2023: "articlesWSJ_clean_2023.db",
        2024: "articlesWSJ_clean_2024.db",
        2025: "articlesWSJ_clean_2025.db"
    }
    clean_db_path = os.path.join(processed_base, clean_filenames[year])
    os.makedirs(os.path.dirname(clean_db_path), exist_ok=True)

    with sqlite3.connect(clean_db_path) as clean_conn:
        df.to_sql("article", clean_conn, if_exists="replace", index=False)
        df_index.to_sql("articles_index", clean_conn, if_exists="replace", index=False)

    return {
        "n_dropped_articles" = len(removed_articles)
        "db_path" = str(processed_base)
        "removed_path" = str(removed_csv_path)
    }

