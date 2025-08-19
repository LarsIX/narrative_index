# article_reader.py
# -*- coding: utf-8 -*-
"""
Interactive article reader for Jupyter notebooks or console.

Usage in a notebook:
--------------------
from read_articles import read
read(df)

Controls:
- n : show next article
- p : show previous article
- s : stop reading
"""

import pandas as pd
import textwrap

def _format_text(text: str, width: int = 100) -> str:
    """Wrap text nicely for readability."""
    return "\n".join(textwrap.wrap(str(text), width=width))

def read(df: pd.DataFrame, wrap_width: int = 100):
    """
    Open an interactive loop to read articles one by one.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the articles.
    wrap_width : int, optional
        Line width for wrapping text, by default 100
    """
    # --- Required base columns ---
    required = ["title", "date", "section", "cleaned_corpus", "hype_score"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # Ensure optional hype columns exist
    hype_cols = ["hype_score", "hype_score_w0", "hype_score_w1", "hype_score_w2"]
    for col in hype_cols:
        if col not in df.columns:
            df[col] = 0

    # --- Cast to int ---
    for col in hype_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # --- Filter: keep rows where ANY hype col != 0 ---
    mask = (df[hype_cols] != 0).any(axis=1)
    df = df.loc[mask].reset_index(drop=True)

    n_articles = len(df)
    if n_articles == 0:
        print("No articles to display (all hype scores are 0).")
        return

    i = 0

    # --- Interactive loop ---
    while True:
        row = df.iloc[i]

        print("=" * 80)
        print(f"Title   : {row['title']}")
        print(f"Date    : {row['date']}")
        print(f"Section : {row['section']}")
        print(
            "Scores  : "
            f"hype_score={row['hype_score']}, "
            f"hype_score_w0={row['hype_score_w0']}, "
            f"hype_score_w1={row['hype_score_w1']}, "
            f"hype_score_w2={row['hype_score_w2']}"
        )
        print("-" * 80)
        print(_format_text(row["cleaned_corpus"], wrap_width))
        print("=" * 80)
        print(f"Article {i+1} of {n_articles}")

        cmd = input("\nEnter command (n=next, p=previous, s=stop): ").strip().lower()

        if cmd == "n":
            if i + 1 < n_articles:
                i += 1
            else:
                print("Already at last article.")
        elif cmd == "p":
            if i > 0:
                i -= 1
            else:
                print("Already at first article.")
        elif cmd == "s":
            print("Stopping reader.")
            break
        else:
            print("Invalid command. Use n, p, or s.")
