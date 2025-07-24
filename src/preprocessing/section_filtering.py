import pandas as pd

def filter_article_by_section(df: pd.DataFrame, df_index: pd.DataFrame) -> pd.DataFrame:
    """
    Merges article DataFrame with metadata (date, link) from the index,
    extracts section from URL, and filters out irrelevant sections.

    Args:
        df (pd.DataFrame): Raw article data, must contain `index_id`, `corpus`.
        df_index (pd.DataFrame): Article index with at least `id`, `date`, and `link`.

    Returns:
        pd.DataFrame: Merged and filtered article DataFrame.
    """
    # Drop potential conflicting columns if they exist
    df = df.drop(columns=["date", "link", "id"], errors="ignore")

    # Merge on index ID
    df = df.merge(
        df_index[["id", "date", "link"]],
        left_on="index_id",
        right_on="id",
        how="left"
    )

    # Extract 'section' from WSJ URL
    df["section"] = df["link"].astype(str).str.extract(r"wsj\.com/([^/]+)/")

    # Define irrelevant sections to filter out
    irrelevant_sections = {
        "health", "arts-culture", "lifestyle", "real-estate", "sports",
        "livecoverage", "personal-finance", "video", "science", "style", "articles"
    }

    # Filter out rows with irrelevant sections
    df = df[~df["section"].isin(irrelevant_sections)].copy()

    # Log some verification details
    print("Finished merge and filter")
    print("Remaining articles:", len(df))
    print("Unique sections:", sorted(set(df['section'].dropna())))
    print("Missing dates:", df['date'].isna().sum())
    print("Missing corpora:", df['corpus'].isna().sum())
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")

    return df

def filter_article_for_sentiment(df: pd.DataFrame, df_index: pd.DataFrame) -> pd.DataFrame:
    """
    Merges article DataFrame with metadata (date, link) from the index,
    extracts section from URL, and filters out irrelevant sections. More restrictive than filter_article_by_section.

    Args:
        df (pd.DataFrame): Raw article data, must contain `index_id`, `corpus`.
        df_index (pd.DataFrame): Article index with at least `id`, `date`, and `link`.

    Returns:
        pd.DataFrame: Merged and filtered article DataFrame.
    """
    # Drop potential conflicting columns if they exist
    df = df.drop(columns=["date", "link", "id"], errors="ignore")

    # Merge on index ID
    df = df.merge(
        df_index[["id", "date", "link"]],
        left_on="index_id",
        right_on="id",
        how="left"
    )

    # Extract 'section' from WSJ URL
    df["section"] = df["link"].astype(str).str.extract(r"wsj\.com/([^/]+)/")

    # Define irrelevant sections to filter out
    relevant_sections = {
        'business', 'economy', 'finance'
    }

    # Filter out rows with irrelevant sections
    df = df[df["section"].isin(relevant_sections)].copy()

    # Log some verification details
    print("Finished merge and filter")
    print("Remaining articles:", len(df))
    print("Unique sections:", sorted(set(df['section'].dropna())))
    print("Missing dates:", df['date'].isna().sum())
    print("Missing corpora:", df['corpus'].isna().sum())
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")

    return df
