# id_fixers.py
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, Tuple, List
import pandas as pd
import numpy as np


# --------------------------- #
# DB: prefix by file's year   #
# --------------------------- #
def fix_article_ids_in_db(years: Iterable[int]) -> None:
    """
    Prefix article_id with the file's year for WSJ SQLite DBs.
    Matches your original logic exactly .

    Parameters
    ----------
    years : iterable of int, e.g. [2023, 2024, 2025]
    """
    root = Path(__file__).resolve().parents[2]
    db_dir = root / "data" / "processed" / "articles"

    for year in years:
        db_file = db_dir / f"articlesWSJ_clean_{year}.db"
        if not db_file.exists():
            print(f"Skipping {year}: File not found: {db_file}")
            continue

        print(f"Fixing article_id in: {db_file.name}")
        with sqlite3.connect(db_file) as conn:
            df = pd.read_sql("SELECT * FROM article", conn)

            # If already prefixed with this exact year, skip
            if df["article_id"].astype(str).str.startswith(str(year)).all():
                print(f"Already fixed, skipping year {year}")
                continue

            df["article_id"] = (
                df["article_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
            ).apply(lambda x: f"{year}{x}")

            # Overwrite table in place
            df.to_sql("article", conn, if_exists="replace", index=False)

        print(f"Done fixing year {year}")


# --------------------------- #
# CSV helpers                 #
# --------------------------- #
# --------------------------- #
# CSV helpers                 #
# --------------------------- #
def _parse_dates_iso(series: pd.Series) -> pd.Series:
    """Parse dates, prefer strict ISO (YYYY-MM-DD); one lenient fallback."""
    dt = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
    if dt.isna().any():
        dt2 = pd.to_datetime(series, errors="coerce")
        dt = dt.fillna(dt2)
    return dt


def _prefix_year_to_article_id(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    id_col: str = "article_id",
    strict: bool = True,
) -> pd.DataFrame:
    """
    ALWAYS prepend the year extracted from `date_col` to `article_id`.

    - No checks for existing prefixes.
    - No correction of "wrong" prefixes.
    - If `strict` and any date is invalid -> raise.
    - If not strict, rows with invalid dates keep their original IDs.

    Returns a NEW DataFrame.
    """
    if date_col not in df.columns or id_col not in df.columns:
        missing = [c for c in (date_col, id_col) if c not in df.columns]
        raise KeyError(f"Missing required columns: {missing}")

    out = df.copy()
    out[date_col] = _parse_dates_iso(out[date_col])

    if strict and out[date_col].isna().any():
        bad = out[out[date_col].isna()][[id_col, date_col]].head(10)
        raise ValueError(
            f"Invalid dates after parsing in '{date_col}'. Examples (first 10):\n"
            f"{bad.to_string(index=False)}"
        )

    # clean id as string, strip, and drop trailing ".0"
    id_str = out[id_col].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    # compute year string; only use where date is valid
    year_str = out[date_col].dt.year.astype("Int64").astype(str)
    valid = out[date_col].notna()

    # ALWAYS prepend year for valid dates; leave as-is if invalid and strict=False
    id_new = np.where(valid, year_str + id_str, id_str)
    out[id_col] = id_new

    return out


def _read_csvs(var_path: Path, pattern_fn, years: Iterable[int]) -> List[Path]:
    """Return file paths for the given pattern & years (no read yet)."""
    return [var_path / pattern_fn(y) for y in years]


def _fix_csv_file_inplace(
    csv_path: Path,
    *,
    date_col: str = "date",
    id_col: str = "article_id",
    strict_dates: bool = True,
) -> None:
    """Load CSV, ALWAYS prepend year from date to article_id, and write back to SAME file."""
    df = pd.read_csv(csv_path)
    df_fixed = _prefix_year_to_article_id(
        df, date_col=date_col, id_col=id_col, strict=strict_dates
    )
    df_fixed.to_csv(csv_path, index=False)


# -------------------------------------- #
# CSV: load + fix IDs + (optional) write #
# -------------------------------------- #
def load_aini_and_fix_ids(
    var_path: Path,
    years: Tuple[int, ...] = (2023, 2024, 2025),
    date_col: str = "date",
    id_col: str = "article_id",
    strict_dates: bool = True,
    write_back: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Load all AINI CSVs (custom, w0, w1, w2), ALWAYS prepend YYYY to `article_id`
    based on `date`, and return a dict with DataFrames + list aini_dfs (c_df, w0_df, w1_df, w2_df).
    If write_back=True, also overwrites the original CSV files in place.
    """
    # define file patterns
    patterns = {
        "c":  lambda y: f"FinBERT_AINI_prediction_{y}_on_binary.csv",
        "w0": lambda y: f"FinBERT_AINI_prediction_{y}_windsize_0.csv",
        "w1": lambda y: f"FinBERT_AINI_prediction_{y}_windsize_1.csv",
        "w2": lambda y: f"FinBERT_AINI_prediction_{y}_windsize_2.csv",
    }

    # optionally modify CSVs in place first
    if write_back:
        for _, patt in patterns.items():
            for p in _read_csvs(var_path, patt, years):
                if not p.exists():
                    raise FileNotFoundError(f"Missing CSV: {p}")
                _fix_csv_file_inplace(p, date_col=date_col, id_col=id_col, strict_dates=strict_dates)

    # load (post-fix if write_back=True)
    def _concat(pattern_fn):
        dfs = []
        for y in years:
            f = var_path / pattern_fn(y)
            if not f.exists():
                raise FileNotFoundError(f"Missing CSV: {f}")
            dfs.append(pd.read_csv(f))
        return pd.concat(dfs, ignore_index=True)

    c_df  = _concat(patterns["c"])
    w0_df = _concat(patterns["w0"])
    w1_df = _concat(patterns["w1"])
    w2_df = _concat(patterns["w2"])

    # if not written back, fix in-memory now (ALWAYS prepend)
    if not write_back:
        c_df  = _prefix_year_to_article_id(c_df,  date_col=date_col, id_col=id_col, strict=strict_dates)
        w0_df = _prefix_year_to_article_id(w0_df, date_col=date_col, id_col=id_col, strict=strict_dates)
        w1_df = _prefix_year_to_article_id(w1_df, date_col=date_col, id_col=id_col, strict=strict_dates)
        w2_df = _prefix_year_to_article_id(w2_df, date_col=date_col, id_col=id_col, strict=strict_dates)

    aini_dfs = [c_df, w0_df, w1_df, w2_df]
    return {"c_df": c_df, "w0_df": w0_df, "w1_df": w1_df, "w2_df": w2_df, "aini_dfs": aini_dfs}


def fix_aini_csv_ids_inplace(
    var_path: Path,
    years: Tuple[int, ...] = (2023, 2024, 2025),
    date_col: str = "date",
    id_col: str = "article_id",
    strict_dates: bool = True,
) -> None:
    """
    Convenience wrapper: directly rewrite ALL four AINI CSV sets in place
    with IDs prefixed by the year from `date`.
    """
    patterns = [
        lambda y: f"FinBERT_AINI_prediction_{y}_on_binary.csv",
        lambda y: f"FinBERT_AINI_prediction_{y}_windsize_0.csv",
        lambda y: f"FinBERT_AINI_prediction_{y}_windsize_1.csv",
        lambda y: f"FinBERT_AINI_prediction_{y}_windsize_2.csv",
    ]
    for patt in patterns:
        for p in _read_csvs(var_path, patt, years):
            if not p.exists():
                raise FileNotFoundError(f"Missing CSV: {p}")
            _fix_csv_file_inplace(p, date_col=date_col, id_col=id_col, strict_dates=strict_dates)