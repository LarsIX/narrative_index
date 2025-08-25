from __future__ import annotations
from typing import Sequence, List, Union
import numpy as np
import pandas as pd

Number = Union[int, float, np.number]


def _get_numeric_columns(df: pd.DataFrame, exclude: Sequence[str], numeric_only: bool) -> List[str]:
    """Return the list of candidate numeric columns after exclusions."""
    excluded = set(exclude or [])
    if numeric_only:
        cols = df.select_dtypes(include="number").columns
    else:
        cols = [c for c in df.columns if c not in excluded]
    return [c for c in cols if c not in excluded]


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Coerce to numeric where possible (non-convertibles become NaN)."""
    return pd.to_numeric(series, errors="coerce")


def _summarize_columns(
    df: pd.DataFrame,
    *,
    exclude: Sequence[str] = ("date",),
    numeric_only: bool = True,
) -> pd.DataFrame:
    """Helper: compute count, mean, std, min, median, max per numeric column."""
    cols = _get_numeric_columns(df, exclude, numeric_only)

    rows = []
    for col in cols:
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s) and not numeric_only:
            s = _coerce_numeric(s)
        if s.notna().sum() == 0:
            continue

        rows.append(
            {
                "variable": col,
                "count": int(s.notna().sum()),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)),
                "min": float(s.min()),
                "median": float(s.median()),
                "max": float(s.max()),
            }
        )

    return pd.DataFrame(rows).sort_values("variable").reset_index(drop=True)


def calculate_aini_statistics(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    exclude: Sequence[str] = ("date",),
    numeric_only: bool = True,
) -> pd.DataFrame:
    """
    Compute summary stats overall (Total) and per calendar year, based on `date_col`.

    Columns in output:
        ['scope', 'variable', 'count', 'mean', 'std', 'min', 'median', 'max']
    """
    # Total
    total_df = _summarize_columns(df, exclude=exclude, numeric_only=numeric_only)
    total_df.insert(0, "scope", "Total")

    # Per-year
    dts = pd.to_datetime(df[date_col], errors="coerce")
    df_year = df.copy()
    df_year["_year"] = dts.dt.year

    per_year_frames = []
    for year, sub in df_year.dropna(subset=["_year"]).groupby("_year", sort=True):
        year_summary = _summarize_columns(sub.drop(columns=["_year"]), exclude=exclude, numeric_only=numeric_only)
        if len(year_summary):
            year_summary.insert(0, "scope", int(year))
            per_year_frames.append(year_summary)

    # Combine
    out = pd.concat([total_df, *per_year_frames], ignore_index=True)
    out["__scope_sort"] = out["scope"].apply(lambda x: -1 if x == "Total" else int(x))
    out = out.sort_values(["__scope_sort", "variable"]).drop(columns="__scope_sort").reset_index(drop=True)
    return out
