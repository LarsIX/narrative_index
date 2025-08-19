# -*- coding: utf-8 -*-
"""
Compute minima and maxima for AINI variants.

Merges 4 AINI datasets (w0, w1, w2, custom) by date,
finds per-variable min/max dates, counts extrema occurrences,
and returns four DataFrames:
    merged, tidy, pivot, extrema_clean
"""

import re
import pandas as pd


def compute_aini_extrema(
    aini_w0: pd.DataFrame,
    aini_w1: pd.DataFrame,
    aini_w2: pd.DataFrame,
    aini_custom: pd.DataFrame,
    *,
    date_col: str = "date",
    exclude_substr=("growth",),
    date_fmt: str = "%d.%m.%Y",
):
    """Return (merged, tidy, pivot, extrema_clean) DataFrames."""

    def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        return df.sort_values(date_col)

    def _suffix_cols(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        rename_map = {c: f"{c}_{suffix}" for c in df.columns if c != date_col}
        return df.rename(columns=rename_map)

    def _should_exclude(col: str) -> bool:
        return any(sub in col for sub in exclude_substr)

    # --- Step 1: normalize & suffix ---
    w0 = _suffix_cols(_ensure_datetime(aini_w0), "w0")
    w1 = _suffix_cols(_ensure_datetime(aini_w1), "w1")
    w2 = _suffix_cols(_ensure_datetime(aini_w2), "w2")
    wc = _suffix_cols(_ensure_datetime(aini_custom), "custom")

    # --- Step 2: merge ---
    merged = w0.merge(w1, on=date_col).merge(w2, on=date_col).merge(wc, on=date_col)

    # --- Step 3: compute tidy extrema ---
    rows = []
    for col in merged.columns:
        if col == date_col or _should_exclude(col):
            continue
        s = merged[col]
        if s.notna().sum() == 0:
            continue
        i_max = s.idxmax()
        i_min = s.idxmin()
        rows.append({"variable": col, "type": "max", "date": merged.loc[i_max, date_col], "value": s.loc[i_max]})
        rows.append({"variable": col, "type": "min", "date": merged.loc[i_min, date_col], "value": s.loc[i_min]})

    tidy = pd.DataFrame(rows)
    if tidy.empty:
        return merged, tidy, pd.DataFrame(), pd.DataFrame()

    tidy["type"] = tidy["type"].astype(pd.CategoricalDtype(categories=["min", "max"], ordered=True))

    # --- Step 4: pivot table ---
    pivot = (
        tidy.pivot_table(
            index="variable",
            columns="type",
            values=["date", "value"],
            aggfunc="first"
        )
        .sort_index(axis=1, level=0)
    )

    # --- Step 5: extrema counts ---
    counts_by_date = (
        tidy.groupby(["type", "date"])
            .size()
            .reset_index(name="count")
            .sort_values(["type", "count"], ascending=[True, False])
    )

    # attach variable lists
    date_vars = (
        tidy.groupby(["type", "date"])["variable"]
            .apply(list)
            .reset_index()
            .rename(columns={"variable": "measure"})
    )
    extrema = counts_by_date.merge(date_vars, on=["type", "date"], how="left")

    # --- Step 6: clean formatting ---
    extrema_clean = extrema.copy()
    extrema_clean["date"] = pd.to_datetime(extrema_clean["date"]).dt.strftime(date_fmt)
    extrema_clean["type"] = extrema_clean["type"].replace({"min": "minimum", "max": "maximum"})
    extrema_clean.rename(columns={"count": "n measures"}, inplace=True)
    extrema_clean["measure"] = extrema_clean["measure"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")
    extrema_clean = extrema_clean.sort_values(
        by=["type", "n measures"], ascending=[True, False], kind="stable"
    ).reset_index(drop=True)

    return merged, tidy, pivot, extrema_clean
