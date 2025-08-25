# -*- coding: utf-8 -*-
"""
Compute minima and maxima for selected AINI variants (ISO dates).

Merges 4 AINI datasets (w0, w1, w2, custom) by date,
optionally drops specified noisy dates (ISO yyyy-mm-dd), finds per-variable min/max dates
for variables whose names START WITH one of:
    normalized_AINI, EMA_02, EMA_08
counts extrema occurrences, and returns four DataFrames:
    merged, tidy, pivot, extrema_clean
"""

import pandas as pd


def compute_aini_extrema(
    aini_w0: pd.DataFrame,
    aini_w1: pd.DataFrame,
    aini_w2: pd.DataFrame,
    aini_custom: pd.DataFrame,
    *,
    date_col: str = "date",
    include_prefixes=("normalized_AINI", "EMA_02", "EMA_08"),
    date_fmt: str = "%Y-%m-%d",  # ISO output
    # Noisy dates to drop BEFORE extrema (ISO yyyy-mm-dd)
    exclude_dates=["2023-07-15","2023-06-30"], # IMPORTANT: parse as list
):
    """Return (merged, tidy, pivot, extrema_clean) DataFrames."""

    def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        return df.sort_values(date_col)

    def _suffix_cols(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        rename_map = {c: f"{c}_{suffix}" for c in df.columns if c != date_col}
        return df.rename(columns=rename_map)

    def _should_include(col: str) -> bool:
        # Keep only variables whose base name starts with one of the allowed prefixes.
        # Columns look like "<basename>_<suffix>" e.g., "normalized_AINI_w0"
        if col == date_col:
            return False
        base = col.rsplit("_", 1)[0]  # strip trailing suffix like w0/w1/w2/custom
        return any(base.startswith(pref) for pref in include_prefixes)

    def _to_datetime_set_iso(dates) -> set:
        """Strictly parse ISO yyyy-mm-dd strings; ignore invalids."""
        out = set()
        for d in dates or []:
            if isinstance(d, str):
                dt = pd.to_datetime(d, format="%Y-%m-%d", errors="coerce")
            else:
                dt = pd.to_datetime(d, errors="coerce")
            if pd.notna(dt):
                out.add(pd.to_datetime(dt).normalize())
        return out

    # --- Step 1: normalize & suffix ---
    w0 = _suffix_cols(_ensure_datetime(aini_w0), "w0")
    w1 = _suffix_cols(_ensure_datetime(aini_w1), "w1")
    w2 = _suffix_cols(_ensure_datetime(aini_w2), "w2")
    wc = _suffix_cols(_ensure_datetime(aini_custom), "custom")

    # --- Step 2: merge (inner to keep common dates across all windows) ---
    merged = w0.merge(w1, on=date_col).merge(w2, on=date_col).merge(wc, on=date_col)

    # --- Step 2.5: drop noisy ISO dates BEFORE extrema ---
    if exclude_dates:
        drop_set = _to_datetime_set_iso(exclude_dates)
        merged = merged[~merged[date_col].dt.normalize().isin(drop_set)]

    # --- Step 3: compute tidy extrema (ONLY selected prefixes) ---
    rows = []
    for col in merged.columns:
        if not _should_include(col):
            continue
        s = merged[col]
        if s.notna().sum() == 0:
            continue
        i_max = s.idxmax()
        i_min = s.idxmin()
        rows.append({
            "variable": col,
            "type": "max",
            "date": merged.loc[i_max, date_col],
            "value": s.loc[i_max],
        })
        rows.append({
            "variable": col,
            "type": "min",
            "date": merged.loc[i_min, date_col],
            "value": s.loc[i_min],
        })

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

    # --- Step 6: clean formatting (ISO) ---
    extrema_clean = extrema.copy()
    extrema_clean["date"] = pd.to_datetime(extrema_clean["date"]).dt.strftime(date_fmt)
    extrema_clean["type"] = extrema_clean["type"].replace({"min": "minimum", "max": "maximum"})
    extrema_clean.rename(columns={"count": "n measures"}, inplace=True)
    extrema_clean["measure"] = extrema_clean["measure"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")
    extrema_clean = extrema_clean.sort_values(
        by=["type", "n measures"], ascending=[True, False], kind="stable"
    ).reset_index(drop=True)

    return merged, tidy, pivot, extrema_clean
