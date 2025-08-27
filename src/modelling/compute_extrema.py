# -*- coding: utf-8 -*-
"""
Compute minima and maxima for selected AINI variants (ISO dates).

Merges 4 AINI datasets (w0, w1, w2, custom) by date,
optionally drops specified noisy dates (ISO yyyy-mm-dd), finds per-variable min/max dates
for variables whose names START WITH one of:
    normalized_AINI, EMA_02, EMA_08
Counts extrema occurrences, and returns four DataFrames:
    merged, tidy, pivot, extrema_clean

Contract:
- `merged` includes zero-only rows (after exclude_dates), i.e., no zero-row filtering.
- Extrema (tidy/pivot/extrema_clean) are computed on a nonzero-only view.
"""

from typing import Iterable, Optional, Tuple
import pandas as pd


def compute_aini_extrema(
    aini_w0: pd.DataFrame,
    aini_w1: pd.DataFrame,
    aini_w2: pd.DataFrame,
    aini_custom: pd.DataFrame,
    *,
    date_col: str = "date",
    include_prefixes: Tuple[str, ...] = ("normalized_AINI", "EMA_02", "EMA_08"),
    date_fmt: str = "%Y-%m-%d",  # ISO output
    exclude_dates: Optional[Iterable] = None,  # noisy dates (ISO yyyy-mm-dd or datetime-like)
):
    """Return (merged, tidy, pivot, extrema_clean) DataFrames."""

    def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        return out.dropna(subset=[date_col]).sort_values(date_col)

    def _suffix_cols(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        rename_map = {c: f"{c}_{suffix}" for c in df.columns if c != date_col}
        return df.rename(columns=rename_map)

    def _should_include(col: str) -> bool:
        if col == date_col:
            return False
        base = col.rsplit("_", 1)[0]  # strip trailing suffix like w0/w1/w2/custom
        return any(base.startswith(pref) for pref in include_prefixes)

    def _to_datetime_set_iso(dates) -> set:
        """Parse ISO yyyy-mm-dd strings (or datetime-like), normalize to midnight."""
        out = set()
        for d in (dates or []):
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

    # Early out if any is empty
    if w0.empty or w1.empty or w2.empty or wc.empty:
        merged = pd.DataFrame(columns=[date_col])
        return merged, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # --- Step 2: merge (inner to keep common dates across all windows) ---
    merged = (
        w0.merge(w1, on=date_col, how="inner")
          .merge(w2, on=date_col, how="inner")
          .merge(wc, on=date_col, how="inner")
    )

    if merged.empty:
        print("[WARNING] No overlapping dates!")
        return merged, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # --- Step 2.1: drop noisy dates BEFORE extrema (applies to all outputs) ---
    if exclude_dates:
        drop_set = _to_datetime_set_iso(exclude_dates)
        merged = merged[~merged[date_col].dt.normalize().isin(drop_set)]

    if merged.empty:
        print("[WARNING] No overlapping dates after excluding noise!")
        return merged, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Keep a copy that preserves zero-only rows for the returned `merged`
    merged_all = merged.copy()

    # --- Build a filtered view ONLY for extrema selection (drop zero-only rows) ---
    value_cols = [c for c in merged_all.columns if c != date_col and _should_include(c)]
    if value_cols:
        row_sums = merged_all[value_cols].abs().sum(axis=1)
        merged_nonzero = merged_all[row_sums != 0]
    else:
        merged_nonzero = merged_all

    if merged_nonzero.empty:
        # Return merged_all (with zero-only dates), but no extrema
        print("[WARNING] No dates without non-zero AINI measures!")
        return merged_all, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # --- Step 3: compute tidy extrema USING merged_nonzero (not merged_all) ---
    # ensure numeric
    merged_nonzero[value_cols] = merged_nonzero[value_cols].apply(pd.to_numeric, errors="coerce")

    rows = []
    for col in value_cols:
            eps = 1e-12  # treat |value| <= eps as zero

            s = pd.to_numeric(merged_nonzero[col], errors="coerce")

            # mask out (near-)zeros for THIS measure only
            s_nz = s.where(s.abs() > eps)

            # if everything is zero/NaN, skip this measure entirely
            if s_nz.notna().sum() == 0:
                continue

            # tie-breaker: pick latest occurrence among the remaining (non-zero) dates
            vmax = s_nz.max(skipna=True)
            vmin = s_nz.min(skipna=True)
            i_max = s_nz[s_nz == vmax].index[-1]
            i_min = s_nz[s_nz == vmin].index[-1]

            rows.extend([
                {"variable": col, "type": "max", "date": merged_nonzero.loc[i_max, date_col], "value": float(vmax)},
                {"variable": col, "type": "min", "date": merged_nonzero.loc[i_min, date_col], "value": float(vmin)},
            ])


    tidy = pd.DataFrame(rows)
    if tidy.empty:
        return merged_all, tidy, pd.DataFrame(), pd.DataFrame()

    tidy["type"] = pd.Categorical(tidy["type"], categories=["min", "max"], ordered=True)

    # --- Step 4: pivot table (variable x [date,value] per type) ---
    pivot = (
        tidy.pivot_table(
            index="variable",
            columns="type",
            values=["date", "value"],
            aggfunc="first"
        )
        .sort_index(axis=1, level=0)
    )

    # --- Step 5: extrema counts + variable lists ---
    counts_by_date = (
        tidy.groupby(["type", "date"])
            .size()
            .reset_index(name="count")
            .sort_values(["type", "count"], ascending=[True, False], kind="stable")
    )

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
    extrema_clean["measure"] = extrema_clean["measure"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else ""
    )
    extrema_clean = extrema_clean.sort_values(
        by=["type", "n measures"], ascending=[True, False], kind="stable"
    ).reset_index(drop=True)

    # Return: merged_all keeps zero-only rows; extrema uses nonzero view
    return merged_all, tidy, pivot, extrema_clean
