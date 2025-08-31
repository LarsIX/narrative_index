# -*- coding: utf-8 -*-
"""
Export regression / Granger-causality results to a DIN A4–friendly LaTeX longtable.

Key behavior
------------
- Renames 'p_x' -> 'n_lags' (kept), positioned right after 'AINI_variant' if present.
- Drops columns: 'Direction', 'p_ret', 'N_obs', 'r2_u', 'N_boot', 'N_boot_valid'.
- Appends significance stars to: 'BH_corr_F_pval' and 'BH_corr_F_pval_HC3'.
- Uses LaTeX `longtable` (multi-page), sized to \\textwidth with very small font and tight spacing.
- Escapes LaTeX special characters via `escape=True` (no manual underscore hacks needed).
- Writes to: \\AI_narrative_index\\reports\\tables\\{output_filename}.tex

Stars:
  *** for p < 0.01, ** for p < 0.05, * for p < 0.10
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Dict, List, Tuple
import re
import numpy as np
import pandas as pd
# ----------------------------- configuration -------------------------------- #
# Columns to drop (exactly as requested; NOTE: p_x is kept via rename->n_lags)
_DROP_COLS = [
    "Direction",
    "p_ret",
    "N_obs",
    "r2_u",        # unadjusted R^2
    "N_boot",
    "N_boot_valid",
]

# Preferred display order (only existing columns are kept, in this order)
_PREFERRED_ORDER = [
    "Ticker",
    "AINI_variant",
    "n_lags",                 # kept as exact name
    "Year",
    "F_stat",
    "Original_F_pval",
    "Empirical_F_pval",
    "adj_r2_u",               # adjusted R^2 (kept if present)
    "BH_corr_F_pval",
    "BH_corr_F_pval_HC3",
]


# ------------------------------- helpers ------------------------------------ #
def _project_tables_dir() -> Path:
    """Resolve \\AI_narrative_index\\reports\\tables relative to this file."""
    here = Path(__file__).resolve()
    project_root = here.parents[2]  # .../AI_narrative_index
    return project_root / "reports" / "tables"


def _move_col_after(df: pd.DataFrame, col: str, after: str) -> pd.DataFrame:
    """Move column `col` to immediately follow `after` when both exist."""
    if col not in df.columns:
        return df
    cols = list(df.columns)
    cols.remove(col)
    if after in cols:
        insert_pos = cols.index(after) + 1
        cols.insert(insert_pos, col)
        return df[cols]
    return df


def _stars_from_p(p: float) -> str:
    """Return significance stars for p-value."""
    try:
        if pd.isna(p):
            return ""
        p = float(p)
    except Exception:
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def _fmt_p_with_stars(p: float, digits: int = 3) -> str:
    """Format p-value with fixed decimals and append stars."""
    try:
        if pd.isna(p):
            return ""
        p = float(p)
        base = f"<0.{('0'*(digits-1))}1" if p < 10 ** (-digits) else f"{p:.{digits}f}"
        return f"{base}{_stars_from_p(p)}"
    except Exception:
        return ""


def _column_format(columns: Sequence[str]) -> str:
    """
    Build a compact LaTeX column format string for longtable.

    - Right-align numeric-like columns to keep numbers tight.
    - Remove outer padding with @{} to maximize textwidth usage.
    """
    numeric_like = {
        "n_lags",
        "Year",
        "F_stat",
        "Original_F_pval",
        "Empirical_F_pval",
        "adj_r2_u",
        "BH_corr_F_pval",
        "BH_corr_F_pval_HC3",
    }
    # Left (l) for text-like, Right (r) for numeric-like
    spec = ["r" if c in numeric_like else "l" for c in columns]
    return "@{}" + "".join(spec) + "@{}"

def expand_year_range(val: str) -> str:
        parts = val.split("_")
        if all(p.isdigit() for p in parts):
            expanded = []
            for i, p in enumerate(parts):
                if len(p) == 2:  # 2-digit year, prefix with '20'
                    expanded.append("20" + p)
                else:
                    expanded.append(p)
            return "-".join(expanded)
        return val

# ------------------------------- main API ----------------------------------- #
def export_regression_table(
    df: pd.DataFrame,
    title: str,
    output_filename: str,
    *,
    p_digits: int = 3,
    f_digits: int = 2,
    r2_digits: int = 3,
    font_size_cmd: str = "tiny",   # 'tiny' is smaller than 'scriptsize'
    tabcolsep_pt: float = 2.0,     # even tighter spacing for A4 fit
) -> Path:
    """
    Export regression/GC results to LaTeX `longtable` (DIN A4 friendly).

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with results. Expected columns (as available):
        'Ticker','AINI_variant','p_x','n_lags','Year','F_stat',
        'Original_F_pval','Empirical_F_pval','adj_r2_u',
        'BH_corr_F_pval','BH_corr_F_pval_HC3'.
    title : str
        LaTeX caption for the table.
    output_filename : str
        Base filename; '.tex' is appended if missing.

    Other Parameters
    ----------------
    p_digits : int, default=3
        Decimals for p-values.
    f_digits : int, default=2
        Decimals for F-statistics.
    r2_digits : int, default=3
        Decimals for adjusted R².
    font_size_cmd : str, default='tiny'
        LaTeX size command to apply right before the longtable
        (examples: 'tiny', 'scriptsize', 'footnotesize').
    tabcolsep_pt : float, default=2.0
        Value for \\tabcolsep (in pt) to tighten column spacing.

    Returns
    -------
    pathlib.Path
        Path to the written `.tex` file.

    Notes
    -----
    - `p_x` is renamed to `n_lags` (and kept). The original `p_x` is then removed via the drop list.
    - Drops only: 'Direction', 'p_ret', 'N_obs', 'r2_u', 'N_boot', 'N_boot_valid'.
    - Star annotations are added only to BH-adjusted p-values.
    - The output is a **top-level** `longtable` (no float/resizebox), with zero longtable
      left/right margins, very small font, and tight column spacing to fit DIN A4.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")

    # Resolve output path
    outdir = _project_tables_dir()
    outdir.mkdir(parents=True, exist_ok=True)
    if not output_filename.endswith(".tex"):
        output_filename += ".tex"
    out_path = outdir / output_filename

    # 1) Rename p_x -> n_lags (keep)
    work = df.copy()
    if "p_x" in work.columns:
        work["n_lags"] = work["p_x"]

    # 2) Drop the requested columns (p_x removed implicitly via rename+drop)
    work = work.drop(columns=[c for c in _DROP_COLS if c in work.columns], errors="ignore")

    # 3) Ensure n_lags is positioned after AINI_variant if both exist
    work = _move_col_after(work, "n_lags", "AINI_variant")

    # 4) Keep/arrange a compact, explicit column order
    ordered_cols = [c for c in _PREFERRED_ORDER if c in work.columns]
    if ordered_cols:
        work = work[ordered_cols]

    # Year column: expand underscores to full year range

    if "Year" in work.columns:
        work["Year"] = work["Year"].astype(str).apply(expand_year_range)


    # 5) Formatting (use strings for display)
    # F-stat
    if "F_stat" in work.columns:
        work["F_stat"] = work["F_stat"].apply(lambda v: "" if pd.isna(v) else f"{float(v):.{f_digits}f}")

    # Year column

    if "Year" in work.columns:
        work["Year"] = work["Year"].astype(str).str.replace("_", "-", regex=False)

    # Adjusted R^2 (if present)
    if "adj_r2_u" in work.columns:
        work["adj_r2_u"] = work["adj_r2_u"].apply(lambda v: "" if pd.isna(v) else f"{float(v):.{r2_digits}f}")

    # Plain p-values (no stars)
    for pcol in ("Original_F_pval", "Empirical_F_pval"):
        if pcol in work.columns:
            work[pcol] = work[pcol].apply(
                lambda v: ""
                if pd.isna(v)
                else (f"<0.{('0'*(p_digits-1))}1" if float(v) < 10 ** (-p_digits) else f"{float(v):.{p_digits}f}")
            )

    # BH-adjusted p-values WITH stars
    if "BH_corr_F_pval" in work.columns:
        work["BH_corr_F_pval"] = work["BH_corr_F_pval"].apply(lambda v: _fmt_p_with_stars(v, p_digits))
    if "BH_corr_F_pval_HC3" in work.columns:
        work["BH_corr_F_pval_HC3"] = work["BH_corr_F_pval_HC3"].apply(lambda v: _fmt_p_with_stars(v, p_digits))

    # 6) Column format (tight, right-align numerics)
    colspec = _column_format(work.columns)

    # 7) Build LaTeX longtable (do NOT wrap in table/resizebox)
    # Use escape=True so underscores, %, etc. are escaped correctly.
    latex_core = work.to_latex(
        index=False,
        longtable=True,
        escape=True,
        column_format=colspec,
        caption=title,
        label=f"tab:{out_path.stem}",
        bold_rows=False,
        multicolumn=False,
        multicolumn_format="c",
    )

    # 8) A4-fit helpers: zero longtable margins, very small font, very tight col spacing
    # Wrap in a group so the settings don't leak into the rest of the document.
    preamble = (
        "% AUTO-GENERATED by export_regression_table\n"
        "\\begingroup\n"
        f"\\setlength\\LTleft{{0pt}}\\setlength\\LTright{{0pt}}\n"
        f"\\setlength\\tabcolsep{{{tabcolsep_pt}pt}}\n"
        f"\\{font_size_cmd}\n"
    )
    postamble = "\n\\endgroup\n"

    # 9) Write .tex
    out_path.write_text(preamble + latex_core + postamble, encoding="utf-8")
    return out_path


# rmd tables for presentation

import re
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

# --- helpers ---
def parse_markdown_table(md: str):
    """
    Parse a GitHub-style pipe table into (headers, rows).
    md: one table string starting with a header row, then a separator row, then data rows.
    """
    lines = [ln.strip() for ln in md.strip().splitlines() if ln.strip()]
    # filter out markdown headings accidentally included
    lines = [ln for ln in lines if ln.startswith("|") and ln.endswith("|")]

    if len(lines) < 2:
        raise ValueError("Markdown table must have at least a header and a separator row.")

    def split_row(ln):
        # drop leading and trailing pipe, then split
        parts = [c.strip() for c in ln.strip("|").split("|")]
        return parts

    header = split_row(lines[0])
    # skip the separator line (second)
    body_lines = lines[2:] if len(lines) >= 2 else []
    rows = [split_row(ln) for ln in body_lines]
    return header, rows

def add_markdown_text_to_cell(cell, text: str, base_size=12):
    """
    Write text to a PPTX table cell and make **bold** segments bold.
    """
    # clear default paragraph
    cell.text = ""
    p = cell.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT

    # split on **...**
    tokens = re.split(r"(\*\*.*?\*\*)", text)
    for tok in tokens:
        run = p.add_run()
        if tok.startswith("**") and tok.endswith("**") and len(tok) >= 4:
            run.text = tok[2:-2]
            run.font.bold = True
        else:
            run.text = tok
        run.font.size = Pt(base_size)

def add_table_slide(prs: Presentation, title: str, headers, rows, font_size=12):
    # layout: title + content
    slide = prs.slides.add_slide(prs.slide_layouts[5])  # Title Only (5) or Blank (6)
    # Title
    txbox = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.6))
    p = txbox.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(20)
    p.font.bold = True

    n_rows = max(1, len(rows)) + 1
    n_cols = len(headers)

    # Table size/position
    left, top, width, height = Inches(0.5), Inches(1.0), Inches(9.0), Inches(0.4 + 0.3*n_rows)
    tbl = slide.shapes.add_table(n_rows, n_cols, left, top, width, height).table

    # Header row
    for j, h in enumerate(headers):
        add_markdown_text_to_cell(tbl.cell(0, j), h, base_size=font_size)
        # Header styling
        for run in tbl.cell(0, j).text_frame.paragraphs[0].runs:
            run.font.bold = True

    # Body rows
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            add_markdown_text_to_cell(tbl.cell(i, j), val, base_size=font_size)

    # Optional: column width tweaks (uniform)
    for j in range(n_cols):
        tbl.columns[j].width = Inches(9.0 / n_cols)

    return slide

# --- main export function ---
def markdown_tables_to_pptx(markdown_dict, outpath="a2r_tables_auto.pptx"):
    """
    markdown_dict: {model: [md_table_chunk_1, md_table_chunk_2, ...]}
    Creates one slide per chunk. Slide title: 'Model = {model} (Chunk k)'.
    """
    prs = Presentation()
    for model, chunks in markdown_dict.items():
        for k, md in enumerate(chunks, 1):
            headers, rows = parse_markdown_table(md)
            title = f"Model = {model} (Chunk {k})" if len(chunks) > 1 else f"Model = {model}"
            add_table_slide(prs, title, headers, rows, font_size=12)
    prs.save(outpath)
    return Path(outpath).resolve()

def make_mrkdwn_tables(
    df: pd.DataFrame,
    drop_cols=("BH_corr_F_pval","BH_corr_F_pval_HC3","Direction","A2R_beta_const"),
    priority=(("w0","2023"),("w2","2025"),("w2","2023_24_25")),
    chunk_size=20
):
    """
    Create Markdown table chunks from the input dataframe.

    Returns: dict {model: [markdown_table_chunk1, markdown_table_chunk2, ...]}
    """

    # 1. Reorder rows (priority first)
    mask = df[["Model","Year"]].apply(tuple, axis=1).isin(priority)
    df = pd.concat([df[mask], df[~mask]], ignore_index=True)

    # 2. Drop unwanted cols
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # 3. Rename to Greek symbols
    rename_map = {
        "A2R_beta_ret_lag1": "β_ret",
        "A2R_beta_x_lag1": "γ₁",
        "A2R_beta_x_lag2": "γ₂",
        "A2R_beta_x_lag3": "γ₃",
    }
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})

    # 4. Bold positives
    def bold_pos(x: float) -> str:
        try:
            return f"**{x:.6f}**" if float(x) > 0 else f"{x:.6f}"
        except Exception:
            return str(x)

    for col in ["β_ret","γ₁","γ₂","γ₃"]:
        if col in df.columns:
            df[col] = df[col].apply(bold_pos)

    # 5. Split into Markdown by Model
    out = {}
    for model, sub in df.groupby("Model", sort=False):
        chunks = []
        for i in range(0, len(sub), chunk_size):
            chunk = sub.iloc[i:i+chunk_size]
            chunks.append(chunk.to_markdown(index=False))
        out[model] = chunks

    return out
import pandas as pd

def make_mrkdwn_tables_focus(
    df: pd.DataFrame,
    focus_rows=(("w0", "2023"), ("w2", "2025"), ("w2", "2023_24_25")),
    drop_cols=("BH_corr_F_pval","BH_corr_F_pval_HC3","Direction","A2R_beta_const"),
    chunk_size=11
):
    """
    Create Markdown tables only for the specified (Model, Year) pairs.
    Returns: dict {model: [markdown_table_chunk1, ...]}  (never None).
    """

    # --- sanity: show what pairs exist in the data
    available = set(df[["Model","Year"]].astype(str).apply(tuple, axis=1).unique())
    focus = set((str(m), str(y)) for m, y in focus_rows)
    missing = focus - available
    if missing:
        print("WARN: These (Model, Year) pairs not found in df:", sorted(missing))
        print("INFO: Available pairs sample:", sorted(list(available))[:10])

    # 1) Filter only the focus rows
    mask = df[["Model","Year"]].astype(str).apply(tuple, axis=1).isin(focus)
    sub = df.loc[mask].copy()
    if sub.empty:
        print("WARN: After filtering, no rows remain. Returning empty dict.")
        return {}

    # 2) Drop columns
    sub.drop(columns=[c for c in drop_cols if c in sub.columns], errors="ignore", inplace=True)

    # 3) Rename to Greek
    rename_map = {
        "A2R_beta_ret_lag1": "β_ret",
        "A2R_beta_x_lag1": "γ₁",
        "A2R_beta_x_lag2": "γ₂",
        "A2R_beta_x_lag3": "γ₃",
    }
    sub.rename(columns={k:v for k,v in rename_map.items() if k in sub.columns}, inplace=True)

    # 4) Bold positives
    def bold_pos(x):
        try:
            x = float(x)
            return f"**{x:.6f}**" if x > 0 else f"{x:.6f}"
        except Exception:
            return str(x)

    for col in ("β_ret","γ₁","γ₂","γ₃"):
        if col in sub.columns:
            sub[col] = sub[col].apply(bold_pos)

    # 5) Split by Model and chunk (≤11 rows)
    out = {}
    for model, g in sub.groupby("Model", sort=False):
        if {"Ticker","Year"}.issubset(g.columns):
            g = g.sort_values(["Ticker","Year"], kind="stable")
        chunks = [g.iloc[i:i+chunk_size].to_markdown(index=False) for i in range(0, len(g), chunk_size)]
        out[model] = chunks

    return out
