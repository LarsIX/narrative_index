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

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
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

# Columns where positive values should be bolded
BOLD_IF_POSITIVE_COLS = {'β₁','β₂','β₃','γ₁','γ₂','γ₃'}

# --- helpers ---
def add_text_to_cell(cell, text: str, bold=False, base_size=12):
    """
    Write plain text to a PPTX cell.
    If bold=True, only set bold. Do NOT change size or color.
    """
    cell.text = ""
    p = cell.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run = p.add_run()
    run.text = text
    run.font.size = Pt(base_size)
    run.font.bold = bold  # no color changes

def add_table_slide(prs: Presentation, title: str, headers, rows, font_size=12):
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    txbox = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.6))
    p = txbox.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(20)
    p.font.bold = True

    n_rows, n_cols = len(rows) + 1, len(headers)
    tbl = slide.shapes.add_table(
        n_rows, n_cols,
        Inches(0.5), Inches(1.0), Inches(9.0), Inches(0.4 + 0.3*n_rows)
    ).table

    # headers
    for j, h in enumerate(headers):
        add_text_to_cell(tbl.cell(0, j), str(h), bold=False, base_size=font_size)
        for run in tbl.cell(0, j).text_frame.paragraphs[0].runs:
            run.font.bold = True

    # body
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            col_name = headers[j]
            text = "not tested" if pd.isna(val) else str(val)

            # Bold only if column in target set AND numeric positive
            make_bold = False
            if col_name in BOLD_IF_POSITIVE_COLS and pd.notna(val):
                try:
                    make_bold = float(val) > 0
                except Exception:
                    make_bold = False

            add_text_to_cell(tbl.cell(i, j), text, bold=make_bold, base_size=font_size)

    # equal column widths
    for j in range(n_cols):
        tbl.columns[j].width = Inches(9.0 / n_cols)

    return slide

def df_to_pptx(df: pd.DataFrame, outpath="df_tables.pptx", rows_per_slide=12):
    prs = Presentation()
    headers = list(df.columns)

    for start in range(0, len(df), rows_per_slide):
        chunk = df.iloc[start:start+rows_per_slide]
        rows = chunk.fillna("not tested").values.tolist()
        add_table_slide(
            prs, f"Rows {start+1}-{start+len(chunk)}",
            headers, rows, font_size=12
        )

    prs.save(outpath)
    return Path(outpath).resolve()