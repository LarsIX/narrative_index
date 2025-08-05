"""
construct_latex_tables.py

This module provides functions to export formatted LaTeX tables from 
Granger causality estimation results and model diagnostics.

It supports:
- Coefficient reporting (including p-value significance stars)
- Model fit statistics (AIC, BIC)
- Residual diagnostics (Breusch–Pagan and White tests)
- Context-specific output filenames and LaTeX labels
- Automatic LaTeX formatting with column renaming and escaping

Functions
---------
build_coef_table(df, w, direction, caption, label)
    Create a LaTeX table summarizing Granger causality test coefficients 
    and significance levels (***, **, *).

build_diagnostics_table(df, w, direction, caption, label)
    Create a LaTeX table summarizing model diagnostics including AIC, BIC,
    and tests for heteroskedasticity (BP and White tests).

Examples
--------
>>> from export_latex_tables import build_coef_table, build_diagnostics_table
>>> build_coef_table(df_gc, w="1", direction="AINI_to_ret", caption="...", label="tab:gc_coef")
>>> build_diagnostics_table(df_diag, w="1", direction="AINI_to_ret", caption="...", label="tab:gc_diag")
"""


import pandas as pd
from pathlib import Path

root = Path(__file__).parents[2]
output_path = root / "reports" / "tables"
output_path.mkdir(parents=True, exist_ok=True)

def build_coef_table(df, w, direction, caption, label):
    """
    Transforms estimated Granger Causality results to LaTeX table & writes table to /reports/tables.
    """

    coef_list = ["Ticker", "AINI_variant", "Year", "coef_x1", "coef_x2", "Original_F", "BH_corr_F_pval"]
    display_df = df.loc[:, coef_list].copy()

    display_df.columns = [
        "Ticker", "AINI Variant", "Year", "Coef x1", "Coef x2", 
        "F-Stat", "FDR p-val"
    ]

    display_df["FDR p-val"] = pd.to_numeric(display_df["FDR p-val"], errors="coerce")

    def format_f_stat(row):
        stars = ""
        p = row["FDR p-val"]
        if pd.notnull(p):
            if p < 0.01:
                stars = "***"
            elif p < 0.05:
                stars = "**"
            elif p < 0.1:
                stars = "*"
        f = row["F-Stat"]
        return f"{f:.2f}{stars}" if pd.notnull(f) else ""

    display_df["Coef x1"] = pd.to_numeric(display_df["Coef x1"], errors="coerce").map(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
    display_df["Coef x2"] = pd.to_numeric(display_df["Coef x2"], errors="coerce").map(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
    display_df["F-Stat"] = display_df.apply(format_f_stat, axis=1)
    display_df["FDR p-val"] = display_df["FDR p-val"].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "")

    display_df["AINI Variant"] = display_df["AINI Variant"].replace({
        "EMA_02": "EMA\\_02",
        "EMA_08": "EMA\\_08",
        "normalized_AINI": "normalized\\_AINI",
        "normalized_AINI_growth": "normalized\\_AINI\\_growth"
    })

    display_df["Year"] = display_df["Year"].replace({
        "2023_24": "2023--2024",
        "2024_25": "2024--2025",
        "2023_24_25": "2023--2025"
    })

    inner_table = display_df.to_latex(
        index=False,
        caption='',
        label='',
        column_format="l l l r r r r",
        escape=False
    )

    latex_note = "\\vspace{0.2cm}\n\\textit{Note:} \\textbf{***} p$<$0.01, \\textbf{**} p$<$0.05, \\textbf{*} p$<$0.1"

    latex_table = f"""\\begin{{table}}[H]
\\normalsize
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\resizebox{{\\textwidth}}{{!}}{{
{inner_table}
}}
{latex_note}
\\end{{table}}"""

    tex_filename = output_path / f"gc_w{w}_{direction}.tex"
    with open(tex_filename, "w", encoding="utf-8") as f:
        f.write(latex_table)

    return latex_table


def build_diagnostics_table(df, w, direction, caption, label):
    """
    Transforms Granger model diagnostics into a LaTeX table & writes it to /reports/tables.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing: 'Ticker', 'AINI_variant', 'Year', 'Lag',
        'AIC', 'BIC', 'bp_stat', 'BP_pval', 'White_stat', 'White_pval'.
    w : str
        Context window size used in estimation.
    direction : str
        Direction of causality tested (e.g., 'AINI_to_ret').
    caption : str
        Caption for the LaTeX table.
    label : str
        LaTeX label for referencing the table.

    Returns
    -------
    str
        The LaTeX-formatted table string.
    """

    # Ensure consistent column names
    df = df.rename(columns={"bp_stat": "BP_stat"})

    col_list = ['Ticker', 'AINI_variant', 'Year', 'Lag', 'AIC', 'BIC', 'BP_stat', 'BP_pval', 'White_stat', 'White_pval']
    display_df = df.loc[:, col_list].copy()

    # Rename for LaTeX
    display_df.columns = [
        "Ticker", "AINI Variant", "Year", "Lag", 
        "AIC", "BIC", 
        "BP Stat", "BP p-val", 
        "White Stat", "White p-val"
    ]

    # Format all floats first
    for col in ["AIC", "BIC", "BP Stat", "BP p-val", "White Stat", "White p-val"]:
        display_df[col] = pd.to_numeric(display_df[col], errors="coerce")

    # Add stars based on significance for BP and White tests
    def add_stars(stat, pval):
        if pd.isnull(stat) or pd.isnull(pval):
            return ""
        stars = ""
        if pval < 0.01:
            stars = "***"
        elif pval < 0.05:
            stars = "**"
        elif pval < 0.1:
            stars = "*"
        return f"{stat:.2f}{stars}"

    display_df["BP Stat"] = display_df.apply(lambda row: add_stars(row["BP Stat"], row["BP p-val"]), axis=1)
    display_df["White Stat"] = display_df.apply(lambda row: add_stars(row["White Stat"], row["White p-val"]), axis=1)

    # Format p-values
    display_df["BP p-val"] = display_df["BP p-val"].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
    display_df["White p-val"] = display_df["White p-val"].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "")

    # Clean variant names for LaTeX
    display_df["AINI Variant"] = display_df["AINI Variant"].replace({
        "EMA_02": "EMA\\_02",
        "EMA_08": "EMA\\_08",
        "normalized_AINI": "normalized\\_AINI",
        "normalized_AINI_growth": "normalized\\_AINI\\_growth"
    })

    display_df["Year"] = display_df["Year"].replace({
        "2023_24": "2023--2024",
        "2024_25": "2024--2025",
        "2023_24_25": "2023--2025"
    })

    inner_table = display_df.to_latex(
        index=False,
        caption='',
        label='',
        column_format="l l l r r r r r r r",
        escape=False
    )

    latex_table = f"""\\begin{{table}}[H]
\\normalsize
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\resizebox{{\\textwidth}}{{!}}{{
{inner_table}
}}
\\end{{table}}"""

    tex_filename = output_path / f"diag_gc_w{w}_{direction}.tex"
    with open(tex_filename, "w", encoding="utf-8") as f:
        f.write(latex_table)

    return latex_table
