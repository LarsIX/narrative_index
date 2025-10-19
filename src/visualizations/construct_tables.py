from __future__ import annotations
from pathlib import Path
from typing import Sequence, Optional, Tuple, Union
import pandas as pd
import math

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

_OUTPUT_ORDER = [
    "Model", "AINI^var", "Period", "Ticker",
    "β₀", "β1", "β2", "β3", "γ1", "γ2", "γ3",
    "ζ1", "ζ2", "ζ3",
    "R^2", "R^2_{adj}",
    "Analytical F_stat p", "Empirical F_stat p",
    "BH Analytical p", "BH Empirical p",
]

_DROP_COLS = [
    "Direction", "Lags",
    "N_obs", "N_boot", "N_boot_valid",
    "F_stat", "df_num", "df_den",
    "BH_reject_F", "BH_reject_F_HC3",
    "joint rej. (α=0.1)",
]

_NUMERIC_LIKE = {
    "β₀", "β1", "β2", "β3", "γ1", "γ2", "γ3",
    "ζ1", "ζ2", "ζ3",
    "R^2", "R^2_{adj}",
    "Analytical F_stat p", "Empirical F_stat p",
    "BH Analytical p", "BH Empirical p",
}

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _project_tables_dir() -> Path:
    here = Path(__file__).resolve()
    project_root = here.parents[2]
    return project_root / "reports" / "tables"

def _is_blank(x) -> bool:
    return x is None or (isinstance(x, str) and x.strip() == "")

def _to_float_or_none(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return None
        try:
            return float(s)
        except Exception:
            return None
    return None

def _stars_from_p(p) -> str:
    p = _to_float_or_none(p)
    if p is None:
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""

def _fmt_p_val_for_pipeline(p: float, digits: int = 3) -> str:
    p = _to_float_or_none(p)
    if p is None:
        return ""
    floor = 10 ** (-digits)
    return f"<0.{('0'*(digits-1))}1" if p < floor else f"{p:.{digits}f}"

def _attach_stars(series: pd.Series, digits: int) -> pd.Series:
    vals = series.apply(lambda v: _fmt_p_val_for_pipeline(v, digits))
    stars = series.apply(_stars_from_p)
    return vals + stars

def _escape_latex_text(s: str) -> str:
    if s is None or pd.isna(s):
        return ""
    s = str(s)
    return (s.replace("\\", r"\textbackslash{}")
             .replace("&", r"\&")
             .replace("%", r"\%")
             .replace("$", r"\$")
             .replace("#", r"\#")
             .replace("_", r"\_")
             .replace("{", r"\{")
             .replace("}", r"\}")
             .replace("~", r"\textasciitilde{}")
             .replace("^", r"\textasciicircum{}")
             .replace("<", r"\textless{}")
             .replace(">", r"\textgreater{}"))

def _sanitize_latex_header(col: str) -> str:
    # Display names (headers only). Internal keys remain unchanged.
    mapping = {
        "Model": "Model",
        "AINI^var": r"$AINI^{\mathrm{var}}$",
        "Period": "Period",
        "Ticker": "Ticker",
        "β₀": r"$\beta_{0}$",
        "β1": r"$\beta_{1}$",
        "β2": r"$\beta_{2}$",
        "β3": r"$\beta_{3}$",
        "γ1": r"$\gamma_{1}$",
        "γ2": r"$\gamma_{2}$",
        "γ3": r"$\gamma_{3}$",
        "ζ1": r"$\zeta_{1}$",
        "ζ2": r"$\zeta_{2}$",
        "ζ3": r"$\zeta_{3}$",
        "R^2": r"$R^{2}$",
        "R^2_{adj}": r"$R^{2}_{\mathrm{adj}}$",
        # Long-form p’s (non-BH)
        "Analytical F_stat p": r"\textrm{p (analytical)}",
        "Empirical F_stat p":  r"\textrm{p (empirical)}",
        # BH p’s with short math labels
        "BH Analytical p":     r"$p^{\mathrm{BH}}_{a}$",
        "BH Empirical p":      r"$p^{\mathrm{BH}}_{e}$",
    }
    return mapping.get(col, col)

def _escape_caption(s: str) -> str:
    if s is None:
        return ""
    return (s.replace("&", r"\&")
             .replace("%", r"\%")
             .replace("#", r"\#")
             .replace("_", r"\_")
             .replace("{", r"\{")
             .replace("}", r"\}")
             .replace("~", r"\textasciitilde{}")
             .replace("^", r"\textasciicircum{}")
             .replace("→", r"$\to$"))

def _transform_aini_variant_values_latex(series: pd.Series) -> pd.Series:
    aini_map = {
        "EMA_02": r"$EMA^{0.2}$",
        "EMA_08": r"$EMA^{0.8}$",
        "normalized_AINI": r"$AINI^{norm}$",
    }
    return series.map(lambda v: aini_map.get(v, "" if pd.isna(v) else str(v)))

def _strip_stars(s: str) -> str:
    if s is None or pd.isna(s) or s == "":
        return ""
    s = str(s)
    while s.endswith("*"):
        s = s[:-1]
    return s

def _ensure_zeta_triplet(df: pd.DataFrame) -> pd.DataFrame:
    candidates = {
        "ζ1": ["ζ1", "zeta1", "ζ_1", "z1", "Z1", "ζ1′", "zeta1_prime", "zeta1'"],
        "ζ2": ["ζ2", "zeta2", "ζ_2", "z2", "Z2"],
        "ζ3": ["ζ3", "zeta3", "ζ_3", "z3", "Z3"],
    }
    out = df.copy()
    for target, opts in candidates.items():
        if target in out.columns:
            continue
        found = next((c for c in opts if c in out.columns), None)
        if found:
            out[target] = out[found]
    return out

def _format_coef_series(s: pd.Series, digits: int) -> pd.Series:
    def f(v):
        fv = _to_float_or_none(v)
        return "" if fv is None else f"{fv:.{digits}f}"
    return s.apply(f)

# ---------------------------------------------------------------------
# Core preparation
# ---------------------------------------------------------------------

def _prep_dataframe(
    df: pd.DataFrame,
    coef_digits: Optional[int],
    p_digits: int
) -> pd.DataFrame:
    work = df.copy()

    # Accept multiple likely source names for ζ1
    zeta_prime_sources = ["ζ1", "ζ1_prime", "ζ1′", "zeta1_prime", "zeta1'"]
    found_zeta = next((c for c in zeta_prime_sources if c in work.columns), None)
    if found_zeta and "ζ1" not in work.columns:
        work["ζ1"] = work[found_zeta]

    # Map external names to internal ones (covers your schemas)
    rename_map = {
        "AINI_variant": "AINI^var",
        "r2_u": "R^2",
        "adj_r2_u": "R^2_{adj}",
        # p-values (various spellings)
        "Original_F_pval": "Analytical F_stat p",
        "Empirical_F_pval": "Empirical F_stat p",
        "analytical P": "Analytical F_stat p",
        "empirical P": "Empirical F_stat p",
        # BH p-values (case variations)
        "BH analytical p": "BH Analytical p",
        "BH empirical p": "BH Empirical p",
        "BH Analytical p": "BH Analytical p",
        "BH Empirical p": "BH Empirical p",
    }
    for src, dst in rename_map.items():
        if src in work.columns:
            work[dst] = work[src]

    # Coefficient formatting (robust to blanks)
    if coef_digits is not None:
        for c in ["β₀", "β1", "β2", "β3", "γ1", "γ2", "γ3", "ζ1", "ζ2", "ζ3"]:
            if c in work.columns:
                work[c] = _format_coef_series(work[c], coef_digits)

    # Preliminary p formatting for non-BH
    for pcol in ["Analytical F_stat p", "Empirical F_stat p"]:
        if pcol in work.columns:
            work[pcol] = work[pcol].apply(lambda v: _fmt_p_val_for_pipeline(v, p_digits))

    # DISTINCT stars per BH column
    if "BH Analytical p" in work.columns:
        work["BH Analytical p"] = _attach_stars(work["BH Analytical p"], p_digits)
    if "BH Empirical p" in work.columns:
        work["BH Empirical p"] = _attach_stars(work["BH Empirical p"], p_digits)

    # Drop unused
    work = work.drop(columns=[c for c in _DROP_COLS if c in work.columns], errors="ignore")

    # Order
    cols = [c for c in _OUTPUT_ORDER if c in work.columns]
    if cols:
        work = work[cols]
    return work

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def export_regression_table(
    df: pd.DataFrame,
    title: str,
    output_filename: str,
    *,
    sort_by: Sequence[str] | None = None,
    output_format: str = "tex",
    latex_env: str = "longtable",         # kept for API compatibility
    include_caption_label: bool = True,
    font_size_cmd: str = "scriptsize",    # ignored; we set size in template
    tabcolsep_pt: float = 1.6,            # slightly tighter
    coef_digits: Optional[int] = 3,
    p_digits: int = 3,
    reverse: bool = False,                # NEW: switch column set for LaTeX
    first4_gap_pt: float = 0.5,           # NEW: tighter gaps between first 4 columns
    gap_after_first4_pt: float = 2.0,     # NEW: small gap before numeric block
) -> Union[Path, Tuple[Path, Path]]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")
    if output_format not in {"tex", "html"}:
        raise ValueError("output_format must be 'tex' or 'html'")

    outdir = _project_tables_dir()
    outdir.mkdir(parents=True, exist_ok=True)
    if not output_filename.endswith(f".{output_format}"):
        output_filename = f"{output_filename}.{output_format}"
    out_path = outdir / output_filename

    # Reporting filter (guarded)
    if {"Model", "Period", "AINI_variant"}.issubset(df.columns):
        df = df[(df["Model"] == "custom") | (df["Period"] != "2025")]
        df = df[~((df["AINI_variant"] == "EMA_02") & (df["Period"] == "2025"))]

    # Prep (generic)
    work = _prep_dataframe(df, coef_digits=coef_digits, p_digits=p_digits)

    # Ensure ζ1..ζ3 surfaced from alternates
    work = _ensure_zeta_triplet(work)

    # Optional sorting
    if sort_by:
        missing = [c for c in sort_by if c not in work.columns]
        if not missing:
            work = work.sort_values(list(sort_by)).reset_index(drop=True)

    # ---------------------------- LaTeX ----------------------------
    if output_format == "tex":
        # Choose column layout
        if reverse:
            # β1, β2, β3, γ1 (no ζ1, no γ2, γ3)
            latex_cols = [
                "Model", "AINI^var", "Period", "Ticker",
                "β1", "β2", "β3", "γ1",
                "R^2_{adj}", "BH Analytical p", "BH Empirical p",
            ]
        else:
            # Default layout
            latex_cols = [
                "Model", "AINI^var", "Period", "Ticker",
                "β1", "ζ1", "γ1", "γ2", "γ3",
                "R^2_{adj}", "BH Analytical p", "BH Empirical p",
            ]

        # Create empty columns if missing so the table still compiles
        for c in latex_cols:
            if c not in work.columns:
                work[c] = ""
        work = work[latex_cols]

        # Map AINI_variant values to TeX strings
        if "AINI^var" in work.columns:
            raw_like = work["AINI^var"].astype(str).str.contains("EMA_|normalized_AINI", na=False)
            if raw_like.any():
                work.loc[raw_like, "AINI^var"] = _transform_aini_variant_values_latex(
                    work.loc[raw_like, "AINI^var"]
                )

        # Format R^2_adj
        if "R^2_{adj}" in work.columns:
            work["R^2_{adj}"] = work["R^2_{adj}"].apply(
                lambda v: "" if _to_float_or_none(v) is None else f"{_to_float_or_none(v):.3f}"
            )

        # Keep stars; p to 2 decimals; preserve "<0.01"
        def _fmt_with_stars(val: str) -> str:
            if _is_blank(val):
                return ""
            s = str(val)
            stars = ""
            while s.endswith("*"):
                stars += "*"
                s = s[:-1]
            if s.startswith("<"):
                return "<0.01" + stars
            num = _to_float_or_none(s)
            return (f"{num:.2f}" if num is not None else s) + stars

        for pc in ["BH Analytical p", "BH Empirical p"]:
            if pc in work.columns:
                work[pc] = work[pc].apply(_fmt_with_stars)

        # Escape object columns except AINI^var (already TeX)
        for c in work.select_dtypes(include="object").columns:
            if c == "AINI^var":
                continue
            work[c] = work[c].apply(_escape_latex_text)

        # Headers
        header_cols = [_sanitize_latex_header(c) for c in work.columns]

        # Column spec: X X X X (with tiny gaps) + r... (with a slightly larger gap after the 4th)
        ncols = len(latex_cols)
        gap = f"@{{\\hspace{{{first4_gap_pt}pt}}}}"
        gap_after = f"@{{\\hspace{{{gap_after_first4_pt}pt}}}}"
        colspec = "@{}" + ("X" + gap) * 3 + "X" + gap_after + " " + " ".join(["r"] * (ncols - 4)) + "@{}"

        # Titles and legend
        big_title = "Granger–Causality: jointly significant results (AINI $\\to$ Returns, VIX-controlled)"
        cont_title = "Granger–Causality: jointly significant results (continued)"
        legend = r"Signif.: * $p<0.10$, ** $p<0.05$, *** $p<0.01$."
        latex_caption_main = _escape_caption(title) if include_caption_label else ""
        latex_caption = (latex_caption_main + (" " if latex_caption_main else "") + legend).strip()
        latex_label = f"\\label{{tab:{out_path.stem}}}" if include_caption_label else ""

        # Body
        def row_to_tex(row) -> str:
            return " & ".join("" if pd.isna(row[c]) else str(row[c]) for c in latex_cols) + r" \\"

        body_rows = "\n".join(row_to_tex(row) for _, row in work.iterrows())

        # Compose
        content = (
            "{\\tiny\n"  # slightly smaller than scriptsize to make it fit
            f"\\setlength{{\\tabcolsep}}{{{tabcolsep_pt}pt}}\n"
            "\\renewcommand{\\arraystretch}{1.03}\n"
            "\\setlength{\\LTleft}{0pt}\n"
            "\\setlength{\\LTright}{0pt}\n\n"
            f"\\begin{{tabularx}}{{\\textwidth}}{{{colspec}}}\n"
            "\\toprule\n"
            f"\\multicolumn{{{ncols}}}{{c}}{{\\textbf{{{big_title}}}}}\\\\\n"
            "\\midrule\n"
            f"{' & '.join(header_cols)} \\\\\n"
            "\\midrule\n"
            "\\endfirsthead\n"
            "\\toprule\n"
            f"\\multicolumn{{{ncols}}}{{c}}{{\\textbf{{{cont_title}}}}}\\\\\n"
            "\\midrule\n"
            f"{' & '.join(header_cols)} \\\\\n"
            "\\midrule\n"
            "\\endhead\n"
            "\\midrule\n"
            f"\\multicolumn{{{ncols}}}{{r}}{{\\small Continued on next page}}\\\\\n"
            "\\midrule\n"
            "\\endfoot\n"
            "\\bottomrule\n"
            "\\captionsetup{belowskip=6pt}\n"
            f"\\caption{{{latex_caption}}}\n"
            f"{latex_label}\\\\\n"
            "\\endlastfoot\n"
            f"{body_rows}\n"
            "\\end{tabularx}\n"
            "}\n"
        )

        out_path.write_text(content, encoding="utf-8")
        return out_path

    # ---------------------------- HTML ----------------------------
    html_work = work.copy()

    if "Analytical F_stat p" in html_work.columns:
        html_work["analytical p"] = html_work["Analytical F_stat p"].apply(_strip_stars)
    if "Empirical F_stat p" in html_work.columns:
        html_work["empirical p"] = html_work["Empirical F_stat p"].apply(_strip_stars)

    HTML_OUTPUT_ORDER = [
        "Model", "AINI^var", "Period", "Ticker",
        "β₀", "β1", "β2", "β3",
        "γ1", "γ2", "γ3",
        "ζ1", "ζ2", "ζ3",
        "R^2", "R^2_{adj}",
        "analytical p", "empirical p",
        "BH Analytical p", "BH Empirical p",
    ]
    html_cols = [c for c in HTML_OUTPUT_ORDER if c in html_work.columns]
    if html_cols:
        html_work = html_work[html_cols]

    legend_html = "<em>Signif.: * p&lt;0.10, ** p&lt;0.05, *** p&lt;0.01.</em>"

    html_core = (
        html_work.style
        .set_caption(f"{title} — {legend_html}")
        .set_table_styles([
            {"selector": "caption", "props": [("caption-side", "top"), ("font-weight", "bold"), ("font-size", "120%")]},
            {"selector": "th, td", "props": [("border", "1px solid #ddd"), ("padding", "4px 6px"), ("text-align", "center")]},
            {"selector": "thead", "props": [("background-color", "#f2f2f2")]}
        ])
        .hide(axis="index")
        .to_html()
    )

    html_doc = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 40px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
</style>
</head>
<body>
<h2>{title}</h2>
<p>{legend_html}</p>
{html_core}
</body>
</html>
"""
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path
