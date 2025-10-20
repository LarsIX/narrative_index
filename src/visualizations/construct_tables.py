from __future__ import annotations
from pathlib import Path
from typing import Sequence, Optional, Tuple, Union
import pandas as pd
import math

_OUTPUT_ORDER = [
    "Model", "AINI^var", "Period", "Ticker", "Direction",
    "β₀", "β1", "β2", "β3", "γ1", "γ2", "γ3",
    "ζ1", "ζ2", "ζ3",
    "R^2", "R^2_{adj}",
    "Analytical F_stat p", "Empirical F_stat p",
    "BH analytical p", "BH empirical p",
]

_DROP_COLS = [
    "Lags",
    "N_obs", "N_boot", "N_boot_valid",
    "F_stat", "df_num", "df_den",
    "BH_reject_F", "BH_reject_F_HC3",
    "joint rej. (α=0.1)", "p_x",
]

_NUMERIC_LIKE = {
    "β₀", "β1", "β2", "β3", "γ1", "γ2", "γ3",
    "ζ1", "ζ2", "ζ3",
    "R^2", "R^2_{adj}",
    "Analytical F_stat p", "Empirical F_stat p",
    "BH analytical p", "BH empirical p",
}

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
    mapping = {
        "Model": "Model",
        "AINI^var": r"$AINI^{\mathrm{var}}$",
        "Period": "Period",
        "Ticker": "Ticker",
        "Direction": "Direction",
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
        "Analytical F_stat p": r"\textrm{p (analytical, Wald-F)}",
        "Empirical F_stat p":  r"\textrm{p (empirical, bootstrap)}",
        "BH analytical p":     r"\textrm{BH adj. analytical p}",
        "BH empirical p":      r"\textrm{BH adj. empirical p}",
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

def _uniquify_columns(df: pd.DataFrame) -> pd.DataFrame:
    counts = {}
    new_cols = []
    for c in df.columns:
        if c in counts:
            counts[c] += 1
            new_cols.append(f"{c}.{counts[c]-1}")
        else:
            counts[c] = 1
            new_cols.append(c)
    df = df.copy()
    df.columns = new_cols
    return df

def _apply_custom_renames(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    rename_pairs = {
        "β_x1_ar": "β1",
        "β_x2_ar": "β2",
        "β_x3_ar": "β3",
        "β_x1": "γ1",
        "β_x2": "γ2",
        "β_x3": "γ3",
        "A2R_beta_ret_1": "β1",
        "A2R_beta_ret_2": "β2",
        "A2R_beta_ret_3": "β3",
        "A2R_beta_x_1": "γ1",
        "A2R_beta_x_2": "γ2",
        "A2R_beta_x_3": "γ3",
        "β₀.1": "β0 (rev.)",
        "β_x1_ar.1": "β1 (rev)",
        "β_x2_ar.1": "β2 (rev)",
        "β_x3_ar.1": "β3 (rev)",
        "β1.1": "β1 (rev)",
        "β2.1": "β2 (rev)",
        "β3.1": "β3 (rev)",
        "β_ctrl_log_growth_closed1": "ζ1",
        "β_ctrl_log_growth_closed2": "ζ2",
        "β_ctrl_log_growth_closed3": "ζ3",
        "β_ctrl_log_growth_closed1.1": "ζ1 (rev.)",
        "β_ctrl_log_growth_closed2.1": "ζ2 (rev.)",
        "β_ctrl_log_growth_closed3.1": "ζ3 (rev.)",
        "γ1.1": "γ1 (rev.)",
        "γ2.1": "γ2 (rev.)",
        "γ3.1": "γ3 (rev.)",
        "BH_corr_F_pval_HC3": "BH analytical p",
        "BH_corr_F_pval": "BH empirical p",
        "BH empirical p": "BH empirical p",
        "BH analytical p": "BH analytical p",
    }
    for src, dst in rename_pairs.items():
        if src in d.columns:
            d[dst] = d[src]
    return d

def _prep_dataframe(
    df: pd.DataFrame,
    coef_digits: Optional[int],
    p_digits: int
) -> pd.DataFrame:
    work = _uniquify_columns(df)
    zeta_prime_sources = ["ζ1", "ζ1_prime", "ζ1′", "zeta1_prime", "zeta1'"]
    found_zeta = next((c for c in zeta_prime_sources if c in work.columns), None)
    if found_zeta and "ζ1" not in work.columns:
        work["ζ1"] = work[found_zeta]
    rename_map = {
        "AINI_variant": "AINI^var",
        "r2_u": "R^2",
        "adj_r2_u": "R^2_{adj}",
        "Original_F_pval": "Analytical F_stat p",
        "Empirical_F_pval": "Empirical F_stat p",
        "analytical P": "Analytical F_stat p",
        "empirical P": "Empirical F_stat p",
        "BH Empirical p": "BH empirical p",
        "BH Analytical p": "BH analytical p",
    }
    for src, dst in rename_map.items():
        if src in work.columns:
            work[dst] = work[src]
    work = _apply_custom_renames(work)
    if "p_x" in work.columns:
        work = work.drop(columns=["p_x"])
    for c in ["β₀","β1","β2","β3","γ1","γ2","γ3","ζ1","ζ2","ζ3",
              "β0 (rev.)","β1 (rev)","β2 (rev)","β3 (rev)",
              "ζ1 (rev.)","ζ2 (rev.)","ζ3 (rev.)","γ1 (rev.)","γ2 (rev.)","γ3 (rev.)"]:
        if c in work.columns and coef_digits is not None:
            work[c] = _format_coef_series(work[c], coef_digits)
    for pcol in ["Analytical F_stat p", "Empirical F_stat p", "BH analytical p", "BH empirical p"]:
        if pcol in work.columns:
            work[pcol] = work[pcol].apply(lambda v: _fmt_p_val_for_pipeline(v, p_digits))
    work = work.drop(columns=[c for c in _DROP_COLS if c in work.columns], errors="ignore")
    cols = [c for c in _OUTPUT_ORDER if c in work.columns]
    if cols:
        work = work[cols + [c for c in work.columns if c not in cols]]
    return work

def export_regression_table(
    df: pd.DataFrame,
    title: str,
    output_filename: str,
    *,
    sort_by: Sequence[str] | None = None,
    output_format: str = "tex",
    latex_env: str = "longtable",
    include_caption_label: bool = True,
    font_size_cmd: str = "scriptsize",
    tabcolsep_pt: float = 1.6,
    coef_digits: Optional[int] = 3,
    p_digits: int = 3,
    reverse: bool = False,
    first4_gap_pt: float = 0.5,
    gap_after_first4_pt: float = 2.0,
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
    if {"Model", "Period", "AINI_variant"}.issubset(df.columns):
        df = df[(df["Model"] == "custom") | (df["Period"] != "2025")]
        df = df[~((df["AINI_variant"] == "EMA_02") & (df["Period"] == "2025"))]
    work = _prep_dataframe(df, coef_digits=coef_digits, p_digits=p_digits)
    work = _ensure_zeta_triplet(work)
    if sort_by:
        missing = [c for c in sort_by if c not in work.columns]
        if not missing:
            work = work.sort_values(list(sort_by)).reset_index(drop=True)
    if output_format == "tex":
        if reverse:
            latex_cols = [
                "Model", "AINI^var", "Period", "Ticker", "Direction",
                "β1", "β2", "β3", "γ1",
                "R^2_{adj}", "BH analytical p", "BH empirical p",
            ]
        else:
            latex_cols = [
                "Model", "AINI^var", "Period", "Ticker", "Direction",
                "β1", "ζ1", "γ1", "γ2", "γ3",
                "R^2_{adj}", "BH analytical p", "BH empirical p",
            ]
        for c in latex_cols:
            if c not in work.columns:
                work[c] = ""
        work = work[latex_cols]
        if "AINI^var" in work.columns:
            raw_like = work["AINI^var"].astype(str).str.contains("EMA_|normalized_AINI", na=False)
            if raw_like.any():
                work.loc[raw_like, "AINI^var"] = _transform_aini_variant_values_latex(
                    work.loc[raw_like, "AINI^var"]
                )
        if "R^2_{adj}" in work.columns:
            work["R^2_{adj}"] = work["R^2_{adj}"].apply(
                lambda v: "" if _to_float_or_none(v) is None else f"{_to_float_or_none(v):.3f}"
            )
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
        for pc in ["BH analytical p", "BH empirical p"]:
            if pc in work.columns:
                work[pc] = work[pc].apply(_fmt_with_stars)
        for c in work.select_dtypes(include="object").columns:
            if c == "AINI^var":
                continue
            work[c] = work[c].apply(_escape_latex_text)
        header_cols = [_sanitize_latex_header(c) for c in work.columns]
        ncols = len(latex_cols)
        gap = f"@{{\\hspace{{{first4_gap_pt}pt}}}}"
        gap_after = f"@{{\\hspace{{{gap_after_first4_pt}pt}}}}"
        colspec = "@{}" + ("X" + gap) * 4 + "X" + gap_after + " " + " ".join(["r"] * (ncols - 5)) + "@{}"
        big_title = "Granger–Causality (rev. indicates coefficients for Return $\\Rightarrow$ AINI direction)"
        cont_title = "Granger–Causality (continued) (rev. indicates coefficients for Return $\\Rightarrow$ AINI direction)"
        legend = r"Signif.: * $p<0.10$, ** $p<0.05$, *** $p<0.01$. BH = Benjamini Hochberg Method."
        latex_caption_main = _escape_caption(title) if include_caption_label else ""
        latex_caption = (latex_caption_main + " — rev. indicates coefficients for Return $\\Rightarrow$ AINI direction. " + legend).strip()
        latex_label = f"\\label{{tab:{out_path.stem}}}" if include_caption_label else ""
        def row_to_tex(row) -> str:
            return " & ".join("" if pd.isna(row[c]) else str(row[c]) for c in latex_cols) + r" \\"
        body_rows = "\n".join(row_to_tex(row) for _, row in work.iterrows())
        content = (
            "{\\tiny\n"
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

    work_html = work.copy()
    if "AINI_variant" not in work_html.columns and "AINI^var" in work_html.columns:
        work_html["AINI_variant"] = work_html["AINI^var"]
    if "r2_u" not in work_html.columns and "R^2" in work_html.columns:
        work_html["r2_u"] = work_html["R^2"]
    if "adj_r2_u" not in work_html.columns and "R^2_{adj}" in work_html.columns:
        work_html["adj_r2_u"] = work_html["R^2_{adj}"]
    if "Original_F_pval" not in work_html.columns and "Analytical F_stat p" in work_html.columns:
        work_html["Original_F_pval"] = work_html["Analytical F_stat p"]
    if "Empirical_F_pval" not in work_html.columns and "Empirical F_stat p" in work_html.columns:
        work_html["Empirical_F_pval"] = work_html["Empirical F_stat p"]
    if "BH analytical p" not in work_html.columns and "BH_corr_F_pval_HC3" in work_html.columns:
        work_html["BH analytical p"] = work_html["BH_corr_F_pval_HC3"]
    if "BH empirical p" not in work_html.columns and "BH_corr_F_pval" in work_html.columns:
        work_html["BH empirical p"] = work_html["BH_corr_F_pval"]

    REQUIRED_HTML_COLS = [
        'Ticker', 'AINI_variant', 'Period', 'Direction',
        'β₀', 'β1', 'β2', 'β3', 'γ1', 'γ2', 'γ3',
        'ζ1', 'ζ2', 'ζ3',
        'N_obs', 'N_boot', 'N_boot_valid',
        'F_stat', 'F_stat_obs_RSS', 'Original_F_pval', 'Empirical_F_pval',
        'r2_u', 'adj_r2_u',
        'β0 (rev.)', 'β1 (rev)', 'β2 (rev)', 'β3 (rev)',
        'ζ1 (rev.)', 'ζ2 (rev.)', 'ζ3 (rev.)',
        'BH analytical p', 'BH empirical p',
        'Model',
    ]
    cols_present = [c for c in REQUIRED_HTML_COLS if c in work_html.columns]
    html_df = work_html[cols_present].copy()

    def _fmt_p_strict(v):
        s = str(v)
        if s.startswith("<"):
            return s
        try:
            x = float(s)
            return f"{x:.3f}"
        except Exception:
            return s

    for pcol in ["Original_F_pval", "Empirical_F_pval", "BH analytical p", "BH empirical p"]:
        if pcol in html_df.columns:
            html_df[pcol] = html_df[pcol].apply(_fmt_p_strict)

    legend_html = "<em>Signif.: * p&lt;0.10, ** p&lt;0.05, *** p&lt;0.01. — rev. indicates coefficients for Return ⇒ AINI direction. BH = Benjamini Hochberg Method.</em>"
    table_html = (
        html_df.style
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
{table_html}
</body>
</html>
"""
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path
