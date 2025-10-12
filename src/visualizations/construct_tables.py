from __future__ import annotations
from pathlib import Path
from typing import Sequence, Optional, Tuple, Union
import pandas as pd

_OUTPUT_ORDER = [
    "Model", "AINI^var", "Period", "Ticker",
    "β₀", "β1", "β2", "β3", "γ1", "γ2", "γ3",
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
    "R^2", "R^2_{adj}",
    "Analytical F_stat p", "Empirical F_stat p",
    "BH Analytical p", "BH Empirical p",
}

def _project_tables_dir() -> Path:
    here = Path(__file__).resolve()
    project_root = here.parents[2]
    return project_root / "reports" / "tables"

def _stars_from_p(p: float) -> str:
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

def _fmt_p(p: float, digits: int = 3) -> str:
    try:
        if pd.isna(p):
            return ""
        p = float(p)
        floor = 10 ** (-digits)
        return f"<0.{('0'*(digits-1))}1" if p < floor else f"{p:.{digits}f}"
    except Exception:
        return ""

def _column_format(columns: Sequence[str]) -> str:
    spec = ["r" if c in _NUMERIC_LIKE else "l" for c in columns]
    return "@{}" + "".join(spec) + "@{}"

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
        "β₀": r"$\beta_{0}$",
        "β1": r"$\beta_{1}$",
        "β2": r"$\beta_{2}$",
        "β3": r"$\beta_{3}$",
        "γ1": r"$\gamma_{1}$",
        "γ2": r"$\gamma_{2}$",
        "γ3": r"$\gamma_{3}$",
        "R^2": r"$R^{2}$",
        "R^2_{adj}": r"$R^{2}_{\mathrm{adj}}$",
        "AINI^var": r"$AINI^{\mathrm{var}}$",
    }
    return mapping.get(col, col)

def _escape_caption(s: str) -> str:
    if s is None:
        return ""
    return (s.replace("&", r"\&")
             .replace("%", r"\%")
             .replace("#", r"\#")
             .replace("_", r"\_")
             .replace("~", r"\textasciitilde{}")
             .replace("^", r"\textasciicircum{}")
             .replace("→", r"$\to$"))

def _transform_aini_variant_values(series: pd.Series) -> pd.Series:
    mapping = {
        "EMA_08": r"$EMA_{0.8}$",
        "EMA_02": r"$EMA_{0.2}$",
        "normalized_AINI": r"$AINI^{\mathrm{norm}}$",
        "normalized_AINI_z": r"$AINI^{z}$",
    }
    return series.map(lambda v: mapping.get(v, "" if pd.isna(v) else str(v)))

def _prep_dataframe(
    df: pd.DataFrame,
    coef_digits: Optional[int],
    p_digits: int
) -> pd.DataFrame:
    work = df.copy()
    rename_map = {
        "AINI_variant": "AINI^var",
        "r2_u": "R^2",
        "adj_r2_u": "R^2_{adj}",
        "Original_F_pval": "Analytical F_stat p",
        "Empirical_F_pval": "Empirical F_stat p",
        "BH analytical p": "BH Analytical p",
        "BH empirical p": "BH Empirical p",
    }
    for src, dst in rename_map.items():
        if src in work.columns:
            work[dst] = work[src]
    if "AINI^var" in work.columns:
        work["AINI^var"] = _transform_aini_variant_values(work["AINI^var"])
    if coef_digits is not None:
        for c in ["β₀", "β1", "β2", "β3", "γ1", "γ2", "γ3"]:
            if c in work.columns:
                work[c] = work[c].apply(lambda v: "" if pd.isna(v) else f"{float(v):.{coef_digits}f}")
    for pcol in ["Analytical F_stat p", "Empirical F_stat p"]:
        if pcol in work.columns:
            work[pcol] = work[pcol].apply(lambda v: _fmt_p(v, p_digits))
    stars_source = None
    if "BH Empirical p" in work.columns:
        stars_source = work["BH Empirical p"].apply(_stars_from_p)
        work["BH Empirical p"] = work["BH Empirical p"].apply(lambda v: _fmt_p(v, p_digits))
    if "BH Analytical p" in work.columns:
        work["BH Analytical p"] = work["BH Analytical p"].apply(lambda v: _fmt_p(v, p_digits))
    if stars_source is not None:
        if "BH Empirical p" in work.columns:
            work["BH Empirical p"] = work["BH Empirical p"] + stars_source
        if "BH Analytical p" in work.columns:
            work["BH Analytical p"] = work["BH Analytical p"] + stars_source
    work = work.drop(columns=[c for c in _DROP_COLS if c in work.columns], errors="ignore")
    cols = [c for c in _OUTPUT_ORDER if c in work.columns]
    work = work[cols]
    return work

def export_regression_table(
    df: pd.DataFrame,
    title: str,
    output_filename: str,
    *,
    sort_by: Sequence[str] | None = None,
    output_format: str = "tex",
    latex_env: str = "tabular",
    include_caption_label: bool = False,
    font_size_cmd: str = "scriptsize",
    tabcolsep_pt: float = 2.0,
    coef_digits: Optional[int] = 3,
    p_digits: int = 3,
) -> Union[Path, Tuple[Path, Path]]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")
    if output_format not in {"tex", "html"}:
        raise ValueError("output_format must be 'tex' or 'html'")
    if output_format == "tex" and latex_env not in {"tabular", "longtable"}:
        raise ValueError("latex_env must be 'tabular' or 'longtable'")
    outdir = _project_tables_dir()
    outdir.mkdir(parents=True, exist_ok=True)
    if not output_filename.endswith(f".{output_format}"):
        output_filename = f"{output_filename}.{output_format}"
    out_path = outdir / output_filename
    work = _prep_dataframe(df, coef_digits=coef_digits, p_digits=p_digits)
    if sort_by:
        missing = [c for c in sort_by if c not in work.columns]
        if not missing:
            work = work.sort_values(list(sort_by)).reset_index(drop=True)
    if output_format == "tex":
        obj_cols = list(work.select_dtypes(include="object").columns)
        for c in obj_cols:
            if c == "AINI^var":
                continue
            work[c] = work[c].apply(_escape_latex_text)
        colspec = _column_format(list(work.columns))
        header_cols = [_sanitize_latex_header(c) for c in work.columns]
        work.columns = header_cols
        preamble = f"\\setlength\\tabcolsep{{{tabcolsep_pt}pt}}\n\\{font_size_cmd}\n\n"
        if latex_env == "tabular":
            latex_core = work.to_latex(
                index=False, escape=False, na_rep="",
                column_format=colspec,
                bold_rows=False, multicolumn=False, longtable=False
            )
            content = preamble + latex_core
            out_path.write_text(content, encoding="utf-8")
            return out_path
        kwargs = dict(
            index=False, escape=False, na_rep="",
            column_format=colspec,
            bold_rows=False, multicolumn=False, longtable=True
        )
        if include_caption_label:
            kwargs["caption"] = _escape_caption(title)
            kwargs["label"] = f"tab:{out_path.stem}"
        latex_core = work.to_latex(**kwargs)
        content = preamble + latex_core
        out_path.write_text(content, encoding="utf-8")
        return out_path
    html_core = (
        work.style
        .set_caption(title)
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
{html_core}
</body>
</html>
"""
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path
