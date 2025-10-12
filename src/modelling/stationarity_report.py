# stationarity_report.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron


# ---------------- utilities ---------------- #
def _clean_series(x: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(x, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )


def _run_adf_pp_kpss(
    series: pd.Series,
    kpss_regression: str = "c",
    kpss_nlags: str | int = "auto",
) -> Tuple[float, float, float, float, float, float]:
    s = _clean_series(series)
    adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
    pp = PhillipsPerron(s)
    pp_stat = float(pp.stat)
    pp_p = float(pp.pvalue)
    try:
        kpss_stat, kpss_p, *_ = kpss(s, regression=kpss_regression, nlags=kpss_nlags)
    except Exception:
        kpss_stat, kpss_p = np.nan, np.nan
    return adf_stat, adf_p, pp_stat, pp_p, kpss_stat, kpss_p


def _agree_stationary(adf_p: float, pp_p: float, kpss_p: float, alpha: float) -> bool:
    # Conservative: ADF & PP reject unit root, KPSS does NOT reject stationarity
    if any(pd.isna([adf_p, pp_p, kpss_p])):
        return False
    return (adf_p < alpha) and (pp_p < alpha) and (kpss_p > alpha)


def _subset_periods(df: pd.DataFrame, date_col: str) -> Dict[str, pd.DataFrame]:
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    return {
        "2023": d[d[date_col] < "2024-01-01"],
        "2024": d[(d[date_col] >= "2024-01-01") & (d[date_col] < "2025-01-01")],
        "2025": d[d[date_col] >= "2025-01-01"],
        "2023-2024": d[d[date_col] < "2025-01-01"],
        "2024-2025": d[d[date_col] >= "2024-01-01"],
        "2023-2025": d,
    }


# ---------------- core ---------------- #
def build_stationarity_html(
    df: pd.DataFrame,
    output_html: Path | str,
    date_col: str = "Date",
    measure_col: str = "Measure",
    value_col: str = "Value",
    alpha: float = 0.05,
    min_obs: int = 20,
    include_period_splits: bool = True,
    kpss_regression: str = "c",
    kpss_nlags: str | int = "auto",
    title: str = "Stationarity Tests by Measure (ADF / PP / KPSS)",
    save_combined_csv: Optional[Path | str] = None,
) -> pd.DataFrame:
    """
    Group by `measure_col`, sort by `date_col`, run ADF/PP/KPSS on `value_col`.
    Writes ONE single HTML report with a section per measure (and optional period splits).
    Returns the combined results DataFrame.

    Parameters are defensive and minimal; your notebook just constructs the tidy df and calls this.
    """
    required = {date_col, measure_col, value_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input df missing required columns: {sorted(missing)}")

    d0 = df.copy()
    d0[date_col] = pd.to_datetime(d0[date_col], errors="coerce")
    d0 = d0.dropna(subset=[date_col]).sort_values([measure_col, date_col]).reset_index(drop=True)

    all_rows: List[dict] = []
    sections_html: List[str] = []
    toc_items: List[str] = []

    # CSS (kept lightweight)
    css = """
    <style>
      body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }
      h1 { margin-bottom: 6px; }
      .meta { color: #666; margin-bottom: 18px; }
      h2 { margin-top: 28px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
      table { border-collapse: collapse; font-size: 14px; }
      th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: right; }
      th { background: #f7f7f7; }
      td:first-child, th:first-child, td:nth-child(2), th:nth-child(2) { text-align: left; }
      .ok { background: #e8f5e9; }  /* green-ish */
      .bad { background: #ffebee; } /* red-ish */
      .warn { background: #fff8e1; } /* yellow-ish */
      .small { font-size: 12px; color: #666; }
      .toc { margin: 12px 0 24px 0; padding-left: 16px; }
      .toc li { margin: 2px 0; }
      .foot { margin-top: 24px; color: #888; font-size: 12px; }
    </style>
    """

    for measure, g in d0.groupby(measure_col, dropna=False):
        g = g.dropna(subset=[value_col])
        g = g.sort_values(date_col)
        measure_name = str(measure) if pd.notna(measure) else "(NA)"

        # Which slices to test?
        slices = {"Full": g}
        if include_period_splits:
            slices.update(_subset_periods(g, date_col=date_col))

        rows_measure: List[dict] = []
        for slice_name, sub in slices.items():
            s = _clean_series(sub[value_col])
            n = int(s.shape[0])
            if n < min_obs:
                rows_measure.append({
                    "Measure": measure_name,
                    "Period": slice_name,
                    "ADF_stat": np.nan, "ADF_p": np.nan,
                    "PP_stat": np.nan, "PP_p": np.nan,
                    "KPSS_stat": np.nan, "KPSS_p": np.nan,
                    "agree_stationarity": False,
                    "error": f"too_few_obs(<{min_obs})",
                })
                continue

            try:
                adf_stat, adf_p, pp_stat, pp_p, kpss_stat, kpss_p = _run_adf_pp_kpss(
                    s, kpss_regression=kpss_regression, kpss_nlags=kpss_nlags
                )
                rows_measure.append({
                    "Measure": measure_name,
                    "Period": slice_name,
                    "ADF_stat": adf_stat, "ADF_p": adf_p,
                    "PP_stat": pp_stat, "PP_p": pp_p,
                    "KPSS_stat": kpss_stat, "KPSS_p": kpss_p,
                    "agree_stationarity": _agree_stationary(adf_p, pp_p, kpss_p, alpha),
                    "error": "",
                })
            except Exception as e:
                rows_measure.append({
                    "Measure": measure_name,
                    "Period": slice_name,
                    "ADF_stat": np.nan, "ADF_p": np.nan,
                    "PP_stat": np.nan, "PP_p": np.nan,
                    "KPSS_stat": np.nan, "KPSS_p": np.nan,
                    "agree_stationarity": False,
                    "error": str(e),
                })

        # Collect
        all_rows.extend(rows_measure)

        # Build measure table HTML with light highlighting
        tbl = pd.DataFrame(rows_measure).sort_values(["Measure", "Period"]).reset_index(drop=True)

        def _row_class(r):
            if r["error"]:
                return "warn"
            return "ok" if r["agree_stationarity"] else "bad"

        # Manually build HTML table for styling
        headers = ["Measure", "Period", "ADF_stat", "ADF_p", "PP_stat", "PP_p", "KPSS_stat", "KPSS_p", "agree_stationarity", "error"]
        thead = "<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
        body_rows = []
        for _, r in tbl.iterrows():
            cls = _row_class(r)
            tds = "".join(
                f"<td>{'' if pd.isna(r[h]) else (f'{r[h]:.6g}' if isinstance(r[h], (int, float, np.floating)) else r[h])}</td>"
                for h in headers
            )
            body_rows.append(f"<tr class='{cls}'>{tds}</tr>")
        table_html = f"<table><thead>{thead}</thead><tbody>{''.join(body_rows)}</tbody></table>"

        anchor = f"measure-{abs(hash(measure_name))}"
        toc_items.append(f"<li><a href='#{anchor}'>{measure_name}</a></li>")
        sections_html.append(f"<h2 id='{anchor}'>{measure_name}</h2>\n{table_html}")

    combined = pd.DataFrame(all_rows).sort_values(["Measure", "Period"]).reset_index(drop=True)

    # Serialize HTML
    joined_sections = "\n\n".join(sections_html)

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>{title}</title>
{css}
</head>
<body>
<h1>{title}</h1>
<div class="meta">
  alpha = {alpha}, KPSS regression = '{kpss_regression}', KPSS nlags = '{kpss_nlags}', min_obs = {min_obs}.
</div>
<div class="small">
  Legend: <span class="ok">OK</span> = ADF & PP reject unit root and KPSS does not reject stationarity;
  <span class="bad">BAD</span> = fails consensus;
  <span class="warn">WARN</span> = error/too few obs.
</div>

<h3>Measures</h3>
<ul class="toc">
  {''.join(toc_items)}
</ul>

{joined_sections}

<div class="foot"> Source: Own</div>
</body>
</html>"""



    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")

    if save_combined_csv:
        Path(save_combined_csv).parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(save_combined_csv, index=False)

    return combined
