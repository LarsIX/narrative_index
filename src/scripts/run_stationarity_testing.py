#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CLI for stationarity testing (ADF, PP, KPSS) on:
- AINI variants (multiple datasets via --aini)
- Financial variables (by ticker)
- VIX z-scores
- SOX & GSPC log-returns (Δ log price)

Outputs: CSV to data/processed/variables, HTML to reports/tables.
A run_id timestamp is appended to labels to avoid overwrites.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import pandas as pd
import typer

# --- path bootstrap ---
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]  # AI_narrative_index/
SRC_ROOT = PROJECT_ROOT / "src"
for p in (PROJECT_ROOT, SRC_ROOT):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

# --- imports from modelling module ---
try:
    from src.modelling.stationarity_testing import (
        test_stationarity_aini_variants,
        test_stationarity_fin_variables,
        test_stationarity_vix_zscores,
        run_stationarity_sox_gspc,
    )
except Exception as e:
    raise ImportError(
        f"Failed to import stationarity functions: {e}\n"
        f"Checked paths:\n- {PROJECT_ROOT}\n- {SRC_ROOT}\n"
        "Run from project root or set PYTHONPATH accordingly."
    )

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)

# --- helpers ---
def _default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _parse_csv_list(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    return [part.strip() for part in s.split(",") if part.strip()]

def _parse_kpss_nlags(value: str) -> str | int:
    if value.lower() == "auto":
        return "auto"
    try:
        return int(value)
    except ValueError:
        raise typer.BadParameter("kpss-nlags must be 'auto' or an integer")

def _parse_aini_spec(spec: str) -> dict:
    """
    Parse AINI spec like:
      csv=data/processed/variables/aini_w0.csv,variants=aini_w0,window=0,cols=normalized_AINI|EMA_02|EMA_08
    Returns: dict(csv: Path, variants: str, window: Optional[int], cols: List[str])
    """
    if not spec or "=" not in spec:
        raise typer.BadParameter("Invalid --aini spec. Use key=value pairs separated by commas.")

    parts: dict[str, str] = {}
    for chunk in [c.strip() for c in spec.split(",") if c.strip()]:
        if "=" not in chunk:
            raise typer.BadParameter(f"Invalid token in --aini spec: '{chunk}' (expected key=value).")
        k, v = chunk.split("=", 1)
        parts[k.strip().lower()] = v.strip()

    if "csv" not in parts or "variants" not in parts:
        raise typer.BadParameter("Each --aini must include at least csv=... and variants=...")

    csv_path = Path(parts["csv"])
    if not csv_path.exists():
        raise typer.BadParameter(f"AINI csv not found: {csv_path}")

    window = None
    if "window" in parts and parts["window"] != "":
        try:
            window = int(parts["window"])
        except ValueError:
            raise typer.BadParameter("window must be an integer if provided.")

    if "cols" in parts and parts["cols"]:
        cols = [c.strip() for c in parts["cols"].split("|") if c.strip()]
    else:
        cols = ["normalized_AINI", "EMA_02", "EMA_08"]

    return {"csv": csv_path, "variants": parts["variants"], "window": window, "cols": cols}

# --- commands ---
@app.command(help="Run stationarity tests for a single AINI dataset (ADF, PP, KPSS).")
def aini(
    aini_csv: Path = typer.Option(..., exists=True, readable=True, help="Path to AINI CSV with a 'date' column."),
    variants: str = typer.Option("aini", help="Label used in filenames."),
    window: Optional[int] = typer.Option(None, help="Context window label."),
    aini_cols: Optional[str] = typer.Option(None, help="Comma-separated columns e.g. 'normalized_AINI,EMA_02,EMA_08'."),
    alpha: float = typer.Option(0.05),
    min_obs: int = typer.Option(20),
    kpss_regression: str = typer.Option("c", help="'c' (level) or 'ct' (trend)"),
    kpss_nlags: str = typer.Option("auto", help="'auto' or integer"),
    run_id: Optional[str] = typer.Option(None),
):
    run_id = run_id or _default_run_id()
    cols = _parse_csv_list(aini_cols)
    nlags = _parse_kpss_nlags(kpss_nlags)

    df = pd.read_csv(aini_csv)
    if "date" not in df.columns:
        raise typer.BadParameter("AINI CSV must contain a 'date' column.")

    typer.echo(f"[AINI] variants={variants}_{run_id} window={window} cols={cols}")
    out = test_stationarity_aini_variants(
        aini_data=df,
        variants=f"{variants}_{run_id}",
        window=window,
        aini_cols=cols,
        alpha=alpha,
        min_obs=min_obs,
        kpss_regression=kpss_regression,
        kpss_nlags=nlags,
    )
    typer.echo(f"[AINI] Done. Rows: {len(out)}")

@app.command(help="Run stationarity tests for financial variables by ticker (ADF, PP, KPSS).")
def fin(
    fin_csv: Path = typer.Option(..., exists=True, readable=True, help="Path to financial CSV."),
    fin_vars: Optional[str] = typer.Option(None, help="Comma-separated variable names, e.g. 'Adj Close,LogReturn'."),
    alpha: float = typer.Option(0.05),
    min_obs: int = typer.Option(20),
    date_col: str = typer.Option("Date", help="Date column name in fin_csv."),
    ticker_col: str = typer.Option("Ticker", help="Ticker column name in fin_csv."),
    label: str = typer.Option("fin_var", help="Filename label base."),
    kpss_regression: str = typer.Option("c", help="'c' (level) or 'ct' (trend)"),
    kpss_nlags: str = typer.Option("auto", help="'auto' or integer"),
    run_id: Optional[str] = typer.Option(None),
):
    run_id = run_id or _default_run_id()
    vars_list = _parse_csv_list(fin_vars)
    nlags = _parse_kpss_nlags(kpss_nlags)

    df = pd.read_csv(fin_csv)
    if date_col not in df.columns:
        raise typer.BadParameter(f"'{date_col}' not found in {fin_csv}.")
    if ticker_col not in df.columns:
        raise typer.BadParameter(f"'{ticker_col}' not found in {fin_csv}.")

    typer.echo(f"[FIN] label={label}_{run_id} vars={vars_list}")
    out = test_stationarity_fin_variables(
        fin_data=df,
        fin_vars=vars_list,
        alpha=alpha,
        min_obs=min_obs,
        date_col=date_col,
        ticker_col=ticker_col,
        label=f"{label}_{run_id}",
        kpss_regression=kpss_regression,
        kpss_nlags=nlags,
    )
    typer.echo(f"[FIN] Done. Rows: {len(out)}")

@app.command(help="Run stationarity tests for VIX z-scores (log_growth_closed).")
def vix(
    vix_csv: Optional[Path] = typer.Option(
        None, exists=True, readable=True,
        help="Path to z_scores_VIX.csv. If omitted, the modelling function's default path is used."
    ),
    date_col: str = typer.Option("date"),
    value_col: str = typer.Option("log_growth_closed"),
    alpha: float = typer.Option(0.05),
    min_obs: int = typer.Option(20),
    kpss_regression: str = typer.Option("c"),
    kpss_nlags: str = typer.Option("auto"),
    label: str = typer.Option("vix_zscores"),
    run_id: Optional[str] = typer.Option(None),
):
    run_id = run_id or _default_run_id()
    nlags = _parse_kpss_nlags(kpss_nlags)

    typer.echo(f"[VIX] label={label}_{run_id}")
    out = test_stationarity_vix_zscores(
        vix_path=vix_csv,
        date_col=date_col,
        value_col=value_col,
        alpha=alpha,
        min_obs=min_obs,
        kpss_regression=kpss_regression,
        kpss_nlags=nlags,
        label=f"{label}_{run_id}",
    )
    typer.echo(f"[VIX] Done. Rows: {len(out)}")

@app.command("sox-gspc", help="Run stationarity tests for SOX & GSPC log returns (Δ log price).")
def sox_gspc(
    alpha: float = typer.Option(0.05),
    min_obs: int = typer.Option(20),
    kpss_regression: str = typer.Option("c"),
    kpss_nlags: str = typer.Option("auto"),
    label: str = typer.Option("sox_gspc_logrets"),
    run_id: Optional[str] = typer.Option(None),
):
    run_id = run_id or _default_run_id()
    nlags = _parse_kpss_nlags(kpss_nlags)

    typer.echo(f"[SOX-GSPC] label={label}_{run_id}")
    out = run_stationarity_sox_gspc(
        alpha=alpha,
        min_obs=min_obs,
        kpss_regression=kpss_regression,
        kpss_nlags=nlags,
        label=f"{label}_{run_id}",
    )
    typer.echo(f"[SOX-GSPC] Done. Rows: {len(out)}")

@app.command("run-all", help="Run multiple AINI specs plus optional FIN, VIX, and SOX/GSPC.")
def run_all(
    aini: List[str] = typer.Option(
        None,
        help=("Repeatable AINI spec: 'csv=PATH,variants=LABEL,window=INT,cols=col1|col2|col3'. "
              "window/cols optional; defaults to normalized_AINI|EMA_02|EMA_08."),
    ),
    # FIN (optional)
    fin_csv: Optional[Path] = typer.Option(None, exists=True, readable=True, help="Path to financial CSV."),
    fin_vars: Optional[str] = typer.Option(None, help="Comma-separated FIN vars (e.g., 'Adj Close,LogReturn')."),
    date_col: str = typer.Option("Date", help="FIN date column."),
    ticker_col: str = typer.Option("Ticker", help="FIN ticker column."),
    label_fin: str = typer.Option("fin_var"),
    # VIX (optional)
    vix_csv: Optional[Path] = typer.Option(None, exists=True, readable=True, help="Path to z_scores_VIX.csv."),
    label_vix: str = typer.Option("vix_zscores"),
    # SOX/GSPC toggle
    include_sox_gspc: bool = typer.Option(True, help="Also run SOX & GSPC log-returns tests."),
    label_sox_gspc: str = typer.Option("sox_gspc_logrets"),
    # Shared/statistics
    alpha: float = typer.Option(0.05),
    min_obs: int = typer.Option(20),
    kpss_regression: str = typer.Option("c"),
    kpss_nlags: str = typer.Option("auto"),
    # overwrite protection
    run_id: Optional[str] = typer.Option(None),
):
    run_id = run_id or _default_run_id()
    nlags = _parse_kpss_nlags(kpss_nlags)
    did_any = False

    # AINI(s)
    if aini:
        for i, spec in enumerate(aini, start=1):
            parsed = _parse_aini_spec(spec)
            df_aini = pd.read_csv(parsed["csv"])
            if "date" not in df_aini.columns:
                raise typer.BadParameter(f"[AINI #{i}] CSV must contain a 'date' column: {parsed['csv']}")
            label = f"{parsed['variants']}_{run_id}"
            typer.echo(f"[RUN-ALL/AINI #{i}] variants={label} window={parsed['window']} cols={parsed['cols']}")
            test_stationarity_aini_variants(
                aini_data=df_aini,
                variants=label,
                window=parsed["window"],
                aini_cols=parsed["cols"],
                alpha=alpha,
                min_obs=min_obs,
                kpss_regression=kpss_regression,
                kpss_nlags=nlags,
            )
            did_any = True

    # FIN
    if fin_csv is not None:
        vars_list = _parse_csv_list(fin_vars)
        df_fin = pd.read_csv(fin_csv)
        if date_col not in df_fin.columns:
            raise typer.BadParameter(f"'{date_col}' not found in FIN CSV.")
        if ticker_col not in df_fin.columns:
            raise typer.BadParameter(f"'{ticker_col}' not found in FIN CSV.")
        typer.echo(f"[RUN-ALL/FIN] label={label_fin}_{run_id} vars={vars_list}")
        test_stationarity_fin_variables(
            fin_data=df_fin,
            fin_vars=vars_list,
            alpha=alpha,
            min_obs=min_obs,
            date_col=date_col,
            ticker_col=ticker_col,
            label=f"{label_fin}_{run_id}",
            kpss_regression=kpss_regression,
            kpss_nlags=nlags,
        )
        did_any = True

    # VIX (always offered; default internal path if vix_csv=None)
    typer.echo(f"[RUN-ALL/VIX] label={label_vix}_{run_id}")
    test_stationarity_vix_zscores(
        vix_path=vix_csv,
        alpha=alpha,
        min_obs=min_obs,
        kpss_regression=kpss_regression,
        kpss_nlags=nlags,
        label=f"{label_vix}_{run_id}",
    )
    did_any = True

    # SOX/GSPC
    if include_sox_gspc:
        typer.echo(f"[RUN-ALL/SOX-GSPC] label={label_sox_gspc}_{run_id}")
        run_stationarity_sox_gspc(
            alpha=alpha,
            min_obs=min_obs,
            kpss_regression=kpss_regression,
            kpss_nlags=nlags,
            label=f"{label_sox_gspc}_{run_id}",
        )
        did_any = True

    if not did_any:
        raise typer.BadParameter("Nothing to run. Provide at least one --aini, --fin-csv, --vix-csv, or enable SOX/GSPC.")
    typer.echo("[RUN-ALL] Completed.")

if __name__ == "__main__":
    app()


