"""
Same-day OLS CLI Runner
-----------------------

This module provides Typer CLI commands to estimate same-day OLS regressions
with Moving Block Bootstrap p-values and Benjamini–Hochberg FDR correction.

Commands
--------
- run       : Run estimation for a single AINI version (e.g., w0, w1, w2, binary).
- run-all   : Run estimation sequentially for the fixed versions ['w0', 'w1', 'w2'].

Key Options
-----------
- --outdir           : Output directory (highly recommended to set explicitly).
- --save-csv/--no-save-csv : Write CSV output (default: save).
- --n-boot           : Number of bootstrap replications (e.g., 10000).
- --aini-variants    : Comma-separated AINI variants to include.
- --control-vars     : Comma-separated control vars from financial data.

Example
-------
python run_estimate_ols.py run-all --n-boot 10000
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List

import typer
import pandas as pd

# Make internal imports work
this_file = Path(__file__).resolve()
src_dir = this_file.parent.parent
sys.path.append(str(src_dir))

from modelling.estimate_OLS import run_sameday_ols_mbboot_fdr  # noqa: E402

app = typer.Typer()


def infer_aini_file_from_version(version: str) -> Path:
    base = src_dir.parent / "data" / "processed" / "variables"
    return base / f"{version}_AINI_variables.csv"


def load_financial_data_by_year() -> dict:
    fin_path = src_dir.parent / "data" / "raw" / "financial" / "full_daily_2023_2025.csv"
    df = pd.read_csv(fin_path)
    df["date"] = pd.to_datetime(df["Date"])
    return {str(y): g.reset_index(drop=True) for y, g in df.groupby(df["date"].dt.year)}


@app.command()
def run(
    version: str = typer.Option(
        "w0",
        help="AINI version to use (e.g., w0, w1, w2).",
        show_default=True,
    ),
    aini_file: Optional[Path] = typer.Option(None, help="Explicit path to AINI CSV (optional)."),
    outdir: Optional[Path] = typer.Option(None, help="Output directory (optional)."),
    aini_variants: Optional[str] = typer.Option(None, help="Comma-separated AINI variants to include."),
    control_vars: Optional[str] = typer.Option(None, help="Comma-separated control vars (must be in financial data)."),
    min_obs: int = typer.Option(30, help="Minimum number of observations."),
    n_boot: int = typer.Option(1000, help="Number of bootstrap replications."),
    block_size: int = typer.Option(5, help="Block size for Moving Block Bootstrap."),
    fdr_alpha: float = typer.Option(0.05, help="FDR alpha (Benjamini–Hochberg)."),
    cov: str = typer.Option("HAC", help="Covariance for analytic p-values: 'HAC' or 'HC3'."),
    hac_lags: Optional[int] = typer.Option(None, help="HAC lags; auto if None."),
    seed: int = typer.Option(42, help="Random seed."),
    save_csv: bool = typer.Option(True, "--save-csv/--no-save-csv", help="Write CSV output."),
    outname_base: Optional[str] = typer.Option(
        None, help="Optional base filename; version suffix is appended automatically."
    ),
):
    """Run same-day OLS bootstrap estimation for a single version."""
    typer.echo(f"[INFO] Running version: {version}")

    # Resolve AINI file
    if aini_file is None:
        aini_file = infer_aini_file_from_version(version)
    typer.echo(f"[INFO] Using AINI file: {aini_file}")
    if not aini_file.exists():
        raise FileNotFoundError(aini_file)

    aini_df = pd.read_csv(aini_file)
    fin_data = load_financial_data_by_year()

    variants = [v.strip() for v in aini_variants.split(",")] if aini_variants else None
    controls = [c.strip() for c in control_vars.split(",")] if control_vars else None

    # Versionierter Dateiname, damit run-all nichts überschreibt
    base_name = outname_base or "ols_sameday_mbboot_fdr"
    outname = f"{base_name}_{version}.csv"
    if controls:
        # Markiere „controlled“
        outname = outname.replace(".csv", "_controlled.csv")

    run_sameday_ols_mbboot_fdr(
        aini_df=aini_df,
        fin_data_by_year=fin_data,
        aini_variants=variants,
        control_vars=controls,
        min_obs=min_obs,
        block_size=block_size,
        n_boot=n_boot,
        seed=seed,
        fdr_alpha=fdr_alpha,
        cov_for_analytic=cov,
        hac_lags=hac_lags,
        outdir=outdir,
        save_csv=save_csv,
        outname=outname,
    )


@app.command("run-all")
def run_all_versions(
    aini_file: Optional[Path] = typer.Option(None, help="Explicit path to AINI CSV (optional)."),
    outdir: Optional[Path] = typer.Option(None, help="Output directory (optional)."),
    aini_variants: Optional[str] = typer.Option(None, help="Comma-separated AINI variants to include."),
    control_vars: Optional[str] = typer.Option(None, help="Comma-separated control vars (must be in financial data)."),
    min_obs: int = typer.Option(30, help="Minimum number of observations."),
    n_boot: int = typer.Option(1000, help="Number of bootstrap replications."),
    block_size: int = typer.Option(5, help="Block size for Moving Block Bootstrap."),
    fdr_alpha: float = typer.Option(0.05, help="FDR alpha (Benjamini–Hochberg)."),
    cov: str = typer.Option("HAC", help="Covariance for analytic p-values: 'HAC' or 'HC3'."),
    hac_lags: Optional[int] = typer.Option(None, help="HAC lags; auto if None."),
    seed: int = typer.Option(42, help="Random seed."),
    save_csv: bool = typer.Option(True, "--save-csv/--no-save-csv", help="Write CSV output."),
    outname_base: Optional[str] = typer.Option(
        None, help="Optional base filename; version suffix is appended automatically."
    ),
):
    """Run same-day OLS bootstrap for w0, w1, w2 in sequence."""
    versions = ["w0", "w1", "w2"]
    for version in versions:
        typer.echo(f"\n[=== Running version: {version} ===]")
        run(
            version=version,
            aini_file=aini_file,
            outdir=outdir,
            aini_variants=aini_variants,
            control_vars=control_vars,
            min_obs=min_obs,
            n_boot=n_boot,
            block_size=block_size,
            fdr_alpha=fdr_alpha,
            cov=cov,
            hac_lags=hac_lags,
            seed=seed,
            save_csv=save_csv,
            outname_base=outname_base,
        )


if __name__ == "__main__":
    app()
