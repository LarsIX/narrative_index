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

from modelling.estimate_ols import run_sameday_ols_mbboot_fdr  # noqa: E402

app = typer.Typer()


def infer_aini_file_from_version(version: str) -> Path:
    base = src_dir.parent / "data" / "processed" / "variables"
    return base / f"{version}_AINI_variables.csv"


def load_financial_data_by_year() -> dict:
    fin_path = src_dir.parent / "data" / "raw" / "financial" / "full_daily_2023_2025.csv"
    df = pd.read_csv(fin_path)
    df["date"] = pd.to_datetime(df["date"])
    return {str(y): g.reset_index(drop=True) for y, g in df.groupby(df["date"].dt.year)}


@app.command()
def run(
    version: str = typer.Option(..., help="AINI version to use: w0, w1, w2, binary"),
    aini_file: Optional[Path] = typer.Option(None, help="Explicit path to AINI CSV (optional)."),
    outdir: Optional[Path] = typer.Option(None, help="Output directory (optional)."),
    aini_variants: Optional[str] = typer.Option(None, help="Comma-separated AINI variants to include."),
    control_vars: Optional[str] = typer.Option(None, help="Comma-separated control vars (must be in financial data)."),
    min_obs: int = 30,
    n_boot: int = 1000,
    block_size: int = 5,
    fdr_alpha: float = 0.05,
    cov: str = "HAC",
    hac_lags: Optional[int] = None,
    seed: int = 42,
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
    )


@app.command("run-all")
def run_all_versions(
    versions: List[str] = typer.Option(["w0", "w1", "w2", "binary"], help="List of AINI versions to run."),
    **kwargs,
):
    """Run same-day OLS bootstrap for multiple versions in sequence."""
    for version in versions:
        typer.echo(f"\n[=== Running version: {version} ===]")
        run(version=version, **kwargs)


if __name__ == "__main__":
    app()
