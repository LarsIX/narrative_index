"""
CLI for estimating Granger causality (both directions) with MBB bootstrap and BH-FDR.

- Supports a lag *range* for AINI lags, e.g. --p-x-range 1,3 runs p_x in {1,2,3}.
- Defaults: n_boot=5000, fdr_alpha=0.1, min_obs=0, HAC covariance with auto NW lags.
- AINI data path is inferred from `version`:
    version in {"w0","w1","w2"} -> loads
    \AI_narrative_index\data\processed\variables\<version>_AINI_variables.csv

Usage example:
    # Run for p_x = 1..3 (inclusive), using inferred AINI file for w1
    python run_estimate_granger_causality.py run --version w1 --p-x-range 1,3 --n-boot 10

"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import pandas as pd
import typer

# --- Make 'modelling' importable  ---
this_file = Path(__file__).resolve()
scripts_dir = this_file.parent                      # .../src/scripts
src_dir = scripts_dir.parent                        # .../src
sys.path.append(str(src_dir))

# Modelling entry-point
from modelling.estimate_granger_causality import run_gc_mbboot_fdr  # noqa: E402

app = typer.Typer(help="Run Granger Causality estimation with MBB bootstrap and BH-FDR.")


# ---------- Helpers ----------
def parse_range(spec: Optional[str]) -> Optional[Tuple[int, int]]:
    """
    Parse a string like '1,3' into (1,3). Returns None if spec is None/empty.
    Raises ValueError for bad formats.
    """
    if spec is None:
        return None
    s = spec.strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Invalid range spec '{spec}'. Use 'start,end' (e.g., '1,3').")
    a, b = int(parts[0]), int(parts[1])
    if a < 1 or b < 1:
        raise ValueError("Lag values must be >= 1.")
    if a > b:
        a, b = b, a
    return a, b


def infer_aini_file_from_version(version: str) -> Path:
    """
    Build the AINI file path from the version token ('w0', 'w1', or 'w2').
    """
    base = src_dir.parent / "data" / "processed" / "variables"
    fname = f"{version}_AINI_variables.csv"
    return base / fname

# ---------- CLI ----------
@app.command()
def run(
    version: str = typer.Option(..., help="Window/version token: 'w0', 'w1', or 'w2'. Used to infer the AINI file and in output naming."),
    fin_data_dir: Path = typer.Option(None, help="Directory containing financial data CSVs by year."),
    aini_file: Optional[Path] = typer.Option(None, help="Optional explicit path to AINI CSV. If omitted, it is inferred from --version."),
    p_ret: int = typer.Option(1, min=1, help="Number of return lags."),
    p_x: int = typer.Option(3, min=1, help="Number of AINI lags (used if --p-x-range is not provided)."),
    p_x_range: Optional[str] = typer.Option("1,3", help="Range for AINI lags as 'start,end' (inclusive). Example: '1,3' runs p_x=1,2,3."),
    n_boot: int = typer.Option(5000, min=1, help="Bootstrap replications (MBB)."),
    block_size: int = typer.Option(5, min=1, help="MBB block size."),
    fdr_alpha: float = typer.Option(0.10, help="BH-FDR alpha."),
    min_obs: int = typer.Option(0, min=0, help="Minimum complete observations after lag construction. 0 = no minimum; df checks still apply downstream."),
    cov_for_analytic: str = typer.Option("HAC", help="Covariance for analytic p-values: 'HAC' or 'HC3'."),
    hac_lags: Optional[int] = typer.Option(None, help="HAC maxlags (None = Newey–West rule)."),
    aini_variants: Optional[str] = typer.Option(None, help="Comma-separated AINI variants (e.g., 'normalized_AINI,EMA_08'). Omit to pass None."),
    outdir: Optional[Path] = typer.Option(None, help="Directory to save output CSV. If omitted, modelling default is used."),
    seed: int = typer.Option(42, help="Base RNG seed (internally sub-seeded per bootstrap block)."),
):
    """
    Run Granger causality estimation. If --p-x-range is provided, the tool loops over the
    inclusive lag range and concatenates the results. Output file naming is handled by the
    modelling function (granger_causality_{version}.csv) unless overridden there.
    """
    typer.echo(f"[INFO] Version              : {version}")
    typer.echo(f"[INFO] Financial data dir   : {fin_data_dir}")

    # Resolve or infer AINI file
    if aini_file is None:
        aini_path = infer_aini_file_from_version(version)
        typer.echo(f"[INFO] AINI file (inferred) : {aini_path}")
    else:
        aini_path = aini_file
        typer.echo(f"[INFO] AINI file (explicit) : {aini_path}")

    if not aini_path.exists():
        raise FileNotFoundError(f"AINI file not found: {aini_path}")

    # Load inputs
    typer.echo("[INFO] Loading AINI data...")
    aini_df = pd.read_csv(aini_path)

    typer.echo("[INFO] Loading financial data by year...")

    # Load financial data
    fin_path = src_dir.parent / "data" / "raw" / "financial" 
    fin_data = pd.read_csv(fin_path / "full_daily_2023_2025.csv")

    # Parse variants
    variants_list: Optional[List[str]]
    if aini_variants is None or str(aini_variants).strip() == "":
        variants_list = None  # let the modelling function use its internal default
        typer.echo("[INFO] AINI variants       : None (use modelling defaults)")
    else:
        variants_list = [v.strip() for v in aini_variants.split(",") if v.strip()]
        typer.echo(f"[INFO] AINI variants       : {variants_list}")

    # Parse lag range
    lag_range = parse_range(p_x_range) if p_x_range is not None else None
    if lag_range is None:
        lag_values = [p_x]
        typer.echo(f"[INFO] AINI lag(s)          : {lag_values} (single)")
    else:
        start, end = lag_range
        lag_values = list(range(start, end + 1))
        typer.echo(f"[INFO] AINI lag range       : {start}..{end} -> {lag_values}")

    # Run for each p_x and collect results
    all_results = []
    for px in lag_values:
        typer.echo(f"[RUN ] Estimating with p_x={px} ...")
        df_out = run_gc_mbboot_fdr(
            aini_df=aini_df,
            fin_data=fin_data,
            version=version,
            aini_variants=variants_list,      # None means use modelling defaults
            p_ret=p_ret,
            p_x=px,
            min_obs=min_obs,
            block_size=block_size,
            n_boot=n_boot,
            seed=seed,
            fdr_alpha=fdr_alpha,
            cov_for_analytic=cov_for_analytic,
            hac_lags=hac_lags,
            save_csv=True,                    # keep modelling-side CSV (per version)
            outdir=outdir,
            outname=None                      # modelling uses granger_causality_{version}.csv
        )
        df_out["p_x"] = px  # make lag explicit in the combined DataFrame
        all_results.append(df_out)

    # Concatenate and (optionally) save combined results alongside the modelling CSV
    combined = pd.concat(all_results, ignore_index=True)
    typer.echo(f"[OK  ] Combined results shape: {combined.shape}")

    # If user provided outdir, also write a combined file for convenience
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        combined_name = f"granger_causality_{version}_combined_px.csv"
        combined_path = outdir / combined_name
        combined.to_csv(combined_path, index=False)
        typer.echo(f"[SAVE] Combined CSV written to: {combined_path}")
    else:
        typer.echo("[NOTE] No --outdir given. Combined results not written separately (modelling CSV still saved).")

    typer.echo("[DONE] Granger causality estimation finished.")
    
@app.command("run-all-versions")
def run_all_versions(
    versions: List[str] = typer.Option(["w0", "w1", "w2"], help="Which versions to run."),
    p_ret: int = 1,
    p_x_range: str = "1,3",
    n_boot: int = 5000,
    block_size: int = 5,
    fdr_alpha: float = 0.10,
    min_obs: int = 0,
    cov_for_analytic: str = "HAC",
    hac_lags: Optional[int] = None,
    aini_variants: Optional[str] = None,
    outdir: Optional[Path] = None,
    seed: int = 42,
):
    """
    Run Granger causality estimation for multiple AINI versions in sequence.
    """
    for v in versions:
        typer.echo(f"\n[=== Running version: {v} ===]")
        run(
            version=v,
            fin_data_dir=None,
            aini_file=None,
            p_ret=p_ret,
            p_x=3,  # will be overridden by p_x_range
            p_x_range=p_x_range,
            n_boot=n_boot,
            block_size=block_size,
            fdr_alpha=fdr_alpha,
            min_obs=min_obs,
            cov_for_analytic=cov_for_analytic,
            hac_lags=hac_lags,
            aini_variants=aini_variants,
            outdir=outdir,
            seed=seed,
        )

# CLI entry point
if __name__ == "__main__":
    app()
