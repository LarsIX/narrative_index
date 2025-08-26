"""
CLI for estimating Granger causality (both directions) with Wild residual bootstrap and BH-FDR.

- Supports a lag *range* for AINI lags, e.g. --p-x-range 1,3 runs p_x in {1,2,3}.
- Defaults: n_boot=5000, fdr_alpha=0.1, min_obs=0, HAC covariance with auto NW lags.
- AINI file inferred from version (w0,w1,w2,binary).
- Controls: must pass --control-var (string), plus optional --controls-file/--controls-lags.

The modelling function writes its own CSV:
    granger_causality_{control_var}_{version}.csv

Example usage:
python run_estimate_granger_causality.py run-all-versions --versions "binary,w0,w1,w2" --control-var n_articles --controls-file data/processed/variables/n_articles.csv --controls-lags n_articles --p-x-range 1,3 --controls-align-with-px --n-boot 1 --min-obs 0 --outdir data/processed/variables


"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import pandas as pd
import typer

this_file = Path(__file__).resolve()
scripts_dir = this_file.parent
src_dir = scripts_dir.parent
root_dir = src_dir.parent # to load csv
sys.path.append(str(src_dir))

from modelling.estimate_granger_causality import run_gc_mbboot_fdr, run_gc_mbboot_fdr_controls  # noqa: E402

app = typer.Typer(help="Run Granger Causality with wild residual bootstrap and BH-FDR.")

# ----------------------
# Helpers
# ----------------------

def parse_range(spec: Optional[str]) -> Optional[Tuple[int, int]]:
    if spec is None:
        return None
    s = spec.strip()
    if not s:
        return None
    parts = s.split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid lag range '{spec}'. Use 'start,end'.")
    a, b = int(parts[0]), int(parts[1])
    if a > b:
        a, b = b, a
    return a, b


def parse_controls_lags(spec: Optional[str]) -> Optional[Dict[str, int]]:
    """
    Parse controls lags like:
      "n_articles"  -> {"n_articles": 1}
    If --controls-align-with-px is set, the numeric parts are ignored downstream.
    """
    if spec is None:
        return None
    s = spec.strip()
    if not s:
        return None

    out = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            name, lag = [x.strip() for x in part.split(":", 1)]
            out[name] = int(lag)
        else:
            # no lag specified, default to 1 (overridden if controls_align_with_px=True)
            out[part] = 1
    return out



def infer_aini_file_from_version(version: str) -> Path:
    base = src_dir.parent / "data" / "processed" / "variables"
    return base / f"{version}_AINI_variables.csv"


# ----------------------
# CLI
# ----------------------
@app.command()
def run(
    version: str = typer.Option(...),
    control_var: str = typer.Option(...),
    controls_file: Optional[Path] = None,
    controls_lags: Optional[str] = None,
    aini_file: Optional[Path] = None,
    p_ret: int = 1,
    p_x: int = 3,
    p_x_range: Optional[str] = "1,3",
    n_boot: int = 5000,
    weight_dist: str = "rademacher",
    fdr_alpha: float = 0.10,
    min_obs: int = 0,
    cov_for_analytic: str = "HAC",
    hac_lags: Optional[int] = None,
    aini_variants: Optional[str] = None,
    outdir: Optional[Path] = None,
    seed: int = 42,
    n_jobs: int = -1,
    controls_align_with_px: bool = typer.Option(False, "--controls-align-with-px"),
):
    typer.echo(f"[INFO] Running version={version} with control_var={control_var}")

    # --- Load data ---
    aini_path = infer_aini_file_from_version(version) if aini_file is None else aini_file
    if not aini_path.exists():
        raise FileNotFoundError(f"AINI file not found: {aini_path}")
    aini_df = pd.read_csv(aini_path)

    fin_path = src_dir.parent / "data" / "raw" / "financial" / "full_daily_2023_2025.csv"
    fin_data = pd.read_csv(fin_path)

    # --- Variants & lag range ---
    variants_list = [v.strip() for v in aini_variants.split(",")] if aini_variants else None
    lr = parse_range(p_x_range)
    lag_values = list(range(lr[0], lr[1] + 1)) if lr else [p_x]
    typer.echo(f"[INFO] Lag values: {lag_values}")

    # --- Controls ---

    ctrl_df = pd.read_csv(root_dir / controls_file) if controls_file else None
    ctrl_lags_map = parse_controls_lags(controls_lags) if controls_lags else None
    use_controls = (ctrl_df is not None) and (ctrl_lags_map is not None)

    frames = []
    for px in lag_values:
        typer.echo(f"[RUN ] Estimating with p_x={px}")

        ctrl_lags_eff = ({name: px for name in ctrl_lags_map.keys()}
                         if (use_controls and controls_align_with_px)
                         else ctrl_lags_map)

        if use_controls:
            df = run_gc_mbboot_fdr_controls(
                aini_df=aini_df, fin_data=fin_data,
                version=version, control_var=control_var,
                controls_df=ctrl_df, controls_lags=ctrl_lags_eff,
                aini_variants=variants_list, p_ret=p_ret, p_x=px,
                min_obs=min_obs, n_boot=n_boot, seed=seed,
                fdr_alpha=fdr_alpha, cov_for_analytic=cov_for_analytic,
                hac_lags=hac_lags, weight_dist=weight_dist,
                save_csv=False, outdir=outdir, outname=None, n_jobs=n_jobs,
            )
        else:
            df = run_gc_mbboot_fdr(
                aini_df=aini_df, fin_data=fin_data,
                version=version, aini_variants=variants_list,
                p_ret=p_ret, p_x=px, min_obs=min_obs, n_boot=n_boot,
                seed=seed, fdr_alpha=fdr_alpha, cov_for_analytic=cov_for_analytic,
                hac_lags=hac_lags, weight_dist=weight_dist,
                save_csv=False, outdir=outdir, outname=None, n_jobs=n_jobs,
            )

        df["p_x"] = px
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    typer.echo(f"[OK] Combined results shape: {combined.shape}")

    # --- Save exactly one CSV ---
    if outdir is None:
        outdir = src_dir.parent / "data" / "processed" / "variables"
    outdir.mkdir(parents=True, exist_ok=True)

    outname = f"granger_causality_{control_var}_{version}.csv" if use_controls else f"granger_causality_{version}.csv"
    outpath = outdir / outname
    combined.to_csv(outpath, index=False)
    typer.echo(f"[SAVE] {outpath}")

@app.command("run-all-versions")
def run_all_versions(
    versions: List[str] = typer.Option(["binary","w0","w1","w2"]),
    control_var: str = typer.Option(..., help="Name of control variable."),
    controls_file: Optional[Path] = None,
    controls_lags: Optional[str] = None,
    p_x_range: str = "1,3",
    n_boot: int = 5000,
    weight_dist: str = "rademacher",
    fdr_alpha: float = 0.10,
    min_obs: int = 0,
    cov_for_analytic: str = "HAC",
    hac_lags: Optional[int] = None,
    aini_variants: Optional[str] = None,
    outdir: Optional[Path] = None,
    seed: int = 42,
    n_jobs: int = -1,
    controls_align_with_px: bool = typer.Option(False, "--controls-align-with-px", help="Align controls to same lag as p_x."),
):
    """Run Granger causality estimation for multiple AINI versions in sequence."""
    for v in versions:
        typer.echo(f"\n[=== Running version: {v} ===]")
        run(
            version=v,
            control_var=control_var,
            controls_file=controls_file,
            controls_lags=controls_lags,
            p_x_range=p_x_range,
            n_boot=n_boot,
            weight_dist=weight_dist,
            fdr_alpha=fdr_alpha,
            min_obs=min_obs,
            cov_for_analytic=cov_for_analytic,
            hac_lags=hac_lags,
            aini_variants=aini_variants,
            outdir=outdir,
            seed=seed,
            n_jobs=n_jobs,
            controls_align_with_px=controls_align_with_px,
        )

if __name__ == "__main__":
    app()
