"""
CLI for estimating Granger causality (both directions) with Wild residual bootstrap and BH-FDR.

- Supports a lag *range* for AINI lags, e.g. --p-x-range 1,3 runs p_x in {1,2,3}.
- Defaults: n_boot=5000, fdr_alpha=0.1, min_obs=0, HAC covariance with auto NW lags.
- AINI file inferred from version (w0,w1,w2,w3,binary).
- Controls: must pass --control-var (string), plus optional --controls-file/--controls-lags.
- --ar-align-with-px to set p_ret = p_x per run, aligning AR lag count with the independent variable.

The modelling functions DO NOT save inside this CLI:
    granger_causality_{control_var}_{version}.csv  (when using controls)
    granger_causality_{version}.csv               (without controls)

Examples
--------
Run multiple versions with aligned control lags and aligned AR lags for p_x in {1,2,3}:
no control :
python run_estimate_granger_causality.py run-all-versions --versions w0  --versions w1 --versions w2 --versions binary --control-var none --p-x-range 1,3 --ar-align-with-px --p-ret 3 --n-boot 10000 --min-obs 0 --outdir data/processed/variables

control VIX:
python run_estimate_granger_causality.py run-all-versions --versions w0  --versions w1 --versions w2 --versions binary --control-var log_growth_VIX --controls-file data/processed/variables/log_growth_VIX.csv --controls-lags log_growth_VIX --p-x-range 1,3 --controls-align-with-px --ar-align-with-px --p-ret 3 --n-boot 10000 --min-obs 0 --outdir data/processed/variables

S&P 500:
python run_estimate_granger_causality.py run-all-versions --versions w0  --versions w1 --versions w2 --versions binary --control-var log_growth_sp500 --controls-file data/processed/variables/log_growth_sp500.csv --controls-lags log_growth_sp500 --p-x-range 1,3 --controls-align-with-px --ar-align-with-px --p-ret 3 --n-boot 10000 --min-obs 0 --outdir data/processed/variables
log_growth_sox

Sox:
python run_estimate_granger_causality.py run-all-versions --versions w0  --versions w1 --versions w2 --versions binary --control-var log_growth_sox --controls-file data/processed/variables/log_growth_sox.csv --controls-lags log_growth_sox --p-x-range 1,3 --controls-align-with-px --ar-align-with-px --p-ret 3 --n-boot 10000 --min-obs 0 --outdir data/processed/variables

"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import pandas as pd
import typer

this_file = Path(__file__).resolve()
scripts_dir = this_file.parent
src_dir = scripts_dir.parent
root_dir = src_dir.parent  # to load csv relative to repo root
sys.path.append(str(src_dir))

from modelling.estimate_granger_causality import (  # noqa: E402
    run_gc_mbboot_fdr,
    run_gc_mbboot_fdr_controls,
)

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
      "n_articles"       -> {"n_articles": 1}
      "n_articles:3,vix" -> {"n_articles": 3, "vix": 1}
    If --controls-align-with-px is set, the numeric parts are ignored downstream.
    """
    if spec is None:
        return None
    s = spec.strip()
    if not s:
        return None

    out: Dict[str, int] = {}
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

# helper to find output path
def _resolve_outdir(outdir: Optional[Path]) -> Path:
    if outdir is None:
        return (root_dir / "data" / "processed" / "variables").resolve()
    outdir = Path(outdir)
    if not outdir.is_absolute():
        outdir = (root_dir / outdir).resolve()
    return outdir

def canonicalize_betas(df: pd.DataFrame, duplicate_beta0_ar: bool = False) -> pd.DataFrame:
    """
    Rename beta columns to a unified scheme for BOTH controls/no-controls paths:

      Intercept:                    β₀
      Cross (independent) lags:     β_x1, β_x2, ...
      Autoregressive (dependent) lags: β_x1_ar, β_x2_ar, ...
      Controls:                     β_ctrl_{name}{k}

    Works for columns named with or without '_lag' in the source:
      A2R_beta_x_lag2  or  A2R_beta_x_2
      R2A_beta_ret_lag1 or R2A_beta_ret_1
      A2R_beta_ctrl_n_articles_lag3 or A2R_beta_ctrl_n_articles_3
    """
    if df.empty:
        return df
    df = df.copy()
    rename: Dict[str, str] = {}

    # Intercepts (both directions)
    for c in df.columns:
        if c in ("A2R_beta_const", "R2A_beta_const"):
            rename[c] = "β₀"

    # Role maps (we don't need Direction per-row; names carry A2R/R2A)
    role_maps = [
        dict(cross="x", ar="ret", prefix="A2R"),   # AINI → Return
        dict(cross="ret", ar="x", prefix="R2A"),   # Return → AINI
    ]

    cols = list(df.columns)
    for rm in role_maps:
        cross, ar, pref = rm["cross"], rm["ar"], rm["prefix"]
        pat_cross = re.compile(rf"^{pref}_beta_{cross}(?:_lag)?(\d+)$")
        pat_ar = re.compile(rf"^{pref}_beta_{ar}(?:_lag)?(\d+)$")
        pat_ctrl = re.compile(rf"^{pref}_beta_ctrl_(.+?)(?:_lag)?(\d+)$")

        for c in cols:
            if not isinstance(c, str) or c in rename:
                continue
            m = pat_cross.match(c)
            if m:
                rename[c] = f"β_x{m.group(1)}"
                continue
            m = pat_ar.match(c)
            if m:
                rename[c] = f"β_x{m.group(1)}_ar"
                continue
            m = pat_ctrl.match(c)
            if m:
                ctrl_name_raw, lag = m.group(1), m.group(2)
                safe = re.sub(r"[^A-Za-z0-9]+", "_", ctrl_name_raw).strip("_")
                rename[c] = f"β_ctrl_{safe}{lag}"
                continue

    df = df.rename(columns=rename)

    if duplicate_beta0_ar and "β₀" in df.columns:
        df["β₀_ar"] = df["β₀"]

    return df


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
    p_ret: int = 3,
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
    ar_align_with_px: bool = typer.Option(
        False,
        "--ar-align-with-px",
        help="If True, set p_ret = p_x for each run so AR lag count matches the independent variable.",
    ),
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

    frames: List[pd.DataFrame] = []
    for px in lag_values:
        typer.echo(f"[RUN ] Estimating with p_x={px}")

        # Align AR lags with p_x if requested
        pret_eff = (px if ar_align_with_px else p_ret)

        ctrl_lags_eff = (
            {name: px for name in ctrl_lags_map.keys()}
            if (use_controls and controls_align_with_px)
            else ctrl_lags_map
        )

        if use_controls:
            df = run_gc_mbboot_fdr_controls(
                aini_df=aini_df,
                fin_data=fin_data,
                version=version,
                control_var=control_var,
                controls_df=ctrl_df,
                controls_lags=ctrl_lags_eff,
                aini_variants=variants_list,
                p_ret=pret_eff,
                p_x=px,
                min_obs=min_obs,
                n_boot=n_boot,
                seed=seed,
                fdr_alpha=fdr_alpha,
                cov_for_analytic=cov_for_analytic,
                hac_lags=hac_lags,
                weight_dist=weight_dist,
                save_csv=False,
                outdir=outdir,
                outname=None,
                n_jobs=n_jobs,
            )
        else:
            df = run_gc_mbboot_fdr(
                aini_df=aini_df,
                fin_data=fin_data,
                version=version,
                aini_variants=variants_list,
                p_ret=pret_eff,
                p_x=px,
                min_obs=min_obs,
                n_boot=n_boot,
                seed=seed,
                fdr_alpha=fdr_alpha,
                cov_for_analytic=cov_for_analytic,
                hac_lags=hac_lags,
                weight_dist=weight_dist,
                save_csv=False,
                outdir=outdir,
                outname=None,
                n_jobs=n_jobs,
            )

        df["p_x"] = px
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    typer.echo(f"[OK] Combined results shape: {combined.shape}")

    # Canonicalize beta column names BEFORE saving
    combined = canonicalize_betas(combined, duplicate_beta0_ar=False)

    # Save exactly one CSV
    outdir = _resolve_outdir(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"[OUTDIR] {outdir}")

    outname = (
        f"granger_causality_{control_var}_{version}.csv"
        if use_controls
        else f"granger_causality_{version}.csv"
    )
    outpath = outdir / outname
    combined.to_csv(outpath, index=False, encoding="utf-8-sig")
    typer.echo(f"[SAVE] {outpath}")

@app.command("run-all-versions")
def run_all_versions(
    versions: List[str] = typer.Option(["binary", "w0", "w1", "w2", "w3"]),
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
    controls_align_with_px: bool = typer.Option(
        False, "--controls-align-with-px", help="Align controls to same lag as p_x."
    ),
    ar_align_with_px: bool = typer.Option(
        False, "--ar-align-with-px", help="Set p_ret = p_x to align AR lag count with the independent variable."
    ),
    p_ret: int = typer.Option(3, help="Baseline AR lag count when --ar-align-with-px is False."),
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
            ar_align_with_px=ar_align_with_px,
            p_ret=p_ret,
        )


if __name__ == "__main__":
    app()
