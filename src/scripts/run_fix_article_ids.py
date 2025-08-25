# src/scripts/run_fix_article_ids.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple
import typer

# --- Make 'src' importable 
SRC_DIR = Path(__file__).resolve().parents[1]  # .../src
sys.path.append(str(SRC_DIR))

from preprocessing.fix_article_ids import load_aini_and_fix_ids  # noqa: E402

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _default_var_path() -> Path:
    """Default to <repo_root>/data/processed/variables."""
    repo_root = SRC_DIR.parent  # parent of 'src'
    return (repo_root / "data" / "processed" / "variables").resolve()


def _parse_years(years_csv: str) -> Tuple[int, ...]:
    try:
        years = tuple(int(y.strip()) for y in years_csv.split(",") if y.strip())
        if not years:
            raise ValueError
        return years
    except Exception as e:
        raise typer.BadParameter(
            "Use a comma-separated list of years, e.g. '2023,2024,2025'."
        ) from e


@app.command("run")
def run(
    var_path: Path = typer.Option(
        None,
        "--var-path",
        "-p",
        help="Directory containing the AINI CSVs (defaults to <repo_root>/data/processed/variables).",
        exists=False,
        file_okay=False,
        dir_okay=True,
        readable=True,
        writable=True,
        resolve_path=True,
    ),
    years: str = typer.Option(
        "2023,2024,2025",
        "--years",
        "-y",
        help="Comma-separated list of years to process, e.g. '2023,2024,2025'.",
    ),
    strict_dates: bool = typer.Option(
        True, "--strict-dates/--no-strict-dates", help="Fail on invalid dates."
    ),
    write_back: bool = typer.Option(
        True, "--write-back/--no-write-back", help="Overwrite the input CSVs in place."
    ),
    preview: bool = typer.Option(
        False, "--preview", help="Print a tiny before/after sample (does not skip writing)."
    ),
):
    """
    Prefix YYYY to article_id for AINI CSVs (custom, w0, w1, w2) based on each row's `date`.
    """
    if var_path is None:
        var_path = _default_var_path()
    years_tuple = _parse_years(years)

    typer.echo(f"[info] var_path    : {var_path}")
    typer.echo(f"[info] years       : {years_tuple}")
    typer.echo(f"[info] strict_dates: {strict_dates}")
    typer.echo(f"[info] write_back  : {write_back}")

    if preview:
        # show a tiny sample before touching files
        import pandas as pd
        sample_file = var_path / f"FinBERT_AINI_prediction_{years_tuple[0]}_on_binary.csv"
        if sample_file.exists():
            df0 = pd.read_csv(sample_file, nrows=3)[["date", "article_id"]]
            typer.echo("[preview] Before:\n" + df0.to_string(index=False))
        else:
            typer.echo(f"[warn] Sample file not found for preview: {sample_file}")

    dfs = load_aini_and_fix_ids(
        var_path=var_path,
        years=years_tuple,
        date_col="date",
        id_col="article_id",
        strict_dates=strict_dates,
        write_back=write_back,
    )

    # brief summary
    for name in ("c_df", "w0_df", "w1_df", "w2_df"):
        df = dfs[name]
        typer.echo(
            f"[done] {name}: rows={len(df)}, unique article_id={df['article_id'].nunique(dropna=True)}"
        )

    if preview:
        df1 = dfs["c_df"].head(3)[["date", "article_id"]]
        typer.echo("[preview] After:\n" + df1.to_string(index=False))


if __name__ == "__main__":
    app()
