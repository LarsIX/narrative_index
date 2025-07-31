"""
CLI app to generate naive AI relevance labels from article files.

Usage:
    python run_naive_labeling.py 2023 --file articlesWSJ_clean_2023.csv
"""

import typer
import pandas as pd
from pathlib import Path
import sys

# Setup paths
project_root = Path(__file__).resolve().parents[2]
modelling_path = project_root / "src" / "annotation"
sys.path.append(str(modelling_path))

from label_articles import naive_labeling  

app = typer.Typer()


@app.command()
def run(
    year: str = typer.Argument(..., help="Year of the articles (used in output filename)"),
    file: str = typer.Option(..., help="CSV filename in data/processed/articles/ to label"),
    subset: bool = typer.Option(False,help="Indicate if applied to annotated subset"),
    title_col: str = typer.Option("title", help="Column name for article titles"),
    text_col: str = typer.Option("corpus", help="Column name for article text"),
    output_col: str = typer.Option("about_ai", help="Name of the output label column")
):
    """
    Labels articles using naive AI keyword matching.

    This function loads a CSV from `data/processed/articles/`,
    checks for AI keywords in title and corpus,
    flags if a annotated subset is used,
    and saves the labeled file as `naive_AI_labels_{year}.csv`.
    """

    data_dir = project_root / "data" / "processed" / "articles"
    input_path = data_dir / file
    filename = f"naive_AI_labels_annotated_sub_{year}.csv" if subset else f"naive_AI_labels_{year}.csv"
    output_path = data_dir / filename
    df = pd.read_csv(input_path)
    df = naive_labeling(df, title_col=title_col, text_col=text_col, output_col=output_col)

    df.to_csv(output_path, index=False)
    print(f"Saved labeled file to: {output_path}")


if __name__ == "__main__":
    app()
