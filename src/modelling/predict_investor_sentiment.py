"""
Run FinBERT sentiment prediction on WSJ articles.

This script loads WSJ articles from a reduced SQLite database for a specific year,
applies the pretrained FinBERT model to classify sentiment (Positive, Negative, Neutral),
and saves the results as a CSV.

Example
-------
predict_sentiment(2024, batch_size=8)

Notes
-----
- Uses GPU if available.
- Input articles must be stored in a *reduced* SQLite database.
- Combines title and article corpus for prediction.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np
import typer
import sqlite3
from pathlib import Path
from torch.nn.functional import softmax
from tqdm import tqdm


def predict_sentiment(year: int, batch_size: int = 4):
    """
    Predict sentiment for WSJ articles using FinBERT.

    Parameters
    ----------
    year : int
        The year for which WSJ articles are processed.
    batch_size : int, optional
        Number of articles processed in parallel during inference. Default is 4.

    Returns
    -------
    dict
        A dictionary with the following keys:
        - "output_path" : Path to the output CSV file.
        - "device" : Device used for inference (CPU or CUDA).
    """
    typer.echo("🚀 Sentiment prediction started")

    # Select GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define paths
    project_root = Path(__file__).resolve().parents[1]
    db_path = project_root / "data" / "processed" / "articles" / f"articlesWSJ_clean_reduced_{year}.db"
    output_path = project_root / "data" / "processed" / "variables" / f"Sent_prediction_{year}.csv"

    # Load articles from database
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM article", conn)
    conn.close()

    # Concatenate title and corpus for classification input
    df["texts"] = df["title"].fillna("") + ". " + df["corpus"].fillna("")
    texts = df["texts"].tolist()

    # Load pretrained FinBERT
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.to(device)
    model.eval()

    # Initialize result storage
    all_labels = []
    all_scores = []

    # Inference loop
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        batch_texts = texts[i:i+batch_size]

        # Tokenize and move to device
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            probs = softmax(outputs.logits, dim=-1)
            scores, preds = torch.max(probs, dim=-1)

        # Collect results
        all_labels.extend(preds.cpu().numpy())
        all_scores.extend(scores.cpu().numpy())

    # Map label IDs to human-readable labels
    label_map = model.config.id2label
    df["sentiment_label"] = [label_map[label] for label in all_labels]
    df["sentiment_score"] = all_scores

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    typer.echo(f"Saved sentiment predictions to {output_path}")

    return {
        "output_path": output_path,
        "device": str(device)
    }
