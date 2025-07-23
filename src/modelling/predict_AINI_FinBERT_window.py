"""
Predict AINI from WSJ articles using standard FinBERT model & custom window function.
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
from ai_windows import extract_multiple_ai_snippets_title_context

def predict_sentiment(year, batch_size,context_window=1,max_tokens=512):
    """
    Run FinBERT sentiment inference on WSJ articles for a given year.

    - Loads cleaned articles from SQLite database.
    - Combines title and corpus for prediction.
    - Applies pretrained FinBERT model to classify sentiment (Positive, Negative, Neutral).
    - Saves results to CSV in `data/processed/variables/`.

    Uses GPU if available.
    """


    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    db_path = project_root / "data" / "processed" / "articles" / f"articlesWSJ_clean_{year}.db"
    output_path = project_root / "data" / "processed" / "variables" / f"FinBERT_AINI_prediction_{year}_windsize_{context_window}.csv"

    # Load articles
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query('SELECT * FROM article', conn)
    conn.close()

    # Extract AI-related snippets 
    df = extract_multiple_ai_snippets_title_context(
        df,
        text_col='corpus',
        output_col='ai_window',
        tokenizer_name='bert-base-uncased',
        max_tokens=max_tokens,
        context_window=context_window
    )

    # Use the AI snippet column for prediction
    df = df[df["ai_window"].str.strip().astype(bool)]
    texts = df["ai_window"].tolist()

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.to(device)
    model.eval()

    # Storage
    all_labels = []
    all_scores = []

    # Inference loop
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = softmax(outputs.logits, dim=-1)
            scores, preds = torch.max(probs, dim=1)

        all_labels.extend(preds.cpu().numpy())
        all_scores.extend(scores.cpu().numpy())

    # Map labels
    
    label_map = model.config.id2label
    df["sentiment_label"] = [label_map[label] for label in all_labels]
    df["sentiment_score"] = all_scores

    # Map sentiment labels to numeric hype scores
    hype_map = {
        "positive": 1,
        "neutral": 0,
        "negative": -1
    }
    df["hype_score"] = df["sentiment_label"].map(hype_map)

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return {
        "output_path": output_path,
        "device": str(device),
    }