from pathlib import Path
from typing import Dict
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from tqdm import tqdm

def predict_sentiment(
    year: int,
    batch_size: int = 16,
    model_name: str = "ProsusAI/finbert",
) -> Dict[str, str]:
    """
    Run FinBERT sentiment prediction for AI-relevant WSJ articles.

    Parameters
    ----------
    year : int
        Year for which to run sentiment predictions.
        The input CSV must be located at:
        ``data/processed/variables/FinBERT_binary_prediction_{year}.csv``.
    batch_size : int, default=16
        Number of articles to process per batch during inference.
    model_name : str, default="ProsusAI/finbert"
        HuggingFace model identifier for the FinBERT model to use.

    Returns
    -------
    Dict[str, str]
        Dictionary with:
        - ``"output_path"`` : str
            Absolute path to the saved CSV file with predictions.
        - ``"device"`` : str
            Device used for inference ("cpu" or "cuda").

    Raises
    ------
    ValueError
        If the input CSV does not contain the column ``'predicted_label'``.

    Notes
    -----
    - The function filters to AI-relevant articles based on ``predicted_label == 1``.
    - Title and subtitle are concatenated into a short ``_head`` string.
    - The corpus text is passed as a ``text_pair`` to the tokenizer, with
      ``truncation="only_second"`` so the title/subtitle remains intact and only the
      corpus is truncated to fit the 512-token limit.
    - Predictions are mapped to sentiment labels ("positive", "neutral", "negative")
      and a numeric ``hype_score``:
        - positive → 1
        - neutral  → 0
        - negative → -1
    - The output CSV is saved to:
      ``data/processed/variables/FinBERT_AINI_prediction_{year}_on_binary.csv``.

    Examples
    --------
    Run sentiment prediction for 2024 with default settings:

    >>> predict_sentiment(year=2024)

    Run with a custom batch size and model:

    >>> predict_sentiment(year=2023, batch_size=8, model_name="ProsusAI/finbert")
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "data" / "processed" / "variables" / f"FinBERT_binary_prediction_{year}.csv"
    output_path = project_root / "data" / "processed" / "variables" / f"FinBERT_AINI_prediction_{year}_on_binary.csv"

    # Load data
    df = pd.read_csv(input_path)
    if "predicted_label" not in df.columns:
        raise ValueError("Column 'predicted_label' not found in input CSV.")

    # Keep only AI-relevant rows for inference to save compute
    mask_ai = df["predicted_label"].astype(int) == 1
    df["ai_relevant"] = mask_ai

    # Prepare text fields
    for col in ["title", "sub_title", "corpus"]:
        if col not in df.columns:
            df[col] = ""  # avoid KeyErrors; set missing columns to empty

    # Combine title + subtitle (keep small, high-signal)
    df["_head"] = (df["title"].fillna("") + " " + df["sub_title"].fillna("")).str.strip()
    df["_body"] = df["corpus"].fillna("")

    # Init model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    # Output holders sized to the full df (we'll fill only where ai_relevant)
    sentiment_ids = torch.full((len(df),), -1, dtype=torch.long)
    sentiment_scores = torch.zeros((len(df),), dtype=torch.float32)

    # Batched inference only on AI-relevant rows
    idxs = df.index[mask_ai].to_list()
    heads = df.loc[idxs, "_head"].tolist()
    bodies = df.loc[idxs, "_body"].tolist()

    with torch.no_grad():
        for s in tqdm(range(0, len(idxs), batch_size), desc="FinBERT sentiment"):
            b_slice = slice(s, s + batch_size)
            batch_heads = heads[b_slice]
            batch_bodies = bodies[b_slice]

            enc = tokenizer(
                batch_heads,
                batch_bodies,                 # text_pair
                padding=True,
                truncation="only_second",     # keep head intact, truncate body
                max_length=512,
                return_tensors="pt"
            ).to(device)

            logits = model(**enc).logits                          # [B, 3]
            probs = softmax(logits, dim=-1)                       # [B, 3]
            scores, preds = torch.max(probs, dim=1)               # [B], [B]

            # write back to the correct absolute indices
            abs_idx = idxs[b_slice]
            sentiment_ids[abs_idx] = preds.cpu()
            sentiment_scores[abs_idx] = scores.cpu()

    # Map ids → labels; build hype score
    id2label = model.config.id2label
    labels = [id2label[int(i)] if int(i) in id2label else "neutral" for i in sentiment_ids.tolist()]
    hype_map = {"positive": 1, "neutral": 0, "negative": -1}

    df["sentiment_label"] = labels
    df["sentiment_score"] = sentiment_scores.numpy()
    df["hype_score"] = df["sentiment_label"].map(hype_map).fillna(0).astype(int)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["_head", "_body"], errors="ignore").to_csv(output_path, index=False)

    print("AI-relevant articles:", int(mask_ai.sum()))
    print("Total articles saved:", len(df))
    print(df["hype_score"].value_counts(dropna=False))

    return {"output_path": str(output_path), "device": str(device)}
