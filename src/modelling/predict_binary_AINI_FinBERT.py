"""
Label WSJ articles with presence of AI-related narrative using fintetuned FinBERT model.
"""

from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, BertModel
import torch
import pandas as pd
import sqlite3
import sys
from tqdm import tqdm
import typer
from CustomFinBERT import CustomFinBERT

# find the project root 
project_root = Path(__file__).resolve().parents[2]

# move to modelling folder
modelling = project_root / "src" / "modelling"

# Add to Python path
if str(modelling) not in sys.path:
    sys.path.append(str(modelling))

from ai_windows import extract_multiple_ai_snippets_with_context


def label_data(year, batch_size=16):
    """
    Label WSJ articles for the presence of AI-related narratives using a fine-tuned binary FinBERT model.

    This function performs the following steps:
    1. Loads cleaned WSJ articles from SQLite.
    2. Excludes already annotated articles (based on a reference CSV).
    3. Extracts AI-relevant snippets with sentence-level context.
    4. Filters invalid or untokenizable texts.
    5. Applies a fine-tuned binary FinBERT classifier (Narrative vs No Narrative).
    6. Saves the prediction results to CSV.

    Parameters
    ----------
    year : int
        The year of the articles to be labeled.
    batch_size : int, optional
        Number of samples per batch during inference (default: 16).

    Returns
    -------
    dict
        Dictionary containing:
        - "output_path": str, path to saved prediction CSV
        - "n_labeled_articles": int, number of articles labeled

    Notes
    -----
    - Uses a fine-tuned FinBERT binary classifier with class weights.
    - Assumes existence of:
        - Trained model in `models/FinBERT_Binary`
        - Cleaned article DB at `data/processed/articles/articlesWSJ_clean_{year}.db`
        - Reference CSV `annotated_subsample_WSJ_final.csv`
    - Requires GPU for optimal performance, but will fall back to CPU.
    """

  
    model_path = project_root / "models" / "FinBERT_Binary"
    model_path_str = model_path.as_posix()
  
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_path_str,
        local_files_only=True,
        use_auth_token=False
    )
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=model_path_str,
        local_files_only=True,
        use_auth_token=False
    )

    # control progress
    typer.echo("Tokenizer finished")

    # loading class weights
    class_weights_path = model_path / "class_weights.pt"
    if not class_weights_path.exists():
        raise FileNotFoundError("Missing class_weights.pt in model directory.")
    class_weights_tensor = torch.load(class_weights_path, map_location=device)

    # control progress
    typer.echo("class weights loaded")

    # loading backbone, building model
    backbone = BertModel.from_pretrained("ProsusAI/finbert", config=config)
    model = CustomFinBERT(backbone=backbone, class_weights=class_weights_tensor, config=config)

    # loading trained weights
    state_dict_path = model_path / "pytorch_model.bin"
    if not state_dict_path.exists():
        raise FileNotFoundError("Missing model weights at pytorch_model.bin.")
    model.load_state_dict(torch.load(state_dict_path, map_location=device))

    # move to device and evaluate
    model.to(device)

    # moved model to cuda
    typer.echo(f"model loaded on {str(device)}")

    model.eval()

    # Helper: Batch prediction
    def predict_label(texts, model, tokenizer, max_length=512, batch_size=batch_size, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        predictions = []

        # run predictions and track progress
        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
            batch = texts[i:i + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = model(**encoded)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                predictions.extend(preds.cpu().numpy())

        return predictions

    # Load WSJ data
    db_path = project_root / "data" / "processed" / "articles" / f"articlesWSJ_clean_{year}.db"
    conn = sqlite3.connect(db_path)
    df = pd.read_sql('SELECT * FROM article', conn)
    conn.close()

    annotated_csv = project_root / "data" / "processed" / "articles" / "annotated_subsample_WSJ_final.csv"
    df_annotated = pd.read_csv(annotated_csv)
    df_unlabeled = df[~df["article_id"].isin(df_annotated["article_id"])].copy()
    df_unlabeled["cleaned_corpus"] = "Title: " + df_unlabeled["title"] + "\n\n" + df_unlabeled["cleaned_corpus"]
    # Extract snippets
    df_unlabeled = extract_multiple_ai_snippets_with_context(
        df_unlabeled,
        text_col="cleaned_corpus",
        output_col="ai_window",
        context_window=2
    )

    # Tokenization robustness
    def filter_invalid_texts(texts, tokenizer, max_length=512):
        valid_texts = []
        for text in texts:
            try:
                tokens = tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=max_length,
                    return_tensors='pt'
                )
                if tokens['input_ids'].max().item() < tokenizer.vocab_size:
                    valid_texts.append(text)
            except Exception as e:
                continue
        return valid_texts

    texts_raw = df_unlabeled["ai_window"].astype(str).tolist()
    texts = filter_invalid_texts(texts_raw, tokenizer)
    df_unlabeled = df_unlabeled[df_unlabeled["ai_window"].isin(texts)].reset_index(drop=True)

    # Predict
    df_unlabeled["predicted_label"] = predict_label(texts, model, tokenizer, device=device)
    df_unlabeled["predicted_class"] = df_unlabeled["predicted_label"].map({0: "No narrative", 1: "Narrative"})

    # Save
    output_path = project_root / "data" / "processed" / "variables" / f"FinBERT_binary_prediction_{year}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_unlabeled.to_csv(output_path, index=False)

    return {
        "output_path": str(output_path),
        "n_labeled_articles": len(df_unlabeled)
    }
