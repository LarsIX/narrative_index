import pandas as pd
import textwrap
from ai_windows import extract_human_readable_snippet
from pathlib import Path
import torch
from transformers import AutoTokenizer
from ai_windows import extract_multiple_ai_snippets_title_context
from typing import Literal, Optional
import re


def _format_text(text: object, width: int) -> str:
    """Wrap text, preserving paragraph breaks."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)
    paras = [p.strip() for p in s.split("\n") if p.strip()]
    return "\n\n".join(textwrap.fill(p, width=width) for p in paras)

def read(
    df: pd.DataFrame,
    wrap_width: int = 100,
    require_nonzero_hype: bool = True,
) -> None:
    """
    Interactive article reader (Jupyter/console).

    Expects df with (at least):
      - 'date', 'article_id'
      - hype scores: 'hype_score_w0', 'hype_score_w1', 'hype_score_w2', 'hype_score_c'
      - AI windows: any columns starting with 'ai_window' (e.g., 'ai_window',
                    'ai_window_w1', 'ai_window_w2', 'ai_window_c', 'ai_window_t')

    Shows:
      - date, article_id
      - hype_score_w0, hype_score_w1, hype_score_w2, hype_score_c
      - all ai_window* columns (printed in sorted column-name order)

    Parameters
    ----------
    df : pd.DataFrame
    wrap_width : int
        Line width for wrapping AI window text.
    require_nonzero_hype : bool
        If True, only show rows where ANY hype score != 0.
    """
    # --- Validate minimal structure, but don't mutate caller ---
    required_base = ["date", "article_id"]
    required_hype = ["hype_score_w0", "hype_score_w1", "hype_score_w2", "hype_score_c"]

    missing = [c for c in required_base + required_hype if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    ai_window_cols = sorted([c for c in df.columns if c.startswith("ai_window")])
    if not ai_window_cols:
        raise ValueError("No AI window columns found (expected at least one 'ai_window*').")

    data = df.copy()

    # Ensure types for display
    for c in required_hype:
        data[c] = pd.to_numeric(data[c], errors="coerce").fillna(0.0)

    # Optional filter on nonzero hype
    if require_nonzero_hype:
        mask = (data[required_hype] != 0).any(axis=1)
        data = data.loc[mask].reset_index(drop=True)

    # Basic sort for deterministic traversal (by date if present)
    if "date" in data.columns:
        data = data.sort_values("date").reset_index(drop=True)

    n_articles = len(data)
    if n_articles == 0:
        print("No articles to display (empty after filtering).")
        return

    i = 0
    try:
        while True:
            if i < 0:
                i = 0
            elif i >= n_articles:
                i = n_articles - 1

            row = data.iloc[i]

            print("=" * 80)
            print(f"Date       : {row.get('date', '')}")
            print(f"Article ID : {row.get('article_id', '')}")
            print(
                "Hype       : "
                f"w0={row.get('hype_score_w0', 0):.3f}, "
                f"w1={row.get('hype_score_w1', 0):.3f}, "
                f"w2={row.get('hype_score_w2', 0):.3f}, "
                f"c={row.get('hype_score_c', 0):.3f}"
            )
            print("-" * 80)

            # Print all available AI windows
            for col in ai_window_cols:
                txt = row.get(col, "")
                if pd.isna(txt) or str(txt).strip() == "":
                    continue
                print(f"[{col}]")
                print(_format_text(txt, wrap_width))
                print("-" * 80)

            print("=" * 80)
            print(f"Article {i+1} of {n_articles}")

            cmd = input("\nEnter command (n=next, p=previous, s=stop): ").strip().lower()
            if cmd == "n":
                if i + 1 < n_articles:
                    i += 1
                else:
                    print("Already at last article.")
            elif cmd == "p":
                if i > 0:
                    i -= 1
                else:
                    print("Already at first article.")
            elif cmd == "s":
                print("Stopping reader.")
                break
            else:
                print("Invalid command. Use n, p, or s.")
    except KeyboardInterrupt:
        print("\nStopping reader (KeyboardInterrupt).")



def _ensure_ai_windows(df: pd.DataFrame, *, context_window: int, max_tokens: int = 512) -> pd.DataFrame:
    """
    Ensure df has 'ai_window' built with the requested context_window.
    If not present, (re)compute using the provided extractor and tag 'context_window_used'.
    """
    # If you already store which window was used, you could check and only rebuild if mismatched.
    needs_build = "ai_window" not in df.columns
    if needs_build:
        df = extract_multiple_ai_snippets_title_context(
            df,
            text_col="cleaned_corpus",
            output_col="ai_window",
            tokenizer_name="bert-base-uncased",
            max_tokens=max_tokens,
            context_window=context_window,
        )
    df["context_window_used"] = context_window
    return df

AI_KEYWORDS = [
    r'\bAI\b', r'\bA\.I\.\b', r'\bAGI\b', r'\bartificial intelligence\b',
    r'\bartificial general intelligence\b', r'\bhuman-level AI\b',
    r'\blarge language models?\b', r'\bLLM\b', r'\bGPT-?\d*\b',
    r'\bmachine learning\b', r'\bdeep learning\b', r'\bneural networks?\b',
    r'\btransformers?\b', r'\bGANs?\b', r'\bgenerative AI\b',
    r'\bprompt engineering\b', r'\bhallucination(s)?\b',
    r'\bautonomous systems?\b', r'\bfoundation models?\b',
    r'\btraining datasets?\b', r'\bcomputer vision\b',
    r'\binterpretability\b', r'\bresponsible AI\b', r'\bopen[- ]source\b'
]
_AI_RE = re.compile("|".join(AI_KEYWORDS), flags=re.IGNORECASE)

def _sent_split(text: str) -> list[str]:
    """Lightweight sentence splitter; avoids heavy deps. Good enough for auditing."""
    if not isinstance(text, str) or not text.strip():
        return []
    # Split on ., !, ? followed by whitespace/newline; keep punctuation attached.
    parts = re.split(r'(?<=[\.\!\?])\s+', text.strip())
    # Remove obvious empties/whitespace
    return [p.strip() for p in parts if p and p.strip()]

def _build_ai_window_from_fields(
    title: str, sub_title: str, corpus: str,
    *, context_window: int = 2,
    max_tokens: int = 512,
    length_tokenizer: Optional[AutoTokenizer] = None
) -> str:
    """
    Combine title + sub_title + corpus, pull sentences around AI keyword hits,
    and truncate so the *tokenized length* (via length_tokenizer) stays ≤ max_tokens.
    """
    title = title or ""
    sub_title = sub_title or ""
    corpus = corpus or ""
    full_text = " ".join([title.strip(), sub_title.strip(), corpus.strip()]).strip()
    if not full_text:
        return ""

    sents = _sent_split(full_text)
    if not sents:
        return ""

    hit_idxs = set()
    for i, s in enumerate(sents):
        if _AI_RE.search(s):
            # expand window around the hit
            for off in range(-context_window, context_window + 1):
                j = i + off
                if 0 <= j < len(sents):
                    hit_idxs.add(j)

    if not hit_idxs:
        return ""  # no AI context → empty by design

    # Accumulate in order, respecting token budget
    if length_tokenizer is None:
        length_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

    snippet = ""
    for idx in sorted(hit_idxs):
        candidate = (snippet + " " + sents[idx]).strip() if snippet else sents[idx]
        # Count tokens the same way your snippet builder did (bert-base-uncased)
        n_tokens = len(length_tokenizer.tokenize(candidate))
        if n_tokens > max_tokens:
            break
        snippet = candidate

    return snippet.strip()

def _finbert_tokens(text: str, *, max_length: int = 512, tokenizer=None) -> list[str]:
    """Get the *exact* tokens Prosus FinBERT would see (with specials, truncation)."""
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert", use_fast=True)
    enc = tokenizer(
        text if isinstance(text, str) else "",
        padding=False, truncation=True, max_length=max_length,
        add_special_tokens=True, return_tensors=None
    )
    ids = enc["input_ids"] if isinstance(enc["input_ids"], list) else [enc["input_ids"]]
    return tokenizer.convert_ids_to_tokens(ids[0])

def investigate_hype_articles(
    df: pd.DataFrame,
    *,
    version: Literal["w0","w1","w2","c"] = "w1",
    context_window: int = 2,
    max_tokens: int = 512,
    show_n: int = 5,
    add_joined_tokens_col: bool = True
) -> pd.DataFrame:
    """
    Filter to rows where hype_score_<version> != 0, rebuild the AI-window from
    (title + sub_title + cleaned_corpus) with context_window=2, and return tokens
    that Prosus FinBERT would see for that snippet.

    Returns a DataFrame with: article_id, title, sub_title, section, date_<ver>,
    sentiment_label_<ver>, hype_score_<ver>, ai_window, tokens [, tokens_joined].
    """
    # Required text columns
    for c in ["title", "sub_title", "cleaned_corpus"]:
        if c not in df.columns:
            raise ValueError(f"Missing required text column: '{c}'")

    # Map version -> column names
    colmap = {
        "w0": {"hype": "hype_score_w0", "label": "sentiment_label_w0", "date": "date"},
        "w1": {"hype": "hype_score_w1", "label": "sentiment_label_w1", "date": "date_w1"},
        "w2": {"hype": "hype_score_w2", "label": "sentiment_label_w2", "date": "date_w2"},
        "c" : {"hype": "hype_score_c",  "label": "sentiment_label_c",  "date": "date_c"},
    }
    if version not in colmap:
        raise ValueError("version must be one of {'w0','w1','w2','c'}")

    hype_col  = colmap[version]["hype"]
    label_col = colmap[version]["label"]
    date_col  = colmap[version]["date"]
    for c in [hype_col, label_col, date_col]:
        if c not in df.columns:
            raise ValueError(f"DataFrame lacks required column for version '{version}': '{c}'")

    work = df.copy()
    work[hype_col] = pd.to_numeric(work[hype_col], errors="coerce").fillna(0)

    # Keep only hype ≠ 0
    work = work.loc[work[hype_col] != 0].copy()
    if work.empty:
        print(f"No hype rows found for {version} (column '{hype_col}').")
        return work

    # Prepare tokenizers once
    len_tok = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    fin_tok = AutoTokenizer.from_pretrained("ProsusAI/finbert", use_fast=True)

    # Build snippets
    ai_windows = []
    for _, r in work.iterrows():
        snippet = _build_ai_window_from_fields(
            title=str(r.get("title", "") or ""),
            sub_title=str(r.get("sub_title", "") or ""),
            corpus=str(r.get("cleaned_corpus", "") or ""),
            context_window=context_window,
            max_tokens=max_tokens,
            length_tokenizer=len_tok
        )
        # Fallback: if somehow empty, take a short lead from combined text (rare)
        if not snippet:
            base = " ".join([str(r.get("title","")), str(r.get("sub_title","")), str(r.get("cleaned_corpus",""))]).strip()
            # Trim by token count with bert-base-uncased to stay consistent
            toks = len_tok.tokenize(base)[:max_tokens]
            snippet = len_tok.convert_tokens_to_string(toks)
        ai_windows.append(snippet)
    work["ai_window"] = ai_windows

    # FinBERT tokens
    work["tokens"] = work["ai_window"].apply(lambda t: _finbert_tokens(t, max_length=512, tokenizer=fin_tok))
    if add_joined_tokens_col:
        work["tokens_joined"] = work["tokens"].apply(lambda xs: " ".join(xs))

    # Return a compact view for auditing
    cols = [
        "article_id", "title", "sub_title", "section",
        date_col, label_col, hype_col, "ai_window", "tokens"
    ]
    if add_joined_tokens_col:
        cols.append("tokens_joined")

    out = work.loc[:, [c for c in cols if c in work.columns]]

    # Quick peek
    for _, row in out.head(show_n).iterrows():
        print("="*100)
        print(f"Title : {row.get('title','')}")
        print(f"Date  : {row.get(date_col,'')}")
        print(f"Sec   : {row.get('section','')}")
        print(f"Lbl   : {row.get(label_col,'')}, hype={row.get(hype_col)}")
        print("-"*100)
        print(row.get("ai_window","")[:1200])
        print("-"*100)
        print("TOKENS:")
        print(row.get("tokens_joined","") if add_joined_tokens_col else " ".join(row.get("tokens", [])))
        print("="*100)

    return out
