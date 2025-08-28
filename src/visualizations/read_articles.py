import re
import html
import unicodedata
from difflib import SequenceMatcher
from typing import Literal, Optional, Iterable, Tuple

import pandas as pd
import textwrap
from pathlib import Path  # kept in case you use it elsewhere
import torch  # kept in case you use it elsewhere
from transformers import AutoTokenizer
from ai_windows import (
    extract_human_readable_snippet,              # kept for compatibility
    extract_multiple_ai_snippets_title_context,  # used below
)

# -----------------------------------------------------------------------------
# Formatting & basic helpers
# -----------------------------------------------------------------------------

def _format_text(text: object, width: int) -> str:
    """Wrap text, preserving paragraph breaks."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)
    paras = [p.strip() for p in s.split("\n") if p.strip()]
    return "\n\n".join(textwrap.fill(p, width=width) for p in paras)

# -----------------------------------------------------------------------------
# AI keyword config & sentence splitting
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Diff machinery (sentence/word/char) with robust normalization + fuzzy match
# -----------------------------------------------------------------------------

Granularity = Literal["sentence", "word", "char"]

_PUNCT_EDGE = r"[\,\.\;\:\!\?\u2026]"  # includes ellipsis …

def _ascii_punct(s: str) -> str:
    """Normalize common Unicode punctuation to ASCII-like forms."""
    # dashes
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    # smart quotes
    s = (s.replace("\u2018", "'").replace("\u2019", "'")
           .replace("\u201C", '"').replace("\u201D", '"'))
    # non-breaking space
    s = s.replace("\u00A0", " ")
    return s

def _normalize_unit(s: str) -> str:
    """Aggressive normalization for comparison (not for display)."""
    if not s:
        return ""
    # HTML entities, Unicode canonicalization, punctuation smoothing
    s = html.unescape(s)
    s = unicodedata.normalize("NFKC", s)
    s = _ascii_punct(s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s.strip())
    # remove trivial trailing punctuation (.,;:!?…)
    s = re.sub(fr"{_PUNCT_EDGE}+$", "", s)
    # lower for case-insensitive compare
    return s.casefold()

def _split_units(text: str, granularity: Granularity) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    if granularity == "sentence":
        return _sent_split(text)
    if granularity == "word":
        return re.findall(r"\S+", text)
    if granularity == "char":
        return list(text)
    raise ValueError(f"Unknown granularity: {granularity}")

def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def _is_match(nu: str, nv: str, *, similarity: float) -> bool:
    # exact
    if nu == nv:
        return True
    # containment (handles “short sentence” vs “longer sentence, extra clause”)
    if nu in nv or nv in nu:
        return True
    # fuzzy similarity
    if _similar(nu, nv) >= similarity:
        return True
    return False

def _diff_units(
    text_a: str,
    text_b: str,
    *,
    granularity: Granularity,
    similarity_threshold: float = 0.97
) -> list[str]:
    """
    Return the units (sentences/words/chars) that are in A but not in B,
    preserving A's original order, using normalization + containment + fuzzy matching.
    """
    a_units = _split_units(text_a, granularity)
    b_units = _split_units(text_b, granularity)
    if not a_units:
        return []

    b_norm = [_normalize_unit(u) for u in b_units]

    out = []
    emitted = set()  # track normalized strings already emitted

    for u in a_units:
        nu = _normalize_unit(u)
        if not nu or nu in emitted:
            continue
        matched = any(_is_match(nu, nb, similarity=similarity_threshold) for nb in b_norm)
        if not matched:
            out.append(u)     # keep original for display
            emitted.add(nu)
    return out

def _iter_ordered_pairs(cols: list[str], specific_pairs: Optional[Iterable[Tuple[str, str]]] = None):
    """
    Yield ordered (A,B) pairs. If specific_pairs is provided, only yield those that exist.
    Otherwise, generate all ordered pairs A != B.
    """
    if specific_pairs:
        for a, b in specific_pairs:
            if a in cols and b in cols and a != b:
                yield (a, b)
        return
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i != j:
                yield (a, b)

# -----------------------------------------------------------------------------
# Reader with AI window printing + window diffs
# -----------------------------------------------------------------------------

def read(
    df: pd.DataFrame,
    wrap_width: int = 100,
    require_nonzero_hype: bool = True,
    *,
    diff_granularity: Granularity = "sentence",
    pairs: Optional[Iterable[Tuple[str, str]]] = None,
    show_all_diffs: bool = True,
    similarity_threshold: float = 0.97,
) -> None:
    """
    Interactive article reader (Jupyter/console).

    Requires df with:
      - 'date', 'article_id'
      - hype scores: 'hype_score_w0', 'hype_score_w1', 'hype_score_w2', 'hype_score_c'
      - AI windows: any columns starting with 'ai_window' (e.g., 'ai_window',
                    'ai_window_w1', 'ai_window_w2', 'ai_window_c', 'ai_window_t')

    Prints:
      - date, article_id
      - hype_score_w0, hype_score_w1, hype_score_w2, hype_score_c
      - all ai_window* columns (sorted)
      - DIFFS: For each ordered pair (A, B), prints A \ B (what is in A but not in B).
    """
    # Validate structure
    required_base = ["date", "article_id"]
    required_hype = ["hype_score_w0", "hype_score_w1", "hype_score_w2", "hype_score_c"]
    missing = [c for c in required_base + required_hype if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    ai_window_cols = sorted([c for c in df.columns if c.startswith("ai_window")])
    if not ai_window_cols:
        raise ValueError("No AI window columns found (expected at least one 'ai_window*').")

    data = df.copy()

    # Ensure numeric hype for display
    for c in required_hype:
        data[c] = pd.to_numeric(data[c], errors="coerce").fillna(0.0)

    # Optional filter on nonzero hype
    if require_nonzero_hype:
        mask = (data[required_hype] != 0).any(axis=1)
        data = data.loc[mask].reset_index(drop=True)

    # Sort for deterministic traversal
    if "date" in data.columns:
        data = data.sort_values("date").reset_index(drop=True)

    n_articles = len(data)
    if n_articles == 0:
        print("No articles to display (empty after filtering).")
        return

    # Prepare ordered pairs for diffs
    ordered_pairs = list(_iter_ordered_pairs(ai_window_cols, pairs))

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
                print(_format_text(str(txt), wrap_width))
                print("-" * 80)

            # Print DIFFS (A \ B)
            if show_all_diffs and ordered_pairs:
                any_diff_printed = False
                for a, b in ordered_pairs:
                    ta = row.get(a, "") or ""
                    tb = row.get(b, "") or ""
                    diffs = _diff_units(
                        str(ta), str(tb),
                        granularity=diff_granularity,
                        similarity_threshold=similarity_threshold
                    )
                    if diffs:
                        if not any_diff_printed:
                            print("DIFFS (what is in A but not in B):")
                            any_diff_printed = True
                        print(f"  Δ {a} \\ {b}:")
                        for u in diffs:
                            wrapped = _format_text(u, wrap_width)
                            print(f"   • {wrapped}")
                        print("-" * 80)
                if not any_diff_printed:
                    print("No differences across AI windows at selected granularity.")
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

# -----------------------------------------------------------------------------
# Builders
# -----------------------------------------------------------------------------

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

def _ensure_ai_windows(df: pd.DataFrame, *, context_window: int, max_tokens: int = 512) -> pd.DataFrame:
    """
    Ensure df has 'ai_window' built with the requested context_window.
    If not present, (re)compute using the provided extractor and tag 'context_window_used'.
    """
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
