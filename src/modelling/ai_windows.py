"""
Module: ai_windows.py

Functions for extracting AI-related text snippets from financial articles.
These windows are used to fine-tune or infer with models like FinBERT.

Includes:
- extract_ai_context_window: Single match using a simple AI keyword list.
- extract_largedict_ai_context_window: Broader keyword list.
- extract_multiple_ai_snippets_with_context: Multiple matches with contextual sentences.

Dependencies:
- spaCy (en_core_web_sm)
- transformers
"""

import re
import spacy
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

# Load spaCy English model for sentence segmentation
nlp = spacy.load("en_core_web_sm")


# Extract a context window around the first AI-related keyword using a simple keyword list.
# Truncates to the first N tokens (max_tokens), using sentence-level boundaries.
def extract_ai_context_window(df, text_col='corpus', output_col='ai_window', tokenizer_name='bert-base-uncased', max_tokens=512):
    """
    Extract a single snippet around the first AI-related keyword in each text.

    This function identifies the first occurrence of an AI-related keyword,
    segments the text into sentences, and accumulates them up to a max token
    length using the provided tokenizer.

    Args:
        df (pd.DataFrame): DataFrame containing a text column.
        text_col (str): Name of the column with the raw text.
        output_col (str): Name of the new column to store the snippet.
        tokenizer_name (str): HuggingFace tokenizer name or path.
        max_tokens (int): Maximum token length for each extracted snippet.

    Returns:
        pd.DataFrame: DataFrame with a new column of AI-focused snippets.
    """
        
    # Define a basic list of AI-related keywords/patterns
    ai_keywords = [
        r'\bAI\b', r'\bA\.I\.\b', r'\bartificial intelligence\b',
        r'\bmachine learning\b', r'\bdeep learning\b', r'\bLLM\b',
        r'\bGPT[-\d]*\b', r'\bChatGPT\b', r'\bOpenAI\b',
        r'\btransformer model\b', r'\bgenerative AI\b', r'\bneural network\b',
    ]
    ai_pattern = re.compile('|'.join(ai_keywords), flags=re.IGNORECASE)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Internal helper function to extract a snippet from a single text entry
    def truncate_to_ai(text):
        if not isinstance(text, str):
            return ""

        # Search for first AI keyword match
        match = ai_pattern.search(text)
        if not match:
            return text[:2000]  # fallback: truncate to first 2000 characters

        # Segment text into sentences using spaCy
        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents]

        # Accumulate sentences until reaching max token limit
        token_count = 0
        context = []
        for sent in sentences:
            context.append(sent)
            token_count += len(tokenizer.tokenize(sent))
            if token_count >= max_tokens:
                break

        # Final truncation (in case a sentence pushed it slightly over limit)
        tokens = tokenizer.tokenize(' '.join(context))
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        return tokenizer.convert_tokens_to_string(tokens)

    # Apply to DataFrame
    df[output_col] = df[text_col].apply(truncate_to_ai)
    return df


# Similar to above, but with an extended keyword dictionary (for more exhaustive AI-related matches)
def extract_largedict_ai_context_window(df, text_col='corpus', output_col='ai_window', tokenizer_name='bert-base-uncased', max_tokens=512):
    """
    Extracts a single context window around the first match from an extended set of AI-related keywords.

    This function identifies the first AI-related keyword in the text using a comprehensive keyword list.
    It then segments the text into sentences and accumulates them until reaching the specified token limit.
    The resulting snippet is tokenized and truncated accordingly.

    Args:
        df (pd.DataFrame): DataFrame containing a column of raw text.
        text_col (str): Name of the column containing full article text (default: 'corpus').
        output_col (str): Name of the column to store the extracted snippet (default: 'ai_window').
        tokenizer_name (str): HuggingFace-compatible tokenizer name (default: 'bert-base-uncased').
        max_tokens (int): Maximum number of tokens for each snippet (default: 512).

    Returns:
        pd.DataFrame: Modified DataFrame with a new column containing AI-related context windows.
    """
    # Extended AI keyword list capturing broader terminology
    ai_keywords = [
        r'\bAI\b', r'\bA\.I\.\b', r'\bAGI\b', r'\bartificial intelligence\b',
        r'\bartificial general intelligence\b', r'\bhuman-level AI\b',
        r'\bhuman-centered artificial intelligence\b', r'\blarge language model\b',
        r'\blarge language models\b', r'\bLLM\b', r'\balgorithmic bias\b',
        r'\bmachine learning\b', r'\bsupervised learning\b', r'\bunsupervised learning\b',
        r'\breinforcement learning\b', r'\bdeep learning\b', r'\bneural network\b',
        r'\bartificial neural network\b', r'\btransformer model\b', r'\btransformers\b',
        r'\bfine[- ]?tuning\b', r'\bgenerative AI\b', r'\bgenerative adversarial networks?\b',
        r'\bGANs?\b', r'\bprompt engineering\b', r'\bprompts?\b', r'\bhallucination\b',
        r'\bautonomous systems\b', r'\bspeech recognition\b', r'\bfacial recognition\b',
        r'\bsocial chatbots\b', r'\bhuman-robot interaction\b', r'\bautomated decision[- ]making\b',
        r'\bsuperintelligence\b', r'\bcomputer vision\b', r'\bdeepfakes?\b',
        r'\bfoundation models?\b', r'\bfrontier AI\b', r'\bgraphical processing units?\b',
        r'\bGPUs?\b', r'\binterpretability\b', r'\bopen[- ]source\b', r'\bresponsible AI\b',
        r'\btraining datasets?\b'
    ]
    ai_pattern = re.compile('|'.join(ai_keywords), flags=re.IGNORECASE)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Internal helper to extract context window around first match
    def truncate_to_ai(text):
        if not isinstance(text, str):
            return ""

        match = ai_pattern.search(text)
        if not match:
            return text[:2000]

        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents]

        token_count = 0
        context = []
        for sent in sentences:
            context.append(sent)
            token_count += len(tokenizer.tokenize(sent))
            if token_count >= max_tokens:
                break

        tokens = tokenizer.tokenize(' '.join(context))
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        return tokenizer.convert_tokens_to_string(tokens)

    # Apply to DataFrame
    df[output_col] = df[text_col].apply(truncate_to_ai)
    return df


# Extract multiple AI-related snippets with a variable number of surrounding sentences (context_window)
def extract_multiple_ai_snippets_with_context(
    df,
    text_col='corpus',
    output_col='ai_window',
    tokenizer_name='bert-base-uncased',
    max_tokens=512,
    context_window=1
):
    """
    Extracts multiple AI-related snippets with surrounding sentence context from each document.

    This method finds **all** sentences that match AI-related keywords using an extended keyword set.
    For each match, it collects a specified number of surrounding sentences (left and right),
    combines them into a single snippet, and truncates the result by token length.

    Args:
        df (pd.DataFrame): DataFrame containing a column with full article text.
        text_col (str): Name of the column with the article text (default: 'corpus').
        output_col (str): Name of the column to store the extracted AI snippets (default: 'ai_window').
        tokenizer_name (str): Tokenizer name or path compatible with HuggingFace Transformers.
        max_tokens (int): Maximum number of tokens allowed per snippet (default: 512).
        context_window (int): Number of sentences to include before and after each matched sentence (default: 1).

    Returns:
        pd.DataFrame: DataFrame with a new column containing one or more AI-related context snippets per document.
    """
    # Compile pattern only once
    ai_keywords = [
        r'\bAI\b', r'\bA\.I\.\b', r'\bAGI\b', r'\bartificial intelligence\b', r'\bartificial general intelligence\b',
        r'\bhuman-level AI\b', r'\bhuman-centered artificial intelligence\b', r'\blarge language models?\b',
        r'\bLLM\b', r'\balgorithmic bias\b', r'\bmachine learning\b', r'\bsupervised learning\b',
        r'\bunsupervised learning\b', r'\breinforcement learning\b', r'\bdeep learning\b',
        r'\bneural network\b', r'\bartificial neural network\b', r'\btransformer model\b', r'\btransformers\b',
        r'\bfine[- ]?tuning\b', r'\bgenerative AI\b', r'\bgenerative adversarial networks?\b', r'\bGANs?\b',
        r'\bprompt engineering\b', r'\bprompts?\b', r'\bhallucination\b', r'\bautonomous systems\b',
        r'\bspeech recognition\b', r'\bfacial recognition\b', r'\bsocial chatbots\b', r'\bhuman-robot interaction\b',
        r'\bautomated decision[- ]making\b', r'\bsuperintelligence\b', r'\bcomputer vision\b', r'\bdeepfakes?\b',
        r'\bfoundation models?\b', r'\bfrontier AI\b', r'\bgraphical processing units?\b',
        r'\bGPUs?\b', r'\binterpretability\b', r'\bopen[- ]source\b', r'\bresponsible AI\b',
        r'\btraining datasets?\b'
    ]
    ai_pattern = re.compile('|'.join(ai_keywords), flags=re.IGNORECASE)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Internal helper
    def extract_snippets(text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return ""

        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        selected = set()

        for i, sent in enumerate(sentences):
            if ai_pattern.search(sent):
                for offset in range(-context_window, context_window + 1):
                    idx = i + offset
                    if 0 <= idx < len(sentences):
                        selected.add(idx)

        combined = ' '.join([sentences[i] for i in sorted(selected)])
        if not combined.strip():
            return ""

        encoded = tokenizer(
            combined,
            truncation=True,
            padding=False,
            max_length=max_tokens,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False
        )

        return tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)

    # Replace apply() with tqdm-loop
    tqdm.pandas(desc="🔍 Extracting AI snippets")
    df[output_col] = df[text_col].progress_apply(extract_snippets)

    return df

# Extract multiple AI-related snippets with a variable number of surrounding sentences (context_window) from text and headline
def extract_multiple_ai_snippets_title_context(
    df,
    text_col='cleaned_corpus',
    output_col='ai_window',
    tokenizer_name='bert-base-uncased',
    max_tokens=512,
    context_window=1
):
    """
    Extracts multiple AI-related snippets with surrounding sentence context from each document.

    This method finds **all** sentences that match AI-related keywords using an extended keyword set.
    For each match, it collects a specified number of surrounding sentences (left and right),
    combines them into a single snippet, and truncates the result by token length.

    Args:
        df (pd.DataFrame): DataFrame containing a column with full article text.
        text_col (str): Name of the column with the article text (default: 'corpus').
        output_col (str): Name of the column to store the extracted AI snippets (default: 'ai_window').
        tokenizer_name (str): Tokenizer name or path compatible with HuggingFace Transformers.
        max_tokens (int): Maximum number of tokens allowed per snippet (default: 512).
        context_window (int): Number of sentences to include before and after each matched sentence (default: 1).

    Returns:
        pd.DataFrame: DataFrame with a new column containing one or more AI-related context snippets per document.
    """
    # Load spaCy for sentence splitting
    nlp = spacy.load("en_core_web_sm")

    # Compile pattern only once
    ai_keywords = [
        r'\bAI\b', r'\bA\.I\.\b', r'\bAGI\b', r'\bartificial intelligence\b', r'\bartificial general intelligence\b',
        r'\bhuman-level AI\b', r'\bhuman-centered artificial intelligence\b', r'\blarge language models?\b',
        r'\bLLM\b', r'\balgorithmic bias\b', r'\bmachine learning\b', r'\bsupervised learning\b',
        r'\bunsupervised learning\b', r'\breinforcement learning\b', r'\bdeep learning\b',
        r'\bneural network\b', r'\bartificial neural network\b', r'\btransformer model\b', r'\btransformers\b',
        r'\bfine[- ]?tuning\b', r'\bgenerative AI\b', r'\bgenerative adversarial networks?\b', r'\bGANs?\b',
        r'\bprompt engineering\b', r'\bprompts?\b', r'\bhallucination\b', r'\bautonomous systems\b',
        r'\bspeech recognition\b', r'\bfacial recognition\b', r'\bsocial chatbots\b', r'\bhuman-robot interaction\b',
        r'\bautomated decision[- ]making\b', r'\bsuperintelligence\b', r'\bcomputer vision\b', r'\bdeepfakes?\b',
        r'\bfoundation models?\b', r'\bfrontier AI\b', r'\bgraphical processing units?\b',
        r'\bGPUs?\b', r'\binterpretability\b', r'\bopen[- ]source\b', r'\bresponsible AI\b',
        r'\btraining datasets?\b'
    ]
    ai_pattern = re.compile('|'.join(ai_keywords), flags=re.IGNORECASE)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Internal helper
    def extract_snippets(text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return ""

        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        selected = set()

        for i, sent in enumerate(sentences):
            if ai_pattern.search(sent):
                for offset in range(-context_window, context_window + 1):
                    idx = i + offset
                    if 0 <= idx < len(sentences):
                        selected.add(idx)

        combined = ' '.join([sentences[i] for i in sorted(selected)])
        if not combined.strip():
            return ""

        encoded = tokenizer(
            combined,
            truncation=True,
            padding=False,
            max_length=max_tokens,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False
        )

        return tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)

    # track progress
    tqdm.pandas(desc="Extracting AI snippets")

    # concatenate title + text
    df["text"] = df["title"] + " "+ df[text_col]
    df[output_col] = df["text"].progress_apply(extract_snippets)

    return df

def extract_human_readable_snippet(
    df,
    title_col="title",
    text_col="cleaned_corpus",
    output_col="ai_window",
    tokenizer_name="bert-base-uncased",
    max_tokens=512,
    context_window=1
):
    """
    Combines title and corpus, filters for AI keyword matches, applies sentence context window,
    and returns a human-readable snippet within max token length.
    """
    # Define and compile AI keyword pattern
    ai_keywords = [
        r'\bAI\b', r'\bA\.I\.\b', r'\bAGI\b', r'\bartificial intelligence\b', r'\bartificial general intelligence\b',
        r'\bhuman-level AI\b', r'\blarge language models?\b', r'\bLLM\b', r'\bmachine learning\b',
        r'\bsupervised learning\b', r'\breinforcement learning\b', r'\bdeep learning\b',
        r'\bneural networks?\b', r'\btransformers?\b', r'\bGANs?\b', r'\bgenerative AI\b',
        r'\bprompt engineering\b', r'\bhallucination\b', r'\bautonomous systems\b',
        r'\bfoundation models?\b', r'\btraining datasets?\b', r'\bcomputer vision\b',
        r'\binterpretability\b', r'\bresponsible AI\b', r'\bopen[- ]source\b'
    ]
    ai_pattern = re.compile("|".join(ai_keywords), flags=re.IGNORECASE)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def extract(text_title, text_body):
        # Combine title and corpus
        full_text = f"{text_title.strip()} {text_body.strip()}"

        if not isinstance(full_text, str) or len(full_text.strip()) == 0:
            return ""

        doc = nlp(full_text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        matched_idxs = set()

        # Find all AI keyword matches and expand with context
        for i, sent in enumerate(sentences):
            if ai_pattern.search(sent):
                for offset in range(-context_window, context_window + 1):
                    idx = i + offset
                    if 0 <= idx < len(sentences):
                        matched_idxs.add(idx)

        if not matched_idxs:
            return ""

        # Order and accumulate while keeping within token limit
        snippet = ""
        for idx in sorted(matched_idxs):
            candidate = (snippet + " " + sentences[idx]).strip() if snippet else sentences[idx]
            token_count = len(tokenizer.tokenize(candidate))
            if token_count > max_tokens:
                break
            snippet = candidate

        return snippet.strip()

    # Apply row-wise with tqdm
    tqdm.pandas(desc="Extracting AI keyword context windows")
    df[output_col] = df.progress_apply(
        lambda row: extract(row[title_col], row[text_col]),
        axis=1
    )

    return df
