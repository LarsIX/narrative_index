"""
Interactive Article Annotation Module

This module provides interactive tools to annotate news articles for:
- AI relevance (`label_ai_related`, `relevance`, or `about_ai`)
- AI hype level or sentiment (`hype_level`, `hype_score`)
- Annotation rationale (`comments`) for model training or few-shot use

Designed primarily for use in Jupyter Notebooks, these tools facilitate human-in-the-loop labeling
of financial news articles, e.g., from the Wall Street Journal, for training and evaluating NLP models.

Functions
---------
- annotate_articles_with_hype(df, min_words=30, save_path=None):
    Classifies articles as AI-related and assigns a hype level (0–2) interactively.

- annotate_articles_hype_pos_neg(df, min_words=30, save_path=None):
    Classifies articles as AI-related and assigns a sentiment-based hype score (−1 to +1).

- find_training_examples(df, min_words=30, save_path=None, context_window=1):
    Interactive annotation function for manually labeling articles with relevance tags
    ("0", "1", "2") and optional comments. Outputs are suitable for few-shot learning
    with ChatPromptTemplate or OpenAI API-based LLMs.

- naive_labeling(df, title_col="title", text_col="corpus", output_col="about_ai"):
    Assigns a binary label (`about_ai`) based on the presence of AI-related keywords
    in the title or corpus, using simple pattern matching. Useful for rule-based pre-labeling
    or weak supervision.

Intended Use
------------
Use this module during the manual labeling phase of a supervised learning pipeline to:
- Construct high-quality training data for transformer-based models like FinBERT
- Create few-shot prompts with human-validated context windows
- Apply rule-based filtering for noisy or unlabeled corpora
- Capture rationale to improve explainability and prompt engineering
"""


from tqdm import tqdm
import pandas as pd
from IPython.display import display
import os
from IPython.display import Markdown
import sys
from pathlib import Path
import re

# Add the src directory to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# import window from modelling
from modelling.ai_windows import extract_human_readable_snippet

def annotate_articles_with_hype(df, min_words=30, save_path=None):
    """
    Interactive function to annotate news articles for AI relevance and hype level.
    Optimized for Jupyter Notebooks using Markdown display.

    Parameters:
    ----------
    df : DataFrame
        DataFrame containing at least the columns:
        'article_id', 'title', 'sub_title', 'corpus'

    min_words : int
        Minimum number of words required in an article to be shown

    save_path : str or None
        If provided, annotations are autosaved to this path after each step

    Returns:
    -------
    DataFrame
        Updated DataFrame with 'label_ai_related' and 'hype_level' columns
    """
    
    # Add annotation columns if they don't exist
    if 'label_ai_related' not in df.columns:
        df['label_ai_related'] = None
    if 'hype_level' not in df.columns:
        df['hype_level'] = None

    for i, row in df.iterrows():
        # Skip already labeled entries
        if pd.notnull(row['label_ai_related']):
            continue

        # Skip articles with too few words
        word_count = len(str(row['corpus']).split())
        if word_count < min_words:
            print(f"\n Skipping short article ({word_count} words)...")
            continue

        # Display article metadata and cleaned text in Markdown format
        display(Markdown(f"""
---
### Article {i+1}/{len(df)} — ID: `{row['article_id']}`  
** Title:** {row['title']}  
** Sub-title:** {row['sub_title'] if pd.notnull(row['sub_title']) else "*[No subtitle]*"}  
** Length:** {word_count} words  

---

#### Cleaned Text:
{row['corpus']}
"""))

        # Prompt user for AI relevance
        label = input(" Is this article related to AI? (y = yes / n = no / s = skip / q = quit): ").strip().lower()

        if label == 'y':
            df.loc[i, 'label_ai_related'] = 1

            # Prompt for hype level only if AI-related
            print("""
 HYPE LEVEL (based on tone and framing):
  0 = Low / No hype (technical, neutral, skeptical)
  1 = Moderate hype (optimistic or moderately fearful)
  2 = High hype (bold claims, euphoric or fear-driven urgency)
""")
            hype = input(" What is the AI hype level? (0 / 1 / 2): ").strip()
            if hype in ['0', '1', '2']:
                df.loc[i, 'hype_level'] = int(hype)
            else:
                print(" Invalid hype input. Marking as missing.")
                df.loc[i, 'hype_level'] = None

        elif label == 'n':
            df.loc[i, 'label_ai_related'] = 0
            df.loc[i, 'hype_level'] = None

        elif label == 's':
            # Skip this entry without annotating
            continue

        elif label == 'q':
            print("Annotation manually stopped.")
            break

        else:
            print("Invalid input — skipping this article.")
            continue

        # Autosave after every annotation if path is provided
        if save_path:
            try:
                df.to_csv(save_path, index=False)
                print(f" Autosaved to: {os.path.basename(save_path)}")
            except Exception as e:
                print("Error while saving:", e)

        print(f"Done. Total labeled: {df['label_ai_related'].notnull().sum()} / {len(df)}")

    print("\n Annotation session completed.")
    return df[['article_id', 'title', 'sub_title', 'corpus', 'label_ai_related', 'hype_level']]



def find_training_examples(df, min_words=30, save_path=None, context_window=1):
    """
    Interactive annotation function for creating few-shot training examples
    for ChatPromptTemplate-based prompting (e.g. OpenAI or LangChain).

    Each article will be labeled for AI relevance:
        - "0" = Not related to AI
        - "1" = Related to AI, easy to identify
        - "2" = Related to AI, but subtle or complex

    Additionally, optional comments can be added as rationale.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: 'article_id', 'title', 'sub_title', 'corpus'

    min_words : int
        Minimum number of words in title + corpus to include the article.

    save_path : str or Path, optional
        CSV path for autosaving after each annotation step.
        Defaults to: /data/processed/articles/sampled_GPT_ex_windsize_{context_window}.csv

    context_window : int
        Number of surrounding sentences included around AI keywords.

    Returns
    -------
    pd.DataFrame
        With added columns:
        - 'ai_window'   : Context-aware snippet used for prompting
        - 'relevance'   : One of {"0", "1", "2"}
        - 'comments'    : Optional annotator rationale
    """

    # Set default save path if not provided
    if save_path is None:
        root = Path(__file__).resolve().parent.parent
        save_path = root / "data" / "processed" / "articles" / f"sampled_GPT_ex_windsize_{context_window}.csv"

    # Ensure annotation columns exist
    if 'relevance' not in df.columns:
        df['relevance'] = None
    if 'comments' not in df.columns:
        df['comments'] = None

    # Extract windowed snippets using external function
    df = extract_human_readable_snippet(
        df,
        title_col="title",
        text_col="corpus",
        output_col="ai_window",
        context_window=context_window
    )

    # Annotation loop
    for i, row in df.iterrows():
        if pd.notnull(row['relevance']):
            continue
        if len(str(row['title']) + " " + str(row['corpus'])).split() < min_words:
            continue

        display(Markdown(f"""
---
### Article {i+1}/{len(df)} — ID: `{row['article_id']}`  
---

#### Context Snippet:
{row['ai_window']}
"""))

        label = input("AI relevance? [0 = Not related, 1 = Related (Easy), 2 = Related (Hard), s = skip, q = quit]: ").strip().lower()

        if label == 'q':
            print("Annotation manually stopped.")
            break
        elif label == 's':
            continue
        elif label in ['0', '1', '2']:
            df.at[i, 'relevance'] = label
            label_name = {"0": "Not related", "1": "Related (Easy)", "2": "Related (Hard)"}[label]
            print(f"Labeled as {label}: {label_name}")
        else:
            print("Invalid input — skipping this article.")
            continue

        comment = input("Optional comment (or press Enter to skip): ").strip()
        if comment:
            df.at[i, 'comments'] = comment

        try:
            df.to_csv(save_path, index=False)
            print(f"Autosaved to: {Path(save_path).name}")
        except Exception as e:
            print("Error while saving:", e)

        total_labeled = df['relevance'].notnull().sum()
        print(f"Labeled total: {total_labeled} / {len(df)}\n")

    print("Annotation session completed.")
    return df[['article_id', 'title', 'sub_title', 'corpus', 'ai_window', 'relevance', 'comments']]




def naive_labeling(df, title_col="title", text_col="corpus", output_col="about_ai"):
    """
    Labels articles as AI-related (1) or not (0) based on presence of AI keywords in title or corpus.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns for title and corpus text.

    title_col : str
        Column name for article titles.

    text_col : str
        Column name for article text bodies.

    output_col : str
        Name of the output column indicating AI relevance (0 or 1).

    Returns
    -------
    pd.DataFrame
        Original dataframe with additional binary column 'about_ai'.
    """
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

    tqdm.pandas(desc="Naive AI keyword labeling")

    def check_ai_relevance(title, text):
        combined = f"{str(title)} {str(text)}"
        return int(bool(ai_pattern.search(combined)))

    df[output_col] = df.progress_apply(
        lambda row: check_ai_relevance(row[title_col], row[text_col]),
        axis=1
    )

    return df
