"""
Interactive Article Annotation Module

This module provides interactive tools to annotate news articles for:
- AI relevance (`label_ai_related` or `relevance`)
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

- annotate_articles_with_window(df, min_words=30, save_path=None, max_tokens=512, context_window=1):
    First extracts human-readable AI-relevant snippets using a Transformer tokenizer window,
    then classifies and annotates them interactively using the −1 to +1 hype scale.

- find_training_examples(df, min_words=30, save_path=None, max_tokens=512, context_window=1):
    Interactive annotation function for manually labeling articles with relevance tags
    ("0", "1e", "1h") and optional comments. Outputs are suitable for few-shot learning
    with ChatPromptTemplate or OpenAI API-based LLMs.

Intended Use
------------
Use this module during the manual labeling phase of a supervised learning pipeline to:
- Construct high-quality training data for transformer-based models like FinBERT
- Create few-shot prompts with human-validated context windows
- Capture rationale to improve explainability and prompt engineering
"""



import pandas as pd
from IPython.display import display
import os
from IPython.display import Markdown
import sys
from pathlib import Path

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
    # import necessary libraries
    from IPython.display import display, Markdown   
    
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
            print(f"\n⏭️ Skipping short article ({word_count} words)...")
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
📈 HYPE LEVEL (based on tone and framing):
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
                print(f"💾 Autosaved to: {os.path.basename(save_path)}")
            except Exception as e:
                print("Error while saving:", e)

        print(f"Done. Total labeled: {df['label_ai_related'].notnull().sum()} / {len(df)}")

    print("\n Annotation session completed.")
    return df[['article_id', 'title', 'sub_title', 'corpus', 'label_ai_related', 'hype_level']]


import pandas as pd
from IPython.display import display, Markdown
import os

def annotate_articles_hype_pos_neg(df, min_words=30, save_path=None):
    """
    Interactive annotation function for AI relevance and AI hype level 
    on a -1 to +1 scale.

    Parameters:
    ----------
    df : pd.DataFrame
        Must contain: 'article_id', 'title', 'sub_title', 'corpus'

    min_words : int
        Minimum number of words to include article in annotation loop.

    save_path : str or None
        Optional CSV path for autosaving after each annotation.

    Returns:
    -------
    pd.DataFrame
        With additional columns: 'label_ai_related', 'hype_score'
    """
    # Add annotation columns if they don't exist
    if 'label_ai_related' not in df.columns:
        df['label_ai_related'] = None
    if 'hype_score' not in df.columns:
        df['hype_score'] = None

    for i, row in df.iterrows():
        if pd.notnull(row['label_ai_related']):
            continue

        word_count = len(str(row['corpus']).split())
        if word_count < min_words:
            print(f"\n Skipping short article ({word_count} words)...")
            continue

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

        # Ask for AI relevance
        label = input(" Is this article related to AI? (y = yes / n = no / s = skip / q = quit): ").strip().lower()

        if label == 'y':
            df.loc[i, 'label_ai_related'] = 1

            print("""
  AI HYPE SCALE (−2 to +2):
  -1 = negative (skeptical, cautious)
   0 = No AI-relation
  +1 = positive (hopeful, positive tone)
""")
            hype = input("What is the AI hype score? (-2 to +2): ").strip()
            if hype in ['-1', '0', '1']:
                df.loc[i, 'hype_score'] = int(hype)
            else:
                print("Invalid hype input. Marking as missing.")
                df.loc[i, 'hype_score'] = None

        elif label == 'n':
            df.loc[i, 'label_ai_related'] = 0
            df.loc[i, 'hype_score'] = None

        elif label == 's':
            continue

        elif label == 'q':
            print("Annotation manually stopped.")
            break

        else:
            print("Invalid input — skipping this article.")
            continue

        if save_path:
            try:
                df.to_csv(save_path, index=False)
                print(f"Autosaved to: {os.path.basename(save_path)}")
            except Exception as e:
                print("Error while saving:", e)

        print(f"Done. Total labeled: {df['label_ai_related'].notnull().sum()} / {len(df)}")

    print("\n Annotation session completed.")
    return df[['article_id', 'title', 'sub_title', 'corpus', 'label_ai_related', 'hype_score']]

def annotate_articles_with_window(df, min_words=30, save_path=None,  max_tokens=512,
    context_window=1):
    """
    Interactive annotation function for AI relevance and AI hype level 
    on a -1 to +1 scale.

    Parameters:
    ----------
    df : pd.DataFrame
        Must contain: 'article_id', 'title', 'sub_title', 'corpus'

    min_words : int
        Minimum number of words to include article in annotation loop.

    save_path : str or None
        Optional CSV path for autosaving after each annotation.

    Returns:
    -------
    pd.DataFrame
        With additional columns: 'label_ai_related', 'hype_score'
    """
    # Add annotation columns if they don't exist
    if 'label_ai_related' not in df.columns:
        df['label_ai_related'] = None
    if 'hype_score' not in df.columns:
        df['hype_score'] = None

    # apply window function
    df = extract_human_readable_snippet(df,
    title_col="title",
    text_col="corpus",
    output_col="ai_window",
    tokenizer_name="bert-base-uncased",
    max_tokens=max_tokens,
    context_window=context_window)

    for i, row in df.iterrows():
        if pd.notnull(row['label_ai_related']):
            continue

        display(Markdown(f"""
---
### Article {i+1}/{len(df)} — ID: `{row['article_id']}`  
---

#### Cleaned Text:
{row['ai_window']}
"""))

        # Ask for AI relevance
        label = input("Is this article related to AI? (y = yes / n = no / s = skip / q = quit): ").strip().lower()

        if label == 'y':
            df.loc[i, 'label_ai_related'] = 1

            print("""
 AI HYPE SCALE (−2 to +2):
  -1 = negative (skeptical, cautious)
   0 = No AI-relation
  +1 = positive (hopeful, positive tone)
""")
            hype = input("What is the AI hype score? (-2 to +2): ").strip()
            if hype in ['-1', '0', '1']:
                df.loc[i, 'hype_score'] = int(hype)
            else:
                print("Invalid hype input. Marking as missing.")
                df.loc[i, 'hype_score'] = None

        elif label == 'n':
            df.loc[i, 'label_ai_related'] = 0
            df.loc[i, 'hype_score'] = None

        elif label == 's':
            continue

        elif label == 'q':
            print("⏹️ Annotation manually stopped.")
            break

        else:
            print("Invalid input — skipping this article.")
            continue

        if save_path:
            try:
                df.to_csv(save_path, index=False)
                print(f"Autosaved to: {os.path.basename(save_path)}")
            except Exception as e:
                print("Error while saving:", e)

        print(f"Done. Total labeled: {df['label_ai_related'].notnull().sum()} / {len(df)}")

    print("\n Annotation session completed.")
    return df[['article_id', 'title', 'sub_title', 'corpus', 'ai_window', 'label_ai_related', 'hype_score']]


def find_shots(df, save_path=None, max_tokens=512, context_window=1):
    """
    Interactive annotation function for AI relevance and ease of annotation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain: 'article_id', 'title', 'sub_title', 'corpus'

    save_path : str or None
        Optional CSV path for autosaving after each annotation.
    
    Returns
    -------
    pd.DataFrame
        Annotated dataframe with 'relevance' column.
    """
    
    # Add annotation column if missing
    if 'relevance' not in df.columns:
        df['relevance'] = None

    # Apply window function (assumed implemented elsewhere)
    df = extract_human_readable_snippet(
        df,
        title_col="title",
        text_col="corpus",
        output_col="ai_window",
        tokenizer_name="bert-base-uncased",
        max_tokens=max_tokens,
        context_window=context_window
    )

    for i, row in df.iterrows():
        # Skip already labeled
        if pd.notnull(row['relevance']):
            continue

        display(Markdown(f"""
---
### Article {i+1}/{len(df)} — ID: `{row['article_id']}`  
#### Title: {row['title']}
---

#### Text Snippet:
{row['ai_window']}
"""))

        # Get user input
        label = input("AI relevance? [0 = Not related, 1e = Related (Easy), 1h = Related (Hard), s = skip, q = quit]: ").strip().lower()

        if label == 'q':
            print("⏹️ Annotation manually stopped.")
            break
        elif label == 's':
            continue
        elif label in ['0', '1e', '1h']:
            df.at[i, 'relevance'] = label
        else:
            print("import pandas as pd
from IPython.display import display, Markdown
import os

def find_training_examples(df, min_words=30, save_path=None, max_tokens=512, context_window=1):
    """
    Interactive annotation function for creating few-shot training examples
    for ChatPromptTemplate-based prompting (e.g. OpenAI or LangChain).

    Each article will be labeled for AI relevance:
        - "0"  = Not related to AI
        - "1e" = Related to AI, easy to identify
        - "1h" = Related to AI, but subtle or complex

    Additionally, optional comments can be added as rationale.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with required columns: 'article_id', 'title', 'sub_title', 'corpus'.

    min_words : int
        Minimum number of words in 'corpus' for an article to be shown.

    save_path : str or None
        Optional CSV path to autosave after each annotation step.

    max_tokens : int
        Token limit for the extracted article snippet (used for model context).

    context_window : int
        Number of surrounding sentences included around AI keywords.

    Returns
    -------
    pd.DataFrame
        The same dataframe with added columns:
        - 'ai_window'   : Context-aware snippet used for prompting
        - 'relevance'   : One of {"0", "1e", "1h"}
        - 'comments'    : Optional annotator rationale
    """

    # Add annotation columns if missing
    if 'relevance' not in df.columns:
        df['relevance'] = None
    if 'comments' not in df.columns:
        df['comments'] = None

    # Apply window function (user-defined function assumed)
    df = extract_human_readable_snippet(
        df,
        title_col="title",
        text_col="corpus",
        output_col="ai_window",
        tokenizer_name="bert-base-uncased",
        max_tokens=max_tokens,
        context_window=context_window
    )

    for i, row in df.iterrows():
        if pd.notnull(row['relevance']):
            continue
        if len(str(row['corpus']).split()) < min_words:
            continue

        display(Markdown(f"""
---
### Article {i+1}/{len(df)} — ID: `{row['article_id']}`  
#### Title: {row['title']}
---

#### Text Snippet:
{row['ai_window']}
"""))

        label = input("AI relevance? [0 = Not related, 1e = Related (Easy), 1h = Related (Hard), s = skip, q = quit]: ").strip().lower()

        if label == 'q':
            print("Annotation manually stopped.")
            break
        elif label == 's':
            continue
        elif label in ['0', '1e', '1h']:
            df.at[i, 'relevance'] = label
            label_name = {
                '0': "Not related",
                '1e': "Related (Easy)",
                '1h': "Related (Hard)"
            }.get(label)
            print(f"Labeled as {label}: {label_name}")
        else:
            print("Invalid input — skipping this article.")
            continue

        comment = input("Optional comment (or press Enter to skip): ").strip()
        if comment:
            df.at[i, 'comments'] = comment

        if save_path:
            try:
                df.to_csv(save_path, index=False)
                print(f"Autosaved to: {os.path.basename(save_path)}")
            except Exception as e:
                print("Error while saving:", e)

        total_labeled = df['relevance'].notnull().sum()
        print(f"Labeled total: {total_labeled} / {len(df)}\n")

    print("Annotation session completed.")
    return df[['article_id', 'title', 'sub_title', 'corpus', 'ai_window', 'relevance', 'comments']]