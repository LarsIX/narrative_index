"""
Annotation Disagreement Resolution Module

This module provides interactive utilities for resolving annotation disagreements between an author
and a second annotator on Wall Street Journal (WSJ) article labels related to AI narratives and
AI hype levels.

Functions
---------
- resolve_label_disagreements_AI(df_author, df_annotator) :
    Compares and interactively resolves binary 'AI-related' label mismatches between author and annotator.

- resolve_hype_disagreements(df_author, df_ai_level_mergen) :
    Compares and resolves disagreements in AI hype level annotations (0: none, 1: moderate, 2: strong).

Intended Use
------------
These tools are designed for small-scale or manual correction workflows during the annotation process.
They support in-notebook interaction via IPython Markdown rendering and command-line input prompts.

from IPython.display import display, Markdown
"""


def resolve_label_disagreements_AI(df_author, df_annotator):
    """
    Compares AI-related labels between author and annotator DataFrames,
    displays disagreements, and interactively allows corrections.

    Parameters
    ----------
    df_author : DataFrame
        DataFrame containing columns:
        - 'article_id'
        - 'label_ai_related'  
        with the author's annotations.

    df_annotator : DataFrame
        DataFrame containing columns:
        - 'article_id'
        - 'label_ai_related'
        - 'corpus'
        with the annotator’s annotations.

    Returns
    -------
    DataFrame
        Annotator DataFrame with possibly corrected 'label_ai_related' values
        and two tracking columns:
        - 'modified' (1 if label was changed)
        - 'hype_level_change' (initialized to 0)
    """

    # Create a copy to apply corrections
    df_ai_level_merge = df_annotator.copy()
    df_ai_level_merge['modified'] = 0
    df_ai_level_merge['hype_level_change'] = 0

    # Merge to find disagreements
    merged = df_author[['article_id', 'label_ai_related']].merge(
        df_annotator[['article_id', 'label_ai_related', 'corpus']],
        on='article_id',
        suffixes=('_author', '_annotator')
    )

    # Filter for label mismatches
    disagreements = merged[merged['label_ai_related_author'] != merged['label_ai_related_annotator']]
    print(f"Number of disagreements: {disagreements.shape[0]}")

    # Iterate and prompt for corrections
    for i, (_, row) in enumerate(disagreements.iterrows()):
        md_text = f"""
---
### Article {i+1}/{len(disagreements)}

**Article ID:** `{row['article_id']}`  
**Author Label:** `{row['label_ai_related_author']}`  
**Annotator Label:** `{row['label_ai_related_annotator']}`

---

#### Text:

{row['corpus']}

---
"""
        display(Markdown(md_text))

        change_label = input("Change label (default=annotator) ? (y/n): ").strip().lower()
        if change_label == 'y':
            new_label = 0 if row['label_ai_related_annotator'] == 1 else 1
            df_ai_level_merge.loc[df_ai_level_merge['article_id'] == row['article_id'], 'label_ai_related'] = new_label
            df_ai_level_merge.loc[df_ai_level_merge['article_id'] == row['article_id'], 'modified'] = 1
            print(f"Label changed to: {new_label}")
        else:
            print("Label not changed")

    return df_ai_level_merge

def resolve_hype_disagreements(df_author, df_ai_level_mergen):
    """
    Compares hype levels between author and annotator DataFrames,
    displays disagreements, and allows interactive corrections.

    Parameters
    ----------
    df_author : DataFrame
        DataFrame containing columns:
        - 'article_id'
        - 'hype_level'
        with the author’s annotations.

    df_ai_level_mergen : DataFrame
        DataFrame with the annotator’s values, containing columns:
        - 'article_id'
        - 'hype_level'
        - 'label_ai_related'
        - 'corpus'
        - 'modified'
        - 'hype_level_change'

    Returns
    -------
    DataFrame
        Annotator DataFrame with possibly corrected 'hype_level' values.
        Tracking columns 'modified' and 'hype_level_change' are updated accordingly.
    """

    # Create working copy
    df_final = df_ai_level_mergen.copy()
    if 'hype_level_change' not in df_final.columns:
        df_final['hype_level_change'] = 0
    if 'modified' not in df_final.columns:
        df_final['modified'] = 0

    # Merge to find disagreements
    merged = df_author[['article_id', 'hype_level']].merge(
        df_ai_level_mergen[['article_id', 'hype_level', 'label_ai_related', 'modified', 'corpus']],
        on='article_id',
        suffixes=('_author', '_annotator')
    )

    # Filter for mismatches
    disagreements = merged[merged['hype_level_author'] != merged['hype_level_annotator']]
    print(f"🔍 Number of hype level disagreements: {disagreements.shape[0]}")

    for i, (_, row) in enumerate(disagreements.iterrows()):
        md_text = f"""
    ---
    ### Article {i+1}/{len(disagreements)}

    **Article ID:** `{row['article_id']}`  
    **Author Hype Level:** `{row['hype_level_author']}`  
    **Annotator Hype Level:** `{row['hype_level_annotator']}`  
    **AI-related Label:** `{row['label_ai_related']}`  
    **Label changed from annotator's to author's view?:** `{"Yes" if row['modified'] == 1 else "No"}`

    ---

    #### Text:

    {row['corpus']}

    ---
    """
        display(Markdown(md_text))

        change = input("Change hype level (default=annotator)? (y/n): ").strip().lower()
        if change == 'y':
            while True:
                try:
                    new_value = int(input("Enter new hype level (0 = none, 1 = moderate, 2 = strong): ").strip())
                    if new_value not in [0, 1, 2]:
                        raise ValueError
                    break
                except ValueError:
                    print("Invalid input. Please enter 0, 1, or 2.")

            df_final.loc[df_final['article_id'] == row['article_id'], 'hype_level'] = new_value
            df_final.loc[df_final['article_id'] == row['article_id'], 'hype_level_change'] = 1
            print(f"Hype level updated to: {new_value}")
        else:
            print("Hype level not changed")

    return df_final
