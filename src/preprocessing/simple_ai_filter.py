import re
import pandas as pd

def flag_ai_mentions(df, text_column='corpus', verbose=True):
    """
    Adds a boolean column 'mentions_ai' to the DataFrame indicating presence of AI-related terms.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing article texts.
        text_column (str): Column name containing the article text to search.
        verbose (bool): If True, prints a summary of AI-related mentions.
    
    Returns:
        pd.DataFrame: The original DataFrame with an additional 'mentions_ai' column.
    """
    # Define AI-related keywords and phrases
    ai_keywords = [
        r'\bAI\b',
        r'\bA\.I\.\b',
        r'\bartificial intelligence\b',
        r'\bmachine learning\b',
        r'\bdeep learning\b',
        r'\bLLM\b',
        r'\bGPT[-\d]*\b',
        r'\bChatGPT\b',
        r'\bOpenAI\b',
        r'\btransformer model\b',
        r'\bgenerative AI\b',
        r'\bneural network\b',
    ]

    # Compile combined regex pattern
    ai_pattern = re.compile('|'.join(ai_keywords), flags=re.IGNORECASE)

    # Apply to each row
    df['mentions_ai'] = df[text_column].apply(lambda x: bool(ai_pattern.search(str(x))))

    if verbose:
        count = df['mentions_ai'].sum()
        total = len(df)
        print(f"{count} out of {total} articles mention AI-related topics.")

    return df