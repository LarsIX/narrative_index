"""
Text preprocessing utilities for cleaning raw WSJ article corpora. 
Removes UI elements, metadata, promo content, and standardizes punctuation and spacing
"""

import re

# === Unified regex patterns to clean WSJ article texts ===
patterns_to_remove = [
    # UI & navigation elements
    r'\bAdvertisement\b',
    r'\bGift unlocked article\b',
    r'\bListen\b.*\(\d+ min\)',
    r'\bMost Popular\b', r'\bFurther Reading\b', r'\bRECOMMENDED\b',
    r'\bSHOW CONVERSATION \(\d+\)\b',
    r'\bVideos\b',
    r'\bGRAB A COPY\b', r'\bBUY BOOK\b', r'\bPreview\b', r'\bSubscribe\b',
    r'\bSAVE\b', r'\bSHARE\b', r'\bTEXT\b', r'RECOMMENDED VIDEOS',

    # Section headers and headline blocks
    r'^\s*[A-Z\s]{3,}\s*$',
    r'\n[A-Z\-]{5,20}\n',
    r'(?:\n[A-Z][^\n]{10,100}){2,10}$',
    r'(?:\n[A-Z][^\n]{10,120}){2,10}$',
    r'\r?\n[A-Z][^\n]{0,99}(?<![.!?:])(?:\r?\n[A-Z][^\n]{0,99}(?<![.!?:]))+\s*$',

    # Metadata and author info
    r'By\s+[A-Z][a-z]+\s+[A-Z][a-z]+',
    r'\bFollow\b',
    r'\bUpdated\b.*?ET',
    r'PHOTO: .*?\n',
    r'Appeared in the .*?edition.*?\n',
    r'Corrections & Amplifications.*?\n',
    r'\d{4,5}',
    r'\d+\s+RESPONSES',
    r'\*\*\*.*?\*\*\*',
    r'Best of the Web Today.*?\n',
    r'About this article.*?\n',
    r'^\s*By\s*\n.*?ET\n\d+\s+\d+\s+min\n',
    r'[A-Z]+\.\s+[A-Z]+\s+FOR\s+THE\s+WALL\s+STREET\s+JOURNAL\.',

    # Datetime & structure fragments
    r'\n?[A-Z\s]{5,}\s+.*?\d{1,2},\s*\d{2,4}\s+\d{1,4}\s*(am|pm)?\s*ET\n\d+\s+\d+\s+min\n(?:[^\n]{5,50}\n)?',
    r'\n?\.?\n?[A-Z\s]{5,}\s+.*?\d{1,2},\s*\d{0,4}\s+\d{1,2}[:.]?\d{2}\s*(am|pm)?\s*ET\n.*?\d+\s+\d+\s+min\n',
    r'\n?[A-Z\s]{5,}\n.*?(am|pm)?\s*ET\n\d+\s+\d+\s+min\n',
    r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\. \d{1,2}, \d{3,4} (am|pm)\. ET \d+ \d+ min\.',

    # Title = first sentence variant
    r'^((?:[A-Z][a-z]+\'?s?\.\s*){5,})$',

    # Repeatedly punctuated sentence fragments
    r'(?:(?:\b\w+\.\s*){4,})(?=\s|$)',
    r'\b(?:[A-Z][a-z]+\. ){2,10}[A-Z][a-z]+\b',

    # Illustration
    r'Photo illustration\..*',
    r'Illustration\..*',

    # Video & trailing promos
    r'Play video .*?\n',
    r'Play video\. [A-Z][a-z]+(?:\.[A-Z][a-z]+)+.*',
    r"SHOW CONVERSATION.*",
    r'Write to.*',

    # Newsletter, promotions, editor notes, podcasts
    r'NEWSLETTER\. SIGN[- ]?UP\..*?(?=\s{2,}|\n|$)',
    r'Catch up on the headlines.*?Enjoy a free article.*',
    r'Let the\. WSJ.*?prepare you.*',
    r"Our weekly markets news.*?News podcast.*?",
    r"Tech,\. Media\. Telecom.*?Dow\. Jones.*?",
    r"Check it out at wsj\.compodcasts.*?",
    r"For more\. WSJ.*?(sign up|check it out|wherever you listen).*",
    r"Editor's note.*?Wall\. Street\. Journal\.",

     # Stray teaser fragments
    r"China'?s trillion-yuan gamble on.*",                     
    r"China'?s tech giants.*alliances.*",    
    r"AI and chips is creating strange bedfellows.*",
    r"Employees'? days are bleeding.*",      
    r"The startup, growing frustrated.*",
    r"Huawei's with rival\. SMIC, for one to build a parallel ecosystem",    
]

# === Compile patterns once for performance ===
compiled_patterns = [re.compile(p, flags=re.IGNORECASE | re.MULTILINE) for p in patterns_to_remove]

def clean_article_corpus(text):
    """
    Cleans a raw WSJ article text by removing UI elements, navigation fragments,
    metadata, headers, and other editorial noise.

    It also normalizes punctuation, spacing, and sentence structure for
    downstream NLP applications (e.g., sentiment analysis, annotation).

    Parameters
    ----------
    text : str
        Raw article text (WSJ-specific formatting)

    Returns
    -------
    str
        Cleaned and normalized article text
    """
    if not isinstance(text, str):
        return text

    # Remove UI/promotional fragments
    for pattern in compiled_patterns:
        text = pattern.sub('', text)

    # Normalize quotes and special characters
    text = re.sub(r"[‘’´`]", "'", text)
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[^a-zA-Z0-9.,;!?\"'\n -]+", '', text)

    # Normalize spacing
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^\s+|\s+$", "", text)

    # Sentence boundary cleanup
    text = text.replace('\n', ' ')
    text = re.sub(r"(?<![.!?])\s+(?=[A-Z])", ". ", text)

    return text.strip()
