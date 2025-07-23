import sqlite3
import os
import sys
from pathlib import Path

def create_db(year, folder=None):
    """
    Create a SQLite database to store WSJ articles.

    Parameters
    ----------
    year : int or str
        Year used for naming the database file.
    folder : str, optional
        Output path to save the database. 
        If None, defaults to "data/raw/articles" relative to project root.

    Returns
    -------
    dict
        Metadata for the created database:
        - "output_path" : str
            Full path to the created SQLite database.
        - "year" : int or str
            Input year used in the file name.

    Notes
    -----
    The database contains the following tables:

    - `articles_index`: Metadata and URLs of WSJ articles.
    - `article`: Full content of scraped articles, linked to `articles_index`.
    - `exploration`: Status tracking for scraping by date and page.
    """
    root = Path(__file__).parent.parent.parent

    if not folder:
        folder = root / "data" / "raw" / "articles"

    os.makedirs(folder, exist_ok=True)
    name = os.path.join(folder, f"articleswsj_{year}.db")
    conn = sqlite3.connect(name)
    c = conn.cursor()

    # Table "articles_index"
    c.execute('''CREATE TABLE IF NOT EXISTS articles_index (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            
            year TEXT, 
            month TEXT, 
            day TEXT, 
            
            headline TEXT, 
            article_time TEXT,
            
            keyword TEXT,
            link TEXT, 
            
            scraped_at TEXT,
            scanned_status INTEGER)''')

    # Table "article"
    c.execute('''CREATE TABLE IF NOT EXISTS article (
                    article_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_src TEXT,

                    scanned_time TEXT,
                    title TEXT,
                    sub_title TEXT,


                    corpus TEXT,
                    index_id INTEGER,

                    FOREIGN KEY(index_id) REFERENCES articles_index(id))''')


    c.execute('''CREATE TABLE IF NOT EXISTS exploration (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            link TEXT,
            
            year TEXT, 
            month TEXT, 
            day TEXT, 
            page_num TEXT,
            
            checked_at TEXT,
            values_or_not INTEGER,
            count_articles INTEGER
            );
    ''')

    conn.commit()
    conn.close()

    return {
        "output_path" : name,
        "year": year
    }
