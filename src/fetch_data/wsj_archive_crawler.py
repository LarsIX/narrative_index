import requests
from bs4 import BeautifulSoup
import sqlite3
import time
from datetime import datetime, timedelta
import re
import os

# Default path to the SQLite database
DEFAULT_DB_PATH = os.path.join("data", "raw", "articles", "articlesWSJ.db")

# Sections to include in the crawl
ALLOWED_SECTIONS = {
    'business', 'opinion', 'world', 'us-news', 'tech', 'us', 'markets',
    'finance', 'economy', 'politics', 'review-outlook', 'business-world', 'commentary'
}


class ManagementDB:
    """
    Interface for managing SQLite operations related to WSJ article metadata.

    This class handles:
    - Inserting article metadata into the 'articles_index' table.
    - Logging page exploration results into the 'exploration' table.
    - Checking for duplicate links.
    - Managing database connections.

    Parameters
    ----------
    db_path : str, optional
        Path to the SQLite database file. Defaults to `data/raw/articles/articlesWSJ.db`.
    """

    def __init__(self, db_path=DEFAULT_DB_PATH):
        self.conn = sqlite3.connect(db_path)
        self.c = self.conn.cursor()

    def insert_elements(self, elements):
         """
        Insert metadata for a single article into the 'articles_index' table.

        Parameters
        ----------
        elements : dict
            Dictionary with keys: 'headline', 'article_time', 'year', 'month',
            'day', 'keyword', 'link', 'scraped_at', 'scanned_status'.
        """
        try:
            self.c.execute("""
                INSERT INTO articles_index 
                (headline, article_time, year, month, day, keyword, link, scraped_at, scanned_status) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                elements["headline"], elements["article_time"], elements["year"], elements["month"], elements["day"],
                elements["keyword"], elements["link"], elements["scraped_at"], elements["scanned_status"]
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"❌ Insert Error: {e}")

    def exploration(self, link, day, month, year, page_num, values_or_not, count_articles):
        """
        Log the result of crawling a WSJ archive page.

        Parameters
        ----------
        link : str
            Full URL of the archive page.
        day : int
            Day of the archive.
        month : int
            Month of the archive.
        year : int
            Year of the archive.
        page_num : int
            Page number within the archive day.
        values_or_not : int
            1 if any articles were found, 0 otherwise.
        count_articles : int
            Number of articles parsed on the page.
        """
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.c.execute("""
                INSERT INTO exploration 
                (link, day, month, year, page_num, checked_at, values_or_not, count_articles)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (link, day, month, year, page_num, current_time, values_or_not, count_articles))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"❌ Exploration Logging Error: {e}")

    def link_exists(self, link):
        """
        Check whether a link already exists in the 'articles_index' table.

        Parameters
        ----------
        link : str
            The article URL to check.

        Returns
        -------
        bool
            True if the link is already in the database, False otherwise.
        """
        self.c.execute("SELECT 1 FROM articles_index WHERE link = ?", (link,))
        return self.c.fetchone() is not None

    def close(self):
        """
        Close the database connection.
        """
        self.conn.close()


class WSJMetadataCrawler:
    """
    Crawler for scraping WSJ article metadata from daily archive pages.

    This class navigates through the WSJ archive by day and page,
    extracts metadata for relevant articles, and stores it in a SQLite database.

    Parameters
    ----------
    db_path : str, optional
        Path to the SQLite database file. Default is `DEFAULT_DB_PATH`.
    allowed_sections : set of str, optional
        Set of allowed section tags to include (e.g., 'business', 'tech').
        Defaults to a pre-defined set `ALLOWED_SECTIONS`.
    """

    def __init__(self, db_path=DEFAULT_DB_PATH, allowed_sections=None):
        self.db_path = db_path
        self.allowed_sections = allowed_sections or ALLOWED_SECTIONS
        self.page_number = 1
        self.total_articles = 0

    def reset(self):
        """
        Reset crawler state (page counter and article counter).
        """
        self.page_number = 1
        self.total_articles = 0

    def get_elements_from_web(self, year, month, day, wait=5):
        """
        Scrape WSJ archive page(s) for a specific date and store article metadata.

        Parameters
        ----------
        year : int
            Target year of the archive.
        month : int
            Target month of the archive.
        day : int
            Target day of the archive.
        wait : int, optional
            Delay in seconds between page requests. Default is 5.
        """
        db = ManagementDB(self.db_path)

        while True:
            url = f"https://www.wsj.com/news/archive/{year}/{month:02}/{day:02}"
            if self.page_number > 1:
                url += f"?page={self.page_number}"

            print(f"\n🔎 Scraping URL: {url}")
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)

            if response.status_code != 200:
                print(f"❌ Page error: {response.status_code}")
                break

            soup = BeautifulSoup(response.content, 'html.parser')
            ol_element = soup.find('ol', class_='WSJTheme--list-reset--3pR-r52l')
            article_elements = ol_element.find_all('article') if ol_element else []

            if not article_elements:
                db.exploration(url, day, month, year, self.page_number, 0, 0)
                print(f"✅ Done with {day}-{month}-{year}. Total articles: {self.total_articles}")
                break

            count_articles = 0
            for article in article_elements:
                headline = article.find('span', class_='WSJTheme--headlineText--He1ANr9C')
                a_tag = article.find('a')
                timestamp = article.find('p', class_='WSJTheme--timestamp--22sfkNDv')
                section_div = article.find('div', class_='WSJTheme--articleType--34Gt-vdG')

                # Parse section info
                if section_div and section_div.find('div'):
                    section_raw = section_div.find('div').text
                    section = re.sub(r'\s+', '-', section_raw.strip().lower())
                    section = re.sub(r'[^\w\-]', '', section)
                else:
                    section = "N/A"

                # Prepare metadata dictionary
                article_data = {
                    'headline': headline.text if headline else 'N/A',
                    'article_time': timestamp.text if timestamp else 'N/A',
                    'year': year,
                    'month': month,
                    'day': day,
                    'keyword': section,
                    'link': a_tag['href'] if a_tag else 'N/A',
                    'scraped_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'scanned_status': 0
                }

                # Save if relevant and not already present
                if section in self.allowed_sections and not db.link_exists(article_data['link']):
                    db.insert_elements(article_data)
                    count_articles += 1

            db.exploration(url, day, month, year, self.page_number, 1, count_articles)
            self.total_articles += count_articles
            self.page_number += 1
            time.sleep(wait)

        db.close()
        self.reset()


def get_dates(year):
    """
    Generate all calendar dates for a given year.

    Parameters
    ----------
    year : int
        Year for which all dates should be generated.

    Returns
    -------
    list of list
        List of [day, month, year] triplets for each day of the year.
    """
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31)
    return [[d.day, d.month, d.year] for d in (start + timedelta(days=n) for n in range((end - start).days + 1))]


def search_year(year, wait=5, db_path=DEFAULT_DB_PATH, allowed_sections=None):
    """
    Crawl WSJ archive metadata for each day of a given year.

    Parameters
    ----------
    year : int
        The year to be crawled.
    wait : int, optional
        Delay in seconds between each day’s crawling. Default is 5.
    db_path : str, optional
        Path to the SQLite database file. Defaults to DEFAULT_DB_PATH.
    allowed_sections : set of str, optional
        Set of allowed section identifiers. If None, defaults to ALLOWED_SECTIONS.
    """
    crawler = WSJMetadataCrawler(db_path=db_path, allowed_sections=allowed_sections)
    for day, month, y in get_dates(year):
        print(f"📅 Processing {day}-{month}-{y}")
        crawler.get_elements_from_web(y, month, day, wait)
        time.sleep(wait)
