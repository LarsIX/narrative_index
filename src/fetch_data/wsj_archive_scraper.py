import time
import sqlite3
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException


class WSJScraper:
    """
    A Selenium-based scraper that extracts full WSJ article content from URLs stored in a SQLite database.

    The scraper loads pages in a full browser session (Chrome), handles cookie banners, and writes
    article titles, subtitles, and content into a separate database table.
    """

    def __init__(self, db_path: str):
     """
        Initialize the scraper with a path to the database and launch a browser session.

        Parameters
        ----------
        db_path : str
            Path to the SQLite database containing WSJ article metadata.
        """
     
        self.db_path = db_path
        self.url = "https://www.wsj.com/"
        self.driver = self.create_driver()

    def create_driver(self):
        """
        Create a Chrome driver instance connected to a browser running with remote debugging.

        Returns
        -------
        selenium.webdriver.Chrome
            A Chrome driver instance connected to an open browser.
        """

        options = Options()
        options.debugger_address = "127.0.0.1:9222"
        driver = webdriver.Chrome(options=options)
        driver.get(self.url)
        time.sleep(7)  # Ensure the browser loads completely
        return driver

    def close_cookie_banner(self):
        """
        Attempt to close the WSJ cookie consent banner.

        Notes
        -----
        If the banner is not present, this function fails silently.
        """

        try:
            btn = self.driver.find_element(By.CSS_SELECTOR, "button.agree-btn[title='YES, I AGREE']")
            btn.click()
            print("🍪 Cookie-Banner accepted")
            time.sleep(4)
        except NoSuchElementException:
            print("ℹ️ No Cookie-Banner found.")

    def get_article_links(self, limit=None, year=None, month=None, day=None):
        """
        Fetch a list of unscreened article links for scraping.

        Parameters
        ----------
        limit : int, optional
            Maximum number of article links to fetch.
        year : int, optional
            Filter articles by year.
        month : int, optional
            Filter articles by month.
        day : int, optional
            Filter articles by day.

        Returns
        -------
        list of tuple
            List of (article_id, link) pairs for articles yet to be scraped.
        """

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        if year and month and day:
            query = """
                SELECT id, link FROM articles_index
                WHERE scanned_status = 0 AND year=? AND month=? AND day=?
            """
            params = [year, month, day]
        else:
            query = "SELECT id, link FROM articles_index WHERE scanned_status = 0"
            params = []

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        c.execute(query, tuple(params))
        rows = c.fetchall()
        conn.close()
        return rows

    def scrape_article(self, article_id: int, url: str):
        """
        Scrape the full text, title, and subtitle of a WSJ article.

        The extracted content is stored in the `article` table, and the scan status is updated.

        Parameters
        ----------
        article_id : int
            ID of the article from the `articles_index` table.
        url : str
            URL of the WSJ article to scrape.

        Notes
        -----
        This function handles different possible layouts for WSJ content containers.
        If no meaningful content is found, the article is skipped.
        """

        print(f"➡️ Opening article {article_id}: {url}")
        self.driver.get(url)
        time.sleep(7)
        self.close_cookie_banner()

        # Try to extract title
        try:
            title = self.driver.find_element(By.CSS_SELECTOR, "h1").text.strip()
        except NoSuchElementException:
            title = ""

        # Try to extract subtitle
        try:
            subtitle = self.driver.find_element(By.CSS_SELECTOR, "h2").text.strip()
        except NoSuchElementException:
            subtitle = ""

        # Try multiple content container strategies
        content = ""
        selectors = [
            "section.ef4qpkp0.css-y2scx8-Container.e1of74uw18",  # new WSJ format
            "div.article-content",  # fallback
            "section",
            "main"
        ]

        for sel in selectors:
            try:
                el = self.driver.find_element(By.CSS_SELECTOR, sel)
                text = el.text.strip()
                if len(text) > len(content):
                    content = text
            except NoSuchElementException:
                continue

        if not content:
            print(f"⚠️ No text found in {article_id}, skipping...")
            return

        # Prepare and save data
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("""
                INSERT INTO article (image_src, scanned_time, title, sub_title, corpus, index_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ('', now, title, subtitle, content, article_id))
            c.execute("UPDATE articles_index SET scanned_status=1 WHERE id=?", (article_id,))
            conn.commit()
            conn.close()
            print(f"✅ Article {article_id} saved successfully.")
        except sqlite3.Error as e:
            print(f"❌ DB Error at article {article_id}: {e}")

    def close(self):
        """
        Close the Selenium browser session.
        """
        
        self.driver.quit()
