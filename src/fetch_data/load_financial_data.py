import yfinance as yf
import pandas as pd
from datetime import datetime
from pathlib import Path

# Data fetcher function
def fetch_and_save_data(start_date, end_date, save_dir, tickers=None):
    """
    Downloads and stores financial OHLCV data for a list of tickers from Yahoo Finance.

    Parameters
    ----------
    start_date : str
        Start date in format 'YYYY-MM-DD'.
    end_date : str
        End date in format 'YYYY-MM-DD'.
    save_dir : Path
        Directory where CSV files will be stored.
    tickers : list of str, optional
        List of ticker symbols to download. If None, uses a default list of AI-related stocks and ETFs.

    Returns
    -------
    tuple
        A tuple of:
        - file_name (str): Filename of the merged CSV file.
        - save_dir (Path): Output directory path where the file is saved.
    """
    if tickers is None:
        # Ticker list
        tickers = [
            # ETFs
            "ROBO", "ARKQ", "BOTZ", "AIQ", "IRBO",

            # Magnificent 7
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",

            # Additional AI-related leaders
            "AVGO", "AMD", "TSM", 

            # controls
            "^SOX", "^GSPC"
        ]

    # Download OHLCV data
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, group_by='ticker')
    year_range = f"{start_date[:4]}_{end_date[:4]}"

    # Save individual ticker files
    for ticker in tickers:
        df_ticker = data[ticker].copy()
        df_ticker.reset_index(inplace=True)
        df_ticker.to_csv(save_dir / f"{ticker}_full_{year_range}.csv", index=False)

    # Save merged long format
    long_form = []
    for ticker in tickers:
        df = data[ticker].copy()
        df["Ticker"] = ticker
        df.reset_index(inplace=True)
        long_form.append(df)

    df_all = pd.concat(long_form)
    df_all = df_all[["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    file_name = f"full_daily_{year_range}.csv"
    df_all.to_csv(save_dir / file_name, index=False)

    return file_name, save_dir
