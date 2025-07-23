'''
Includes helper functions to extract tickers to later predict Transfer Entropy and Granger Causality.
'''

import pandas as pd
import numpy as np

def get_ticker_for_TE(fin_data, aini_data, aini_var):
    """
    Prepares aligned time series arrays for Transfer Entropy (TE) analysis.

    Parameters
    ----------
    fin_data : pd.DataFrame
        Financial data containing columns 'Date', 'LogReturn', and 'Ticker'.
    aini_data : pd.DataFrame
        Sentiment data with a 'date' column and multiple sentiment variants.
    aini_var : str
        Name of the AINI variable (sentiment index) to be aligned.

    Returns
    -------
    dict
        Dictionary mapping each ticker to a 2xN NumPy array:
        [ [log_return_1, ..., log_return_N],
          [aini_value_1, ..., aini_value_N] ]
    """

    array_dict = {}

    # Ensure datetime format for merge
    fin_data['Date'] = pd.to_datetime(fin_data['Date'])
    aini_data['date'] = pd.to_datetime(aini_data['date'])

    # Iterate over all tickers
    for ticker, group in fin_data.groupby('Ticker'):
        # Extract log returns
        log_returns = group[['Date', 'LogReturn']].copy()

        # Merge returns and sentiment data on date
        aligned = pd.merge(
            log_returns,
            aini_data[['date', aini_var]],
            left_on='Date',
            right_on='date',
            how='inner'
        )

        # Convert to array and stack vertically (2 rows: log_return, AINI)
        log_return_array = aligned['LogReturn'].to_numpy().flatten()
        aini_array = aligned[aini_var].to_numpy().flatten()
        array_dict[ticker] = np.vstack([log_return_array, aini_array])

    return array_dict


def get_ticker_for_granger(fin_data, aini_data, aini_var):
    """
    Prepares aligned time series arrays for Granger causality testing.

    Parameters
    ----------
    fin_data : pd.DataFrame
        Financial data containing 'Date', 'LogReturn', and 'Ticker' columns.
    aini_data : pd.DataFrame
        Sentiment data with a 'date' column and sentiment index columns.
    aini_var : str
        The specific AINI variable (sentiment index) to align with financial data.

    Returns
    -------
    dict
        Dictionary mapping each ticker to a NumPy array of shape (N, 2),
        where column 0 = log returns and column 1 = AINI values.
    """

    array_dict = {}

    # Ensure datetime compatibility
    fin_data['Date'] = pd.to_datetime(fin_data['Date'])
    aini_data['date'] = pd.to_datetime(aini_data['date'])

    # Iterate over tickers
    for ticker, group in fin_data.groupby('Ticker'):
        # Extract relevant columns
        log_returns = group[['Date', 'LogReturn']].copy()

        # Join sentiment and return data by date
        aligned = pd.merge(
            log_returns,
            aini_data[['date', aini_var]],
            left_on='Date',
            right_on='date',
            how='inner'
        )

        # Combine into 2D NumPy array (N, 2): [LogReturn, AINI]
        array = np.column_stack([
            aligned['LogReturn'].to_numpy(),
            aligned[aini_var].to_numpy()
        ])

        array_dict[ticker] = array

    return array_dict
