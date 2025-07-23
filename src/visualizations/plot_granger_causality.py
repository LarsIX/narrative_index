import math
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Define default figure output path relative to the project root
base_path = Path(__file__).resolve().parent.parent
fig_path = base_path / "reports" / "figures"

def plot_aini_lags_by_year(gc_by_year, lag='t-1', aini_var_name="Variable", figsize=(14, 5), path=None):
    """
    Create strip plots of Granger regression coefficients (e.g., t-1) across years and AINI variants.

    Each subplot represents one year, with points grouped by AINI variant and colored by ticker symbol.
    Only coefficients with non-missing values for the specified lag are shown.

    Parameters
    ----------
    gc_by_year : pd.DataFrame
        Granger causality results indexed by 'Year'. Must contain columns for lags and 'Ticker'.
    lag : str, optional
        The column name of the regression coefficient to plot (e.g., 't-1').
    aini_var_name : str, optional
        The name of the AINI variable column in the DataFrame (usually 'AINI_variant' or 'Variable').
    figsize : tuple, optional
        Size of each subplot row (width, height).
    path : Path, optional
        Path to save the figure. Defaults to 'reports/figures' in the project root.
    """

    if path is None:
        path = fig_path

    gc_by_year_sub = gc_by_year[gc_by_year[lag].notna()]
    years = gc_by_year_sub.index.unique()
    n_years = len(years)
    n_cols = 3
    n_rows = math.ceil(n_years / n_cols)

    all_tickers = sorted(gc_by_year_sub['Ticker'].unique())
    palette = sns.color_palette("tab20", len(all_tickers))
    ticker_colors = dict(zip(all_tickers, palette))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows), sharey=True)
    axes = axes.flatten()

    for i, year in enumerate(years):
        ax = axes[i]
        data = gc_by_year_sub[gc_by_year_sub.index == year]

        sns.stripplot(
            data=data,
            x=aini_var_name,
            y=lag,
            hue='Ticker',
            hue_order=all_tickers,
            palette=ticker_colors,
            dodge=True,
            ax=ax
        )

        ax.set_title(f"Year: {year}")
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xlabel('Variable')
        ax.set_ylabel('Lag Coefficient' if i % n_cols == 0 else '')
        ax.tick_params(axis='x', rotation=45)

        if ax.get_legend():
            ax.legend_.remove()

    for j in range(len(years), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Granger Causality Regression Coefficient for Lag: {lag} (p < 0.05)", fontsize=16, y=1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Ticker', bbox_to_anchor=(1.02, 0.9), loc='upper left')

    path.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path / f'granger_{lag}_by_year.png')
    print(f'granger_{lag}_by_year.png saved to {path}')
    plt.show()

def plot_aini_lags_for_year(gc_df, year, aini_var_name="Variable", lag_col=None, figsize=(14, 5), path=None):
    """
    Visualize all Granger regression coefficients for a given year across AINI variants and tickers.

    Creates one subplot per lag (e.g., t-1 to t-10) for a specific year, grouping by AINI variant and
    coloring by ticker. Missing values are excluded from each lag plot.

    Parameters
    ----------
    gc_df : pd.DataFrame
        Granger causality results with columns for year, lag coefficients, ticker, and AINI variant.
    year : int or str
        The year to plot results for (must match entries in 'Year' column).
    aini_var_name : str, optional
        Column name representing the AINI variant, default is 'Variable'.
    lag_col : list of str, optional
        List of lag coefficient column names to include (e.g., ['t-1', 't-2', ...]).
        If None, columns starting with 't-' will be auto-detected.
    figsize : tuple, optional
        Figure size per row (width, height).
    path : Path, optional
        Directory to save the resulting figure. Defaults to 'reports/figures'.
    """

    if path is None:
        path = fig_path

    df_year = gc_df[gc_df["Year"] == year]

    if lag_col is None:
        lag_col = [col for col in df_year.columns if col.startswith("t-")]

    n_lags = len(lag_col)
    n_cols = 3
    n_rows = math.ceil(n_lags / n_cols)

    all_tickers = sorted(df_year['Ticker'].unique())
    palette = sns.color_palette("tab20", len(all_tickers))
    ticker_colors = dict(zip(all_tickers, palette))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows), sharey=True)
    axes = axes.flatten()

    for i, lag in enumerate(lag_col):
        ax = axes[i]
        data = df_year[df_year[lag].notna()]

        sns.stripplot(
            data=data,
            x=aini_var_name,
            y=lag,
            hue='Ticker',
            hue_order=all_tickers,
            palette=ticker_colors,
            dodge=True,
            ax=ax
        )

        ax.set_title(f"Lag: {lag}")
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xlabel(aini_var_name)
        ax.set_ylabel('Coefficient' if i % n_cols == 0 else '')
        ax.tick_params(axis='x', rotation=45)

        if ax.get_legend():
            ax.legend_.remove()

    for j in range(n_lags, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Granger Regression Coefficients by Lag – Year {year}", fontsize=16, y=1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Ticker', bbox_to_anchor=(1.02, 0.9), loc='upper left')

    path.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path / f'granger_coefficient_lags_{year}.png')
    print(f'granger_coefficient_lags_{year}.png saved to {path}')
    plt.show()



def plot_avg_beta_lags(gc_df, year, aini_var_name="Variable", figsize=(14, 5), path=None):
    """
    Plot average beta coefficients over lag ranges (e.g., avg_beta_l10, avg_beta_l15, etc.)
    for each AINI variant, colored by ticker.
    """
    if path is None:
        path = Path("reports/figures")

    df_year = gc_df[gc_df["Year"] == year]

    # auto-detect all avg_beta_l* columns
    avg_cols = sorted([col for col in df_year.columns if col.startswith("avg_beta_l") and col[11:]])
    
    if not avg_cols:
        print(f"No 'avg_beta_l*' columns found for year {year}.")
        return

    n_lags = len(avg_cols)
    n_cols = 3
    n_rows = math.ceil(n_lags / n_cols)

    all_tickers = sorted(df_year['Ticker'].unique())
    palette = sns.color_palette("tab20", len(all_tickers))
    ticker_colors = dict(zip(all_tickers, palette))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows), sharey=True)
    axes = axes.flatten()

    for i, col in enumerate(avg_cols):
        ax = axes[i]
        data = df_year[df_year[col].notna()]

        sns.stripplot(
            data=data,
            x=aini_var_name,
            y=col,
            hue='Ticker',
            hue_order=all_tickers,
            palette=ticker_colors,
            dodge=True,
            ax=ax
        )

        ax.set_title(f"{col}")
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xlabel(aini_var_name)
        ax.set_ylabel('Avg Coefficient' if i % n_cols == 0 else '')
        ax.tick_params(axis='x', rotation=45)

        if ax.get_legend():
            ax.legend_.remove()

    for j in range(n_lags, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Average Granger Betas by Lag Window – Year {year}", fontsize=16, y=1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Ticker', bbox_to_anchor=(1.02, 0.9), loc='upper left')

    path.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path / f'avg_beta_lags_{year}.png')
    print(f'avg_beta_lags_{year}.png saved to {path}')
    plt.show()
