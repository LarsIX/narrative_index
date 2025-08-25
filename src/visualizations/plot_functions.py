import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union
from pathlib import Path
from typing import Optional, Tuple, Sequence
import numpy as np



def plot_aini_series_subplots(
    df: pd.DataFrame,
    date_col: str = "date",
    bases: Sequence[str] = ("normalized_AINI", "EMA_02", "EMA_08"),
    suffixes: Sequence[str] = ("w0", "w1", "w2"),
    window_labels: Optional[Dict[str, str]] = None,   # e.g. {"w0":"±0d","w1":"±1d","w2":"±2d"}
    events: Optional[Dict[str, Union[str, pd.Timestamp]]] = None,
    event_style: Optional[Dict] = None,               # overrides for vlines/text
    dpi: int = 300,
    figsize: tuple = (12, 7.5),
    outpath: Optional[Union[str, Path]] = None,       # if given, saves PNG; use outpaths to write PDF/SVG too
    outpaths_vector: Optional[Dict[str, Union[str, Path]]] = None,  # {"pdf":"fig.pdf","svg":"fig.svg"}
    ylabels: Optional[Dict[str, str]] = None,         # per-base y-axis labels
    title: str = "AINI Time Series with Key AI Events",
    grid: bool = True,
    linewidth: float = 1.8,
    alpha: float = 0.95,
    rc_update: Optional[Dict] = None,                 # e.g. {"font.size":11, "axes.titlesize":12}
):
    """
    Plot {bases} by date for each suffix in {suffixes} in stacked subplots.
    Designed for scientific reporting: clean formatting, reproducible defaults,
    robust to missing series, and vector export.

    Parameters
    ----------
    df : DataFrame with a date column and columns like f"{base}_{suffix}".
    date_col : datetime-like column name.
    bases : base variable names to plot (rows).
    suffixes : context windows (series per subplot).
    window_labels : mapping from suffix -> legend label (defaults to suffix).
    events : mapping {label: date} where date is 'YYYY-MM-DD' or pd.Timestamp.
             Defaults include Sam Altman firing and DeepSeek emergence.
    event_style : dict to override line/text styling for events.
    dpi, figsize : rendering settings.
    outpath : if set, saves PNG here.
    outpaths_vector : optional dict to additionally save PDF/SVG, e.g. {"pdf": ".../figure.pdf"}.
    ylabels : optional per-base y-axis labels.
    title : figure title.
    grid : toggle subplot grid.
    linewidth, alpha : series aesthetics.
    rc_update : optional rcParams overrides.
    """

    # ---- Reproducible, neutral style (no seaborn) ----
    # Narrow, journal-friendly defaults; user can override via rc_update.
    rc_defaults = {
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": grid,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "legend.frameon": False,
    }
    if rc_update:
        rc_defaults.update(rc_update)
    with mpl.rc_context(rc_defaults):

        # ---- Input prep ----
        df = df.copy()
        if date_col not in df.columns:
            raise ValueError(f"date_col '{date_col}' not found in DataFrame.")
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)

        # Default labels for windows
        if window_labels is None:
            window_labels = {s: s for s in suffixes}

        # Default events if none provided (you can extend/replace)
        if events is None:
            events = {
                "Sam Altman fired": "2023-11-17",
                # If you want “GPT-5 rumors”, add the most defensible date you track.
                "DeepSeek emerges": "2025-01-20",
            }
        # Parse event dates
        events = {k: pd.to_datetime(v) for k, v in events.items()}

        # Event styling defaults
        evt_style = dict(color="red", linestyle="--", alpha=0.7, linewidth=1.2)
        if event_style:
            evt_style.update(event_style)

        # Color-blind friendly palette for up to 3 series
        # (okabe-ito): blue, orange, green
        palette = {
            "w0": "#0072B2",  # blue
            "w1": "#E69F00",  # orange
            "w2": "#009E73",  # green
            "custom":"#9E0008",  # green
        }
        # Fallbacks if more suffixes than keys
        default_colors = list(palette.values()) + ["#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]
        color_map = {}
        for i, s in enumerate(suffixes):
            color_map[s] = palette.get(s, default_colors[i % len(default_colors)])

        # ---- Figure & axes ----
        nrows = len(bases)
        fig, axes = plt.subplots(nrows, 1, figsize=figsize, sharex=True)
        if nrows == 1:
            axes = [axes]

        # Date axis formatter: compact and journal-friendly
        locator = AutoDateLocator(minticks=4, maxticks=8)
        formatter = ConciseDateFormatter(locator)

        # ---- Plot each base ----
        for i, base in enumerate(bases):
            ax = axes[i]
            plotted_any = False

            for suf in suffixes:
                col = f"{base}_{suf}"
                if col in df.columns and df[col].notna().any():
                    ax.plot(
                        df[date_col],
                        df[col],
                        label=window_labels.get(suf, suf),
                        color=color_map[suf],
                        linewidth=linewidth,
                        alpha=alpha,
                    )
                    plotted_any = True

            # Axes labels and grid
            ax.set_ylabel((ylabels or {}).get(base, base), labelpad=6)

            # Legend only if something was plotted for this subplot
            if plotted_any:
                ax.legend(title="Window", ncol=min(3, len(suffixes)), loc="upper left")
            else:
                # If nothing to show, make it apparent in the plot area
                ax.text(
                    0.5, 0.5, f"No data for base '{base}'",
                    transform=ax.transAxes, ha="center", va="center", fontsize=10
                )

            # Event markers: draw after data so they’re visible
            # Use axis fraction for y to avoid overlap with data ranges
            ax_ymin, ax_ymax = ax.get_ylim()
            y_top = ax_ymax - 0.02 * (ax_ymax - ax_ymin)
            v_spacing = 0.08 * (ax_ymax - ax_ymin)
            for j, (label, date) in enumerate(events.items()):
                ax.axvline(date, **evt_style)
                # Annotate just below the top; rotate for space efficiency
                ax.annotate(
                    label,
                    xy=(date, y_top - j * v_spacing),
                    xytext=(5, 0),
                    textcoords="offset points",
                    rotation=90,
                    va="top",
                    fontsize=9,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1)
                )

            # Dates
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

        axes[-1].set_xlabel("Date")

        # Title and layout
        fig.suptitle(title, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        # ---- Saving ----
        def _ensure_parent(p: Path):
            p.parent.mkdir(parents=True, exist_ok=True)

        if outpath is not None:
            outpath = Path(outpath)
            _ensure_parent(outpath)
            fig.savefig(outpath, bbox_inches="tight")

        if outpaths_vector:
            for ext, p in outpaths_vector.items():
                p = Path(p)
                _ensure_parent(p)
                fig.savefig(p, bbox_inches="tight")

        plt.show()

# function to plot distribution of an AINI variable by years
def plot_aini_hist_grid_by_years(
    df,
    measure: str = "normalized_AINI_w1",
    date_col: str = "date",
    years: list = [2023, 2024, 2025],
    bins: int = 40,
    size: str = "half",  # "half" (default) or "full"
    dpi: int = 300,
    outpath: Path | str | None = None,
):
    """
    2x2 histogram grid for `measure`: All data, 2023, 2024, 2025.
    - Counts (not density)
    - μ, σ, median, N shown in box per subplot
    - Figure size adapted for half or full A4 page

    Parameters
    ----------
    df : DataFrame
        Must contain `date` and `measure`.
    measure : str
        Column name to plot.
    date_col : str
        Date column name.
    years : list[int]
        List of years for subplots (default: [2023,2024,2025]).
    bins : int
        Number of histogram bins.
    size : {"half","full"}
        "half" = ~half A4 width, "full" = full-page figure.
    dpi : int
        Resolution of output file.
    outpath : Path or str, optional
        Save location for figure.
    """

    # Set figure size depending on intended usage
    if size == "half":
        figsize = (7.5, 5.5)   # half-page
    elif size == "full":
        figsize = (10, 7.5)    # full-page
    else:
        raise ValueError("size must be 'half' or 'full'")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[measure] = pd.to_numeric(df[measure], errors="coerce")

    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    axes = axes.flatten()

    vals_all = df[measure].dropna()
    if vals_all.empty:
        raise ValueError(f"No data in column '{measure}'.")

    # Shared x-range (robust)
    p1, p99 = np.nanpercentile(vals_all, [1, 99])
    span = max(p99 - p1, 1e-9)
    x_min = p1 - 0.05 * span
    x_max = p99 + 0.05 * span

    # Build subsets
    subsets = [("All", vals_all)]
    for y in years:
        vals_y = df.loc[df[date_col].dt.year == y, measure].dropna()
        subsets.append((str(y), vals_y))

    for ax, (label, vals) in zip(axes, subsets):
        if vals.empty:
            ax.set_visible(False)
            continue

        # Histogram (counts)
        ax.hist(
            vals,
            bins=bins,
            range=(x_min, x_max),
            density=False,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.6,
            color="tab:blue",
        )

        # Summary stats
        mean_val = float(np.nanmean(vals))
        std_val = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else float("nan")
        median_val = float(np.nanmedian(vals))

        ax.text(
            0.98, 0.95,
            f"μ={mean_val: .3f}\nσ={std_val: .3f}\nmedian={median_val: .3f}\nN={len(vals)}",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2),
        )

        ax.set_title(f"{measure} ({label})", fontsize=10)
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel(measure, fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(f"Distribution of {measure}: All vs {', '.join(map(str, years))}",
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, bbox_inches="tight", dpi=dpi)

    plt.show()