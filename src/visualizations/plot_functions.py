import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from pathlib import Path
from typing import Dict, Optional, Sequence, Union,Tuple, List
import numpy as np
import re


def plot_aini_means_three_panels(
    df: pd.DataFrame,
    date_col: str = "date",
    series_top: Sequence[str] = ("w0_mean", "c_mean"),
    series_mid: Sequence[str] = ("w1_mean", "w2_mean"),
    series_bot: Sequence[str] = ("w0_mean", "w1_mean"),
    labels: Optional[Dict[str, str]] = None,   # e.g. {"w0_mean":"w0 (mean)", ...}
    colors: Optional[Dict[str, str]] = None,   # override if you like
    events: Optional[Dict[str, Union[str, pd.Timestamp]]] = None,
    event_style: Optional[Dict] = None,        # vline styling
    dpi: int = 300,
    figsize: tuple = (12, 7.2),
    outpath: Optional[Union[str, Path]] = None,
    outpaths_vector: Optional[Dict[str, Union[str, Path]]] = None,  # {"pdf":"...","svg":"..."}
    y_label: str = "AINI (means)",
    title: str = "AINI means by variant",
    grid: bool = True,
    linewidth: float = 1.8,
    alpha: float = 0.95,
    rc_update: Optional[Dict] = None,
):
    """
    Three stacked subplots with identical x/y scales:
      - Top   : w0_mean & c_mean
      - Middle: w1_mean & w2_mean
      - Bottom: w0_mean & w1_mean

    Event labels are horizontal, bottom-aligned near the x-axis under each vline.
    """

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
        # ---- Prep ----
        if date_col not in df.columns:
            raise ValueError(f"date_col '{date_col}' not found in DataFrame.")

        needed_cols = list(series_top) + list(series_mid) + list(series_bot)
        missing = [c for c in needed_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        data = df.copy()
        data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
        data = data.sort_values(date_col)

        # Labels
        if labels is None:
            labels = {s: s for s in needed_cols}

        # Color-blind friendly Okabe–Ito palette
        default_palette = {
            "w0_mean": "#0072B2",  # blue
            "c_mean" : "#D55E00",  # vermillion
            "w1_mean": "#E69F00",  # orange
            "w2_mean": "#009E73",  # green
        }
        if colors is None:
            colors = default_palette.copy()
        else:
            # fill any missing with sensible defaults
            for k, v in default_palette.items():
                colors.setdefault(k, v)

        # --- Event markers ---
    if events is None:
        events = {
            "Sam Altman fired": "2023-11-17",
            "EU AI Act: Start of rollout": "2024-08-01",
            "DeepSeek emerges": "2025-01-20",
        }
        events = {k: pd.to_datetime(v) for k, v in events.items()}
        evt_style = dict(color="red", linestyle="--", alpha=0.7, linewidth=1.2)
        if event_style:
            evt_style.update(event_style)

        # ---- Common ranges ----
        x_min = data[date_col].min()
        x_max = data[date_col].max()

        y_vals = []
        for col in needed_cols:
            y_vals.extend(pd.to_numeric(data[col], errors="coerce").dropna().tolist())

        if not y_vals:
            raise ValueError("All selected series are empty.")

        y_arr = np.asarray(y_vals, dtype=float)
        y_min = float(np.nanmin(y_arr))
        y_max = float(np.nanmax(y_arr))
        if not (np.isfinite(y_min) and np.isfinite(y_max)):
            raise ValueError("Y values contain no finite numbers after coercion.")
        if y_min == y_max:
            pad = 0.5 if y_min == 0 else abs(y_min) * 0.05
            y_min, y_max = y_min - pad, y_max + pad

        # ---- Figure ----
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        locator = AutoDateLocator(minticks=4, maxticks=8)
        formatter = ConciseDateFormatter(locator)

        def _plot_panel(ax, series_names):
            plotted = False
            for name in series_names:
                s = pd.to_numeric(data[name], errors="coerce")
                if s.notna().any():
                    ax.plot(
                        data[date_col], s,
                        label=labels.get(name, name),
                        color=colors.get(name, "#333333"),
                        linewidth=linewidth,
                        alpha=alpha,
                    )
                    plotted = True

            ax.set_ylabel(y_label, labelpad=6)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

            # Events: vlines + bottom-aligned horizontal labels
            y_bottom = y_min + 0.04 * (y_max - y_min)  # close to x-axis, same height for alignment
            for lab, d in events.items():
                ax.axvline(d, **evt_style)
                ax.annotate(
                    lab,
                    xy=(d, y_bottom),
                    xytext=(0, 2),  # slight upward offset in points
                    textcoords="offset points",
                    rotation=0,
                    ha="center", va="bottom",
                    fontsize=9,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1),
                )

            if plotted:
                ax.legend(
                    ncol=min(3, len(series_names)),
                    loc="lower left",
                    fontsize=9,
                    frameon=False,
                    bbox_to_anchor=(0.02, 0.02),  # shift inside the axes
                    borderaxespad=0.0,
                )

            else:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")

        # Panels
        _plot_panel(axes[0], series_top)   # w0_mean & c_mean
        _plot_panel(axes[1], series_mid)   # w1_mean & w2_mean
        _plot_panel(axes[2], series_bot)   # w0_mean & w1_mean

        axes[-1].set_xlabel("Date")
        fig.suptitle(title, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        # ---- Save ----
        def _ensure_parent(p: Path):
            p.parent.mkdir(parents=True, exist_ok=True)

        if outpath is not None:
            outpath = Path(outpath)
            _ensure_parent(outpath)
            fig.savefig(outpath, bbox_inches="tight")

        if outpaths_vector:
            for _, p in outpaths_vector.items():
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

# Function to plot timeline of extrema
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional, List, Dict, Tuple
from collections import OrderedDict

def plot_timeline(
    custom_events: Optional[List[Dict]] = None,
    outpath: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 3.8),
    dpi: int = 300,
    date_rotation: int = 60,
    palette: Optional[List[str]] = None,
    # NEW: fractions for non-zero / total articles, in the SAME ORDER as events below
    fractions_min: Optional[List[str]] = None,
    fractions_max: Optional[List[str]] = None,
):
    """
    AINI extrema timeline with:
      - distinct color per 'measures' (legend outside right),
      - '^' for maxima above baseline; 'v' for minima below,
      - vertical offset encodes strength (n = 1/2/3),
      - per-point annotation showing 'nonzero/total' next to each marker.

    Pass `fractions_min` and `fractions_max` as lists of strings like ["24/44", "2/7", ...]
    aligned to the order of minima and maxima events respectively.
    """

    # 1) Events (order matters for fraction mapping!)
    if custom_events is None:
        events = [
            # Minima (6)
            {"date": pd.Timestamp("2025-02-06"), "kind": "min", "n": 3,
             "measures": "normalized_AINI_custom, EMA_02_custom, EMA_08_custom"},
            {"date": pd.Timestamp("2023-08-13"), "kind": "min", "n": 2,
             "measures": "normalized_AINI_w0, EMA_08_w0"},
            {"date": pd.Timestamp("2025-01-28"), "kind": "min", "n": 2,
             "measures": "normalized_AINI_w1, EMA_08_w1"},
            {"date": pd.Timestamp("2024-08-02"), "kind": "min", "n": 2,
             "measures": "normalized_AINI_w2, EMA_08_w2"},
            {"date": pd.Timestamp("2025-01-31"), "kind": "min", "n": 2,
             "measures": "EMA_02_w1, EMA_02_w2 (mixed fast EMAs)"},
            {"date": pd.Timestamp("2025-01-20"), "kind": "min", "n": 1,
             "measures": "EMA_02_w0"},
            # Maxima (7)
            {"date": pd.Timestamp("2025-06-07"), "kind": "max", "n": 3,
             "measures": "normalized_AINI_w0, EMA_02_w0, EMA_08_w0"},
            {"date": pd.Timestamp("2025-06-16"), "kind": "max", "n": 3,
             "measures": "normalized_AINI_w2, EMA_02_w2, EMA_08_w2"},
            {"date": pd.Timestamp("2025-06-10"), "kind": "max", "n": 2,
             "measures": "normalized_AINI_w1, EMA_08_w1"},
            {"date": pd.Timestamp("2024-10-10"), "kind": "max", "n": 1,
             "measures": "normalized_AINI_custom"},
            {"date": pd.Timestamp("2023-04-09"), "kind": "max", "n": 1,
             "measures": "EMA_02_custom"},
            {"date": pd.Timestamp("2023-09-07"), "kind": "max", "n": 1,
             "measures": "EMA_08_custom"},
            {"date": pd.Timestamp("2025-03-22"), "kind": "max", "n": 1,
             "measures": "EMA_02_w1"},
        ]
    else:
        events = list(custom_events)

    df = pd.DataFrame(events)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("date").reset_index(drop=True)

    # Split indices for mapping fractions reliably
    idx_min = [i for i, r in df.iterrows() if r["kind"] == "min"]
    idx_max = [i for i, r in df.iterrows() if r["kind"] == "max"]

    # If user passes fractions, validate lengths
    if fractions_min is not None and len(fractions_min) != len(idx_min):
        raise ValueError(f"fractions_min length {len(fractions_min)} != number of minima {len(idx_min)}")
    if fractions_max is not None and len(fractions_max) != len(idx_max):
        raise ValueError(f"fractions_max length {len(fractions_max)} != number of maxima {len(idx_max)}")

    # 2) Figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.axhline(0, linewidth=1, color="black")

    # Vertical positions by kind & n
    y_pos = {
        ("max", 3): 0.6, ("max", 2): 0.4, ("max", 1): 0.25,
        ("min", 3): -0.6, ("min", 2): -0.4, ("min", 1): -0.25,
    }
    marker_shape = {"max": "^", "min": "v"}

    # Color-blind friendly palette (cycled per unique 'measures')
    if palette is None:
        palette = [
            "#0072B2", "#D55E00", "#009E73", "#E69F00",
            "#56B4E9", "#CC79A7", "#F0E442", "#000000",
            "#7F3C8D", "#11A579", "#FF00E118", "#F2B701"
        ]
    measures_ordered = list(OrderedDict((str(m), None) for m in df["measures"]))
    color_map = {m: palette[i % len(palette)] for i, m in enumerate(measures_ordered)}

    # 3) Plot points, legend handles, and fraction annotations
    handles, labels = [], []
    min_counter, max_counter = 0, 0

    for i, r in df.iterrows():
        kind = r["kind"]
        y = y_pos[(kind, int(r["n"]))]
        m_text = str(r["measures"])
        color = color_map[m_text]

        h = ax.plot(
            r["date"], y,
            marker=marker_shape[kind],
            markersize=8,
            linestyle="None",
            color=color,
            label=m_text,
        )
        handles.append(h[0])
        labels.append(m_text)

        # ---- Fraction annotation per point ----
        # choose fraction based on kind and position in that subset
        if kind == "min":
            frac = (fractions_min[min_counter] if fractions_min is not None else None)
            min_counter += 1
            dy = -6   # below
            va = "top"
        else:
            frac = (fractions_max[max_counter] if fractions_max is not None else None)
            max_counter += 1
            dy = 6    # above
            va = "bottom"

        if frac:
            # small, unobtrusive label near the marker
            ax.annotate(
                frac.replace(" ", ""),  # e.g., "24/44"
                xy=(r["date"], y),
                xytext=(4, dy),  # slight right + up/down
                textcoords="offset points",
                ha="left", va=va,
                fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1),
                color=color,
            )

    # 4) Reduce date ticks + rotate
    span_days = (df["date"].max() - df["date"].min()).days
    if span_days <= 365:
        locator = mdates.MonthLocator(interval=3)   # quarterly
    elif span_days <= 3 * 365:
        locator = mdates.MonthLocator(interval=6)   # semiannual
    else:
        locator = mdates.YearLocator(base=1)        # yearly
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=date_rotation, ha="right")

    # 5) Cosmetics + de-dup legend
    ax.set_ylim(-0.8, 0.8)
    ax.set_yticks([])
    ax.set_title("AINI Extrema Timeline")
    ax.set_xlabel("Date")

    seen = set()
    uniq_handles, uniq_labels = [], []
    for h, lab in zip(handles, labels):
        if lab not in seen:
            seen.add(lab)
            uniq_handles.append(h)
            uniq_labels.append(lab)

    ax.legend(
        uniq_handles, uniq_labels,
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=8, frameon=False, borderaxespad=0.0,
        handlelength=1.2, labelspacing=0.6,
    )

    fig.tight_layout(rect=[0, 0, 0.70, 1])
    if outpath:
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    return fig, ax

# Function to plot timeline vs events 

# ------------------------------------------------------------
# Color palette (Okabe–Ito; color-blind friendly)
# ------------------------------------------------------------
OI_BLUE = "#0072B2"
OI_GREY = "#333333"


# ------------------------------------------------------------
# Measure tokens and event name helpers
# ------------------------------------------------------------
_MEASURE_TOKEN = {
    "normalized_AINI": "norm",
    "EMA_02": "02",
    "EMA_08": "08",
}

def _abbr_window(win: str) -> str:
    """
    Convert window names to compact tokens:
      - 'custom' -> 'c'
      - 'w1+w2'  -> 'w1w2'
      - 'w0'/'w1'/'w2' unchanged
    """
    win = str(win).strip()
    if win.lower() == "custom":
        return "c"
    return win.replace("+", "")

def _tokens_from_measures(measures: str) -> list[str]:
    """
    Parse a comma-separated 'measures' string and return compact tokens in order.
    Example:
        'normalized_AINI_custom, EMA_02_custom, EMA_08_custom'
        -> ['norm', '02', '08']
    """
    toks: list[str] = []
    for item in str(measures).split(","):
        item = item.strip()
        m = re.match(r"(normalized_AINI|EMA_02|EMA_08)", item)
        if m:
            toks.append(_MEASURE_TOKEN[m.group(1)])
    return toks or ["unk"]

def make_event_name(kind: str, window: str, measures: str) -> str:
    """
    Build indicative name like: 'min_c_norm_02_08', 'max_w2_norm_02_08', 'min_w1_08', ...
    The numeric counter is *not* included; (n=...) communicates cardinality.
    """
    kind = str(kind).strip().lower()  # 'min' or 'max'
    w = _abbr_window(window)
    parts = _tokens_from_measures(measures)
    comp = "_".join(parts)
    return f"{kind}_{w}_{comp}"

def rename_events(events: list[Dict]) -> list[Dict]:
    """
    Return a deep-copied list with 'name' replaced by the generated indicative name.
    """
    out: list[Dict] = []
    for e in events:
        e2 = dict(e)  # shallow copy is fine (only simple types used)
        e2["name"] = make_event_name(e["kind"], e["window"], e["measures"])
        out.append(e2)
    return out


# ------------------------------------------------------------
# Default extrema events (edit here if you need to update)
# ------------------------------------------------------------
def default_extrema_events() -> list[Dict]:
    """
    curated extrema list. Names will be overwritten by `rename_events(...)`.
    """
    events = [
        # ---- Minima ----
        {"name": "", "date": pd.Timestamp("2025-02-06"), "kind": "min", "n": 3,
         "window": "custom", "measures": "normalized_AINI_custom, EMA_02_custom, EMA_08_custom"},
        {"name": "", "date": pd.Timestamp("2023-08-13"), "kind": "min", "n": 2,
         "window": "w0", "measures": "normalized_AINI_w0, EMA_08_w0"},
        {"name": "", "date": pd.Timestamp("2025-01-28"), "kind": "min", "n": 2,
         "window": "w1", "measures": "normalized_AINI_w1, EMA_08_w1"},
        {"name": "", "date": pd.Timestamp("2024-08-02"), "kind": "min", "n": 2,
         "window": "w2", "measures": "normalized_AINI_w2, EMA_08_w2"},
        {"name": "", "date": pd.Timestamp("2025-01-31"), "kind": "min", "n": 2,
         "window": "w1+w2", "measures": "EMA_02_w1, EMA_02_w2 (mixed)"},
        {"name": "", "date": pd.Timestamp("2025-01-20"), "kind": "min", "n": 1,
         "window": "w0", "measures": "EMA_02_w0"},

        # ---- Maxima ----
        {"name": "", "date": pd.Timestamp("2025-06-07"), "kind": "max", "n": 3,
         "window": "w0", "measures": "normalized_AINI_w0, EMA_02_w0, EMA_08_w0"},
        {"name": "", "date": pd.Timestamp("2025-06-16"), "kind": "max", "n": 3,
         "window": "w2", "measures": "normalized_AINI_w2, EMA_02_w2, EMA_08_w2"},
        {"name": "", "date": pd.Timestamp("2025-06-10"), "kind": "max", "n": 2,
         "window": "w1", "measures": "normalized_AINI_w1, EMA_08_w1"},

        # ---- Maxima (custom singletons) ----
        {"name": "", "date": pd.Timestamp("2024-10-10"), "kind": "max", "n": 1,
         "window": "custom", "measures": "normalized_AINI_custom"},
        {"name": "", "date": pd.Timestamp("2023-04-09"), "kind": "max", "n": 1,
         "window": "custom", "measures": "EMA_02_custom"},
        {"name": "", "date": pd.Timestamp("2023-09-07"), "kind": "max", "n": 1,
         "window": "custom", "measures": "EMA_08_custom"},
        {"name": "", "date": pd.Timestamp("2025-03-22"), "kind": "max", "n": 1,
         "window": "w1", "measures": "EMA_02_w1"},
    ]
    return rename_events(events)


# ------------------------------------------------------------
# Plot: n_articles line + extrema markers (same semantics as your timeline)
# ------------------------------------------------------------
def plot_n_articles_with_extrema_events(
    dfp: pd.DataFrame,
    date_col: Optional[str] = "date",   # set None to use DatetimeIndex
    count_col: str = "n_articles",
    custom_events: Optional[list[Dict]] = None,
    annotate: bool = True,
    figsize: Tuple[float, float] = (12, 4.0),
    dpi: int = 300,
    outpath: Optional[str | Path] = None,
):
    """
    Line plot of daily article counts with AINI extrema markers overlaid
    (re-using the same marker logic as your `plot_timeline`).

    Left y-axis  : counts (n_articles)
    Right y-axis : a fixed band in [-0.8, 0.8] where extrema markers are placed.

    - '^' for maxima and 'v' for minima.
    - Vertical offsets encode strength (n = 1/2/3).
    - If `custom_events` is None, uses `default_extrema_events()`.

    Parameters
    ----------
    dfp : pd.DataFrame
        Must contain the counts and some date field/index.
    date_col : str | None
        Name of the date column. If None, a DatetimeIndex is expected.
    count_col : str
        Name of the counts column (default: 'n_articles').
    custom_events : list[dict] | None
        Optional events; same schema as default_extrema_events().
    annotate : bool
        Whether to annotate marker labels.
    figsize : (float, float)
        Figure size in inches.
    dpi : int
        Resolution for display/saving.
    outpath : str | Path | None
        If provided, saves the figure.

    Returns
    -------
    (fig, ax_counts)
    """
    dfp = dfp.copy()

    # ---- Resolve date series ----
    if date_col and date_col in dfp.columns:
        date_ser = pd.to_datetime(dfp[date_col], errors="coerce")
    elif isinstance(dfp.index, pd.DatetimeIndex):
        date_ser = pd.to_datetime(dfp.index, errors="coerce")
    else:
        # fallback common names
        date_ser = None
        for cand in ("Date", "ds", "timestamp"):
            if cand in dfp.columns:
                date_ser = pd.to_datetime(dfp[cand], errors="coerce")
                date_col = cand
                break
        if date_ser is None:
            raise ValueError(
                "No usable date column found. Provide `date_col` or set a DatetimeIndex.\n"
                f"Available columns: {list(dfp.columns)}"
            )

    # ---- Validate counts ----
    if count_col not in dfp.columns:
        raise ValueError(
            f"DataFrame must contain '{count_col}'. "
            f"Available columns: {list(dfp.columns)}"
        )

    # ---- Clean & sort ----
    dfp["_date"] = date_ser
    dfp[count_col] = pd.to_numeric(dfp[count_col], errors="coerce")
    dfp = dfp.dropna(subset=["_date", count_col]).sort_values("_date")
    if dfp.empty:
        raise ValueError("No valid rows after cleaning dates and counts.")

    # ---- Events ----
    events = custom_events if custom_events is not None else default_extrema_events()
    ev_df = pd.DataFrame(events).copy()
    ev_df["date"] = pd.to_datetime(ev_df["date"], errors="raise")
    ev_df = ev_df.sort_values("date").reset_index(drop=True)

    # Marker semantics (same as your timeline)
    y_pos = {
        ("max", 3): 0.6, ("max", 2): 0.4, ("max", 1): 0.25,
        ("min", 3): -0.6, ("min", 2): -0.4, ("min", 1): -0.25,
    }
    marker = {"max": "^", "min": "v"}

    # ---- Plot ----
    fig, ax_counts = plt.subplots(figsize=figsize, dpi=dpi)

    # Counts line
    ax_counts.plot(
        dfp["_date"], dfp[count_col],
        color=OI_BLUE, linewidth=1.6, alpha=0.95, label=count_col
    )
    ax_counts.set_ylabel("Articles per day")
    ax_counts.set_xlabel("Date")
    ax_counts.grid(True, linestyle="--", alpha=0.4)

    # Adaptive monthly ticks
    span_days = (dfp["_date"].max() - dfp["_date"].min()).days
    interval = 1 if span_days <= 120 else 2 if span_days <= 365 else 3
    ax_counts.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    ax_counts.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax_counts.get_xticklabels(), rotation=0, ha="center")

    # Secondary axis for markers
    ax_evt = ax_counts.twinx()
    ax_evt.set_ylim(-0.8, 0.8)
    ax_evt.set_yticks([])
    ax_evt.grid(False)

    # Plot markers + annotations
    for _, r in ev_df.iterrows():
        ypos = y_pos[(str(r["kind"]).lower(), int(r["n"]))]
        ax_evt.plot(
            r["date"], ypos,
            marker=marker[str(r["kind"]).lower()],
            markersize=8,
            linestyle="None",
            color=OI_GREY,
            zorder=3,
        )
        if annotate:
            # Use the generated indicative name (already set by default_extrema_events())
            label = f"{r['name']} (n={r['n']})"
            ax_evt.annotate(
                label,
                xy=(r["date"], ypos),
                xytext=(0, 10 if str(r["kind"]).lower() == "max" else -14),
                textcoords="offset points",
                ha="center",
                va="bottom" if str(r["kind"]).lower() == "max" else "top",
                fontsize=8,
                color=OI_GREY,
            )

    fig.tight_layout()
    if outpath:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")

    return fig, ax_counts


# ------------------------------------------------------------
# Public exports
# ------------------------------------------------------------
__all__ = [
    "plot_n_articles_with_extrema_events",
    "default_extrema_events",
    "rename_events",
    "make_event_name",
]

def plot_stock_growth(
    df: pd.DataFrame,
    tickers,
    start="2023-04-01",
    end="2025-06-15",
    group_size=5,
    base=100.0,
    save_dir=None,           # e.g. Path("reports/figures")
    show=True
):
    """
    Plot growth (Adj Close normalized to `base` at each ticker's first in-range date).
    Creates N = ceil(len(tickers)/group_size) figures with equal x/y scales.

    Parameters
    ----------
    df : DataFrame with columns ['Ticker','Adj Close','date'] (or 'Date')
    tickers : iterable of ticker strings
    start, end : date bounds (inclusive)
    group_size : tickers per figure
    base : starting index level per ticker (e.g. 100.0)
    save_dir : if set, saves PNGs as 'growth_group_{i}.png'
    show : whether to display the plots
    """
    d = df.copy()

    # Ensure datetime column named 'date'
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"])
    elif "Date" in d.columns:
        d["date"] = pd.to_datetime(d["Date"])
    else:
        raise KeyError("DataFrame must have a 'date' or 'Date' column.")

    # Filter by date + tickers; keep only necessary cols
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    d = d[(d["date"] >= start) & (d["date"] <= end) & (d["Ticker"].isin(tickers))]
    d = d[["date", "Ticker", "Adj Close"]].dropna(subset=["Adj Close"])

    if d.empty:
        raise ValueError("No data after filtering. Check dates/tickers/columns.")

    # Build per-ticker growth index normalized to first available point in range
    d = d.sort_values(["Ticker", "date"])
    first_vals = d.groupby("Ticker")["Adj Close"].transform("first")
    d["Growth"] = base * (d["Adj Close"] / first_vals)

    # Compute global y-limits for identical scales
    ymin = d["Growth"].min()
    ymax = d["Growth"].max()
    pad = 0.03 * (ymax - ymin) if ymax > ymin else 1.0
    ylims = (ymin - pad, ymax + pad)

    # Consistent groups of tickers
    tickers_sorted = sorted(set(tickers))
    groups = [tickers_sorted[i:i+group_size] for i in range(0, len(tickers_sorted), group_size)]

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    figs = []
    for gi, grp in enumerate(groups, start=1):
        fig, ax = plt.subplots(figsize=(10, 5))
        for t in grp:
            sub = d[d["Ticker"] == t]
            if sub.empty:
                continue
            ax.plot(sub["date"], sub["Growth"], linewidth=1.8, label=t)

        ax.set_title(f"Growth (Adj Close, base={base}) — Group {gi}: {', '.join(grp)}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Index (base = {:.0f})".format(base))
        ax.set_xlim(start, end)
        ax.set_ylim(*ylims)
        ax.legend(ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.02))
        fig.autofmt_xdate()
        plt.tight_layout()

        if save_dir:
            out = save_dir / f"growth_group_{gi}.png"
            fig.savefig(out, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            figs.append(fig)

    return figs if not show else None
