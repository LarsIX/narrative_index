import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_aini_series_subplots(df, date_col="date", outpath=None):
    """
    Plot normalized_AINI, EMA_02, EMA_08 by date for w0, w1, w2 in subplots.
    Marks GPT-5 rumors, Sam Altman firing, and DeepSeek emergence.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    base_vars = ["normalized_AINI", "EMA_02", "EMA_08"]
    suffixes = ["w0", "w1", "w2"]

    # Color mapping for windows
    colors = {"w0": "tab:blue", "w1": "tab:orange", "w2": "tab:green"}

    # Important event dates
    events = {
        "Rumors GPT-5": "2023-04-01",
        "Sam Altman fired": "2023-11-17",
        "DeepSeek emerges": "2025-01-01",
    }
    events = {k: pd.to_datetime(v) for k, v in events.items()}

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes = axes.flatten()

    for i, base in enumerate(base_vars):
        ax = axes[i]

        # Plot series
        for suf in suffixes:
            col = f"{base}_{suf}"
            if col in df.columns:
                ax.plot(
                    df[date_col],
                    df[col],
                    label=f"{suf}",
                    color=colors[suf],
                    linewidth=1.8,
                )

        ax.set_title(base, fontsize=13)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(title="Window")

        # Mark events with vertical lines + offset labels
        ymin, ymax = ax.get_ylim()
        ytext = ymax * 0.95  # slightly below top
        for j, (label, date) in enumerate(events.items()):
            ax.axvline(date, color="red", linestyle="--", alpha=0.7)
            ax.text(
                date, 
                ytext - j * (ymax * 0.08),  # stagger labels down
                label, 
                rotation=90, 
                verticalalignment="top", 
                fontsize=9, 
                color="red",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1)
            )

    # Hide unused 4th subplot
    axes[-1].axis("off")

    fig.suptitle("AINI Time Series with Key AI Events", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=300, bbox_inches="tight")

    plt.show()
