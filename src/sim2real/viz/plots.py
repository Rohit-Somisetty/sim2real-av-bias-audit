"""Visualization helpers for the analysis pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("seaborn-v0_8-darkgrid")


def plot_continuous_distributions(df: pd.DataFrame, features: Iterable[str], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for feature in features:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for domain, color in {"sim": "tab:blue", "real": "tab:orange"}.items():
            subset = df[df["domain"] == domain][feature].dropna()
            axes[0].hist(subset, bins=40, alpha=0.5, label=domain.upper(), color=color)
            axes[1].plot(
                sorted(subset),
                pd.Series(sorted(subset)).rank(pct=True),
                label=domain.upper(),
                color=color,
            )
        axes[0].set_title(f"{feature} histogram")
        axes[1].set_title(f"{feature} CDF")
        for ax in axes:
            ax.set_xlabel(feature)
            ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"{feature}_dist.png", dpi=150)
        plt.close(fig)


def plot_categorical_distributions(df: pd.DataFrame, features: Iterable[str], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for feature in features:
        counts = df.groupby(["domain", feature]).size().unstack(0).fillna(0)
        counts = counts.div(counts.sum(axis=1), axis=0)
        ax = counts.plot(kind="bar", figsize=(8, 4))
        ax.set_ylabel("Share")
        ax.set_title(f"{feature} distribution")
        ax.legend(title="Domain")
        plt.tight_layout()
        plt.savefig(outdir / f"{feature}_categorical.png", dpi=150)
        plt.close()


def plot_shift_leaders(
    shift_df: pd.DataFrame,
    outdir: Path,
    top_k: int = 10,
    filename: str = "shift_leaders.png",
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    leaders = (
        shift_df[shift_df["feature"] != "__overall__"]
        .sort_values("shift_score", ascending=False)
        .head(top_k)
        .set_index("feature")
    )
    ax = leaders["shift_score"].plot(kind="bar", figsize=(8, 4), color="tab:red")
    ax.set_ylabel("Shift Score")
    ax.set_title("Top shifted features")
    plt.tight_layout()
    plt.savefig(outdir / filename, dpi=150)
    plt.close()


def plot_event_rate_comparison(compare_df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if compare_df.empty:
        return
    melted = compare_df.melt(
        id_vars="event_type",
        value_vars=["rate_sim", "rate_real"],
        var_name="metric",
        value_name="rate_per_1000_miles",
    )
    ax = melted.pivot_table(index="event_type", columns="metric", values="rate_per_1000_miles").plot(
        kind="bar", figsize=(8, 4)
    )
    ax.set_ylabel("Rate per 1k miles")
    ax.set_title("Event rate comparison")
    plt.tight_layout()
    plt.savefig(outdir / "event_rates.png", dpi=150)
    plt.close()
