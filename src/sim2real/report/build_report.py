"""Generate stakeholder-friendly Markdown summaries."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd


def build_markdown_report(
    out_path: Path,
    *,
    domain_auc: float,
    shift_df: pd.DataFrame,
    reweight_df: pd.DataFrame,
    rate_compare_df: pd.DataFrame,
    slice_df: pd.DataFrame,
    key_findings: List[str],
) -> None:
    lines = ["# SIM vs REAL bias audit report", ""]
    lines.append("## Key Findings")
    if key_findings:
        lines.extend([f"- {item}" for item in key_findings])
    else:
        lines.append("- No material SIM vs REAL differences detected.")
    lines.append(f"Domain classifier AUC reference: **{domain_auc:.3f}**")
    lines.append("")

    lines.append("## Top shifted features")
    lines.append(_table_from_df(_top_features(shift_df)))

    lines.append("\n## Reweighting impact (Wasserstein distance)")
    if reweight_df.empty:
        lines.append("No reweighting evaluation available.")
    else:
        lines.append(_table_from_df(reweight_df))

    lines.append("\n## Safety event comparison")
    if rate_compare_df.empty:
        lines.append("No safety events detected.")
    else:
        lines.append(_table_from_df(rate_compare_df))

    lines.append("\n## Risky slices (REAL worse than SIM)")
    if slice_df.empty:
        lines.append("No slices passed exposure threshold.")
    else:
        slices = slice_df.sort_values("max_event_gap_per_1000_miles", ascending=False).head(5)
        lines.append(_table_from_df(slices))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _table_from_df(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no data)"
    header = " | ".join(df.columns)
    divider = " | ".join(["---"] * len(df.columns))
    rows = [" | ".join(_format_cell(val) for val in row) for row in df.values.tolist()]
    return "\n".join([header, divider, *rows])


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _top_features(shift_df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    filtered = shift_df[shift_df["feature"] != "__overall__"]
    return filtered.sort_values("shift_score", ascending=False).head(k)[["feature", "shift_score"]]


def _overall_shift(shift_df: pd.DataFrame) -> float:
    summary = shift_df[shift_df["feature"] == "__overall__"]
    return float(summary["shift_score"].iat[0]) if not summary.empty else 0.0
