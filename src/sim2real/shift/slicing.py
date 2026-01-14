"""Slice-level divergence summaries."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from sim2real.shift.divergence import compute_shift_metrics
from sim2real.safety.events import summarize_events
from sim2real.safety.rate_compare import compute_event_rates


MILES_PER_METER = 0.000621371


def compute_slice_gaps(
    df: pd.DataFrame,
    *,
    slice_cols: Sequence[str],
    continuous_features: Sequence[str],
    categorical_features: Sequence[str],
    min_slice_miles: float = 10.0,
) -> pd.DataFrame:
    records = []
    for slice_col in slice_cols:
        for value, slice_df in df.groupby(slice_col):
            exposure = _miles_by_domain(slice_df)
            if min(exposure.values()) < min_slice_miles:
                continue
            shift_metrics = compute_shift_metrics(
                slice_df,
                continuous_features=continuous_features,
                categorical_features=categorical_features,
            )
            top_shift = (
                shift_metrics[shift_metrics["feature"] != "__overall__"]
                .sort_values("shift_score", ascending=False)
                .head(1)
            )
            top_feature = top_shift["feature"].iat[0] if not top_shift.empty else None
            top_score = float(top_shift["shift_score"].iat[0]) if not top_shift.empty else 0.0

            events = summarize_events(slice_df, group_cols=["domain", slice_col])
            rates = compute_event_rates(events)
            gap = _worst_event_gap(rates)
            records.append(
                {
                    "slice_dim": slice_col,
                    "slice_value": value,
                    "miles_sim": exposure.get("sim", 0.0),
                    "miles_real": exposure.get("real", 0.0),
                    "top_shift_feature": top_feature,
                    "top_shift_score": top_score,
                    "max_event_gap_per_1000_miles": gap,
                }
            )
    return pd.DataFrame(records)


def _miles_by_domain(df: pd.DataFrame) -> dict[str, float]:
    meters = (
        df.groupby("domain")
        .apply(lambda x: float((x["ego_speed_mps"] * x["dt_s"]).sum()))
        .to_dict()
    )
    return {domain: meters.get(domain, 0.0) * MILES_PER_METER for domain in ("sim", "real")}


def _worst_event_gap(rate_df: pd.DataFrame) -> float:
    if rate_df.empty:
        return 0.0
    agg = rate_df.pivot_table(
        index="event_type",
        columns="domain",
        values="rate_per_1000_miles",
        aggfunc="first",
    ).fillna(0.0)
    if {"sim", "real"}.issubset(agg.columns):
        diffs = agg["real"] - agg["sim"]
        return float(diffs.max())
    return 0.0
