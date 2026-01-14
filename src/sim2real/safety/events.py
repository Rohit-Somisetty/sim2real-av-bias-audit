"""Safety event detectors mirrored from Project 1 requirements."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd

EVENT_TYPES = [
    "disengagement",
    "hard_brake",
    "lane_deviation",
    "near_miss",
]

MILES_PER_METER = 0.000621371


def summarize_events(df: pd.DataFrame, *, group_cols: Sequence[str] = ("domain",)) -> pd.DataFrame:
    if "domain" not in group_cols:
        raise ValueError("group_cols must include 'domain'")
    group_cols = list(group_cols)

    rows = []
    for keys, group in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_cols, keys))
        group = group.sort_values(["trip_id", "timestamp_s"])
        counts = _count_events(group)
        miles = _miles(group)
        for event_type, count in counts.items():
            row = {**key_map, "event_type": event_type, "count": count, "miles": miles}
            rows.append(row)
    return pd.DataFrame(rows)


def _count_events(group: pd.DataFrame) -> Dict[str, int]:
    counts = {event: 0 for event in EVENT_TYPES}
    for _, trip_df in group.groupby("trip_id"):
        trip_df = trip_df.sort_values("timestamp_s")
        engaged = trip_df["engaged"].astype(bool).to_numpy()
        dt = trip_df["dt_s"].astype(float).to_numpy()
        accel = trip_df["ego_accel_mps2"].astype(float).to_numpy()
        lane_offset = np.abs(trip_df["lane_offset_m"].astype(float).to_numpy())
        ttc = trip_df["ttc_s"].astype(float).to_numpy()

        counts["disengagement"] += _count_disengagements(engaged)
        counts["hard_brake"] += _count_threshold_sequences(accel < -3.0, dt, min_duration=0.3)
        counts["lane_deviation"] += _count_threshold_sequences(lane_offset > 0.6, dt, min_duration=1.0)
        counts["near_miss"] += _count_threshold_sequences(ttc < 1.5, dt, min_duration=0.2)
    return counts


def _count_disengagements(engaged: np.ndarray) -> int:
    transitions = np.logical_and(engaged[:-1], ~engaged[1:])
    return int(transitions.sum())


def _count_threshold_sequences(mask: np.ndarray, dt: np.ndarray, *, min_duration: float) -> int:
    duration = 0.0
    count = 0
    for active, delta in zip(mask, dt):
        if active:
            duration += float(delta)
        else:
            if duration >= min_duration:
                count += 1
            duration = 0.0
    if duration >= min_duration:
        count += 1
    return count


def _miles(group: pd.DataFrame) -> float:
    meters = (group["ego_speed_mps"] * group["dt_s"]).sum()
    return float(meters) * MILES_PER_METER
