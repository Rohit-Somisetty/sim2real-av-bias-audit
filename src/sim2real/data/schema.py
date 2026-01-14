"""Schema validation utilities for simulator vs real frame-level logs."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from sim2real.config import CONTINUOUS_FEATURES, CATEGORICAL_FEATURES, DOMAINS, REQUIRED_COLUMNS


FRAME_DTYPE_HINTS: Mapping[str, str] = {
    "trip_id": "category",
    "timestamp_s": "float64",
    "dt_s": "float32",
    "ego_speed_mps": "float32",
    "ego_accel_mps2": "float32",
    "lane_offset_m": "float32",
    "ttc_s": "float32",
    "engaged": "bool",
    "weather": "category",
    "time_of_day": "category",
    "traffic_density": "category",
    "domain": "category",
}


class SchemaError(ValueError):
    """Raised when a frame-level dataframe violates the expected schema."""


def validate_frame_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Validate required columns, dtypes, and domain membership.

    Parameters
    ----------
    df:
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        A validated dataframe with canonical dtypes applied.
    """

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}")

    df = df.copy()
    for column, dtype in FRAME_DTYPE_HINTS.items():
        if column in df.columns:
            df[column] = _coerce_dtype(df[column], dtype)

    unexpected_domains = set(df["domain"].unique()) - set(DOMAINS)
    if unexpected_domains:
        raise SchemaError(f"Unexpected domain values: {sorted(unexpected_domains)}")

    for column in CONTINUOUS_FEATURES:
        if not np.isfinite(df[column]).all():
            raise SchemaError(f"Non-finite values found in {column}")

    return df


def _coerce_dtype(series: pd.Series, dtype: str) -> pd.Series:
    if dtype == "bool":
        if series.dtype == "bool":
            return series
        lower_map = {"true": True, "false": False, "1": True, "0": False}
        return series.fillna(False).map(lambda v: lower_map.get(str(v).lower(), bool(v)))
    if dtype.startswith("float"):
        return pd.to_numeric(series, errors="coerce").astype(dtype)
    if dtype == "category":
        return series.astype("string").astype("category")
    return series.astype(dtype)
