"""Configuration models and shared constants for the sim2real toolkit."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from pydantic import BaseModel, Field


CONTINUOUS_FEATURES: Sequence[str] = (
    "ego_speed_mps",
    "ego_accel_mps2",
    "lane_offset_m",
    "ttc_s",
)

CATEGORICAL_FEATURES: Sequence[str] = (
    "weather",
    "time_of_day",
    "traffic_density",
)

REQUIRED_COLUMNS: Sequence[str] = (
    "trip_id",
    "timestamp_s",
    "dt_s",
    *CONTINUOUS_FEATURES,
    "engaged",
    *CATEGORICAL_FEATURES,
    "domain",
)

OPTIONAL_COLUMNS: Sequence[str] = (
    "date",
    "driver_version",
)

DOMAINS: Sequence[str] = ("sim", "real")


class GeneratorConfig(BaseModel):
    """Parameters that control the synthetic paired log generator."""

    trips_per_domain: int = Field(200, ge=1)
    hz: float = Field(10.0, gt=0.0)
    min_duration_s: float = Field(30.0, gt=0.0)
    max_duration_s: float = Field(120.0, gt=0.0)
    shift_strength: float = Field(1.0, ge=0.0)
    seed: int = 42

    def samples_per_trip(self) -> int:
        avg_duration = 0.5 * (self.min_duration_s + self.max_duration_s)
        return int(avg_duration * self.hz)


class AnalysisConfig(BaseModel):
    """Global analysis knobs for the CLI pipeline."""

    continuous_features: List[str] = Field(default_factory=lambda: list(CONTINUOUS_FEATURES))
    categorical_features: List[str] = Field(default_factory=lambda: list(CATEGORICAL_FEATURES))
    bins: int = 40
    min_slice_miles: float = 10.0
    plots_dir: Path | None = None


DEFAULT_ANALYSIS_CONFIG = AnalysisConfig()
