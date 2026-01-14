"""Synthetic paired SIM/REAL log generator with controllable shifts."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from sim2real.config import GeneratorConfig, DOMAINS, REQUIRED_COLUMNS


def generate_paired_logs(config: GeneratorConfig, *, out: Path | None = None) -> pd.DataFrame:
    """Generate synthetic SIM and REAL logs with a tunable distribution shift.

    Parameters
    ----------
    config:
        Generator configuration.
    out:
        Optional parquet path to persist the dataframe.
    """

    rng = np.random.default_rng(config.seed)
    rows: List[Dict[str, object]] = []
    base_date = datetime(2025, 1, 1)

    for domain in DOMAINS:
        for trip_idx in range(config.trips_per_domain):
            duration = rng.uniform(config.min_duration_s, config.max_duration_s)
            n = max(2, int(duration * config.hz))
            dt = 1.0 / config.hz
            timestamps = np.arange(n, dtype=float) * dt
            base_speed = 13.0 + rng.normal(0, 1.5)
            speed_shift = -0.8 * config.shift_strength if domain == "real" else 0.6 * config.shift_strength
            speed_noise = 0.8 if domain == "sim" else 1.2
            speeds = np.clip(
                base_speed + speed_shift + rng.normal(0, speed_noise, size=n),
                0.0,
                None,
            )
            accel = np.gradient(speeds, dt)

            lane_drift = rng.normal(0.0, 0.2 + 0.1 * config.shift_strength, size=n)
            lane_offset = lane_drift
            if domain == "sim":
                lane_offset += rng.normal(0.0, 0.05, size=n)
            else:
                lane_offset += rng.normal(0.0, 0.15 * config.shift_strength, size=n)

            base_ttc = rng.lognormal(mean=1.3, sigma=0.4, size=n)
            if domain == "real":
                base_ttc -= np.abs(rng.normal(0, 0.6 * config.shift_strength, size=n))
            else:
                base_ttc += np.abs(rng.normal(0, 0.3, size=n))
            ttc = np.clip(base_ttc, 0.1, None)

            engaged = _simulate_engagement(rng, n, dt, domain, config.shift_strength)

            weather = rng.choice(["clear", "rain", "fog"], p=[0.6, 0.3, 0.1])
            if domain == "sim":
                weather = rng.choice(["clear", "rain", "fog"], p=[0.8, 0.15, 0.05])
            tod = rng.choice(["day", "night", "dusk"], p=[0.55, 0.25, 0.2])
            traffic = rng.choice(
                ["low", "medium", "high"],
                p=[0.2, 0.35, 0.45] if domain == "real" else [0.35, 0.45, 0.2],
            )

            trip_id = f"{domain}_trip_{trip_idx:04d}"
            trip_date = base_date + timedelta(days=int(rng.integers(0, 30)))
            driver_version = f"v{1 + trip_idx % 3}.{0 if domain == 'sim' else 1}"

            for step, ts in enumerate(timestamps):
                rows.append(
                    {
                        "trip_id": trip_id,
                        "timestamp_s": float(ts),
                        "dt_s": float(dt),
                        "ego_speed_mps": float(speeds[step]),
                        "ego_accel_mps2": float(accel[step]),
                        "lane_offset_m": float(lane_offset[step]),
                        "ttc_s": float(ttc[step]),
                        "engaged": bool(engaged[step]),
                        "weather": weather,
                        "time_of_day": tod,
                        "traffic_density": traffic,
                        "domain": domain,
                        "date": trip_date.date().isoformat(),
                        "driver_version": driver_version,
                    }
                )

    df = pd.DataFrame(rows)
    df = df.sort_values(["domain", "trip_id", "timestamp_s"]).reset_index(drop=True)

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)

    return df


def _simulate_engagement(rng: np.random.Generator, n: int, dt: float, domain: str, shift_strength: float) -> np.ndarray:
    engaged = np.ones(n, dtype=bool)
    disengage_rate = 0.001 * (1.5 if domain == "real" else 1.0 + 0.5 * shift_strength)
    dwell = 0.0
    for i in range(1, n):
        dwell += dt
        if engaged[i - 1] and rng.random() < disengage_rate * dt * dwell:
            engaged[i:] = False
            dwell = 0.0
        elif not engaged[i - 1] and rng.random() < 0.05:
            engaged[i:] = True
    return engaged
