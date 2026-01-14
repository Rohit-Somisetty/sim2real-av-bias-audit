"""Safety event rate comparison utilities."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def compute_event_rates(event_summary: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in event_summary.iterrows():
        count = int(row["count"])
        miles = float(row["miles"]) or 1e-6
        rate = 1000.0 * count / miles
        ci_low, ci_high = _poisson_rate_ci(count, miles)
        post = _gamma_posterior(count, miles)
        records.append(
            {
                **{k: row[k] for k in row.index if k not in {"count", "miles"}},
                "count": count,
                "miles": miles,
                "rate_per_1000_miles": rate,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "bayes_mean": post["mean"],
                "bayes_ci_lower": post["ci_lower"],
                "bayes_ci_upper": post["ci_upper"],
                "posterior_shape": post["shape"],
                "posterior_rate": post["rate"],
            }
        )
    return pd.DataFrame(records)


def compare_event_rates(rate_df: pd.DataFrame) -> pd.DataFrame:
    comparisons = []
    for event_type, group in rate_df.groupby("event_type"):
        pivot = group.set_index("domain")
        if not {"sim", "real"}.issubset(pivot.index):
            continue
        real = pivot.loc["real"]
        sim = pivot.loc["sim"]
        delta = real["rate_per_1000_miles"] - sim["rate_per_1000_miles"]
        pct = delta / max(sim["rate_per_1000_miles"], 1e-6) * 100
        prob = _posterior_prob_higher(real, sim)
        comparisons.append(
            {
                "event_type": event_type,
                "rate_sim": sim["rate_per_1000_miles"],
                "rate_real": real["rate_per_1000_miles"],
                "delta_per_1000_miles": delta,
                "percent_diff": pct,
                "p_real_gt_sim": prob,
            }
        )
    return pd.DataFrame(comparisons)


def _poisson_rate_ci(count: int, miles: float, alpha: float = 0.05) -> Tuple[float, float]:
    lower = 0.5 * stats.chi2.ppf(alpha / 2, 2 * count) if count > 0 else 0.0
    upper = 0.5 * stats.chi2.ppf(1 - alpha / 2, 2 * (count + 1))
    lower_rate = 1000.0 * lower / miles
    upper_rate = 1000.0 * upper / miles
    return float(lower_rate), float(upper_rate)


def _gamma_posterior(count: int, miles: float, alpha0: float = 1.0, beta0: float = 1.0) -> Dict[str, float]:
    shape = alpha0 + count
    rate = beta0 + miles
    dist = stats.gamma(a=shape, scale=1 / rate)
    mean = 1000.0 * dist.mean()
    ci_lower = 1000.0 * dist.ppf(0.05)
    ci_upper = 1000.0 * dist.ppf(0.95)
    return {
        "shape": float(shape),
        "rate": float(rate),
        "mean": float(mean),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }


def _posterior_prob_higher(real_row: pd.Series, sim_row: pd.Series, n_samples: int = 10000) -> float:
    samples_real = stats.gamma(
        a=real_row["posterior_shape"], scale=1 / real_row["posterior_rate"]
    ).rvs(size=n_samples)
    samples_sim = stats.gamma(
        a=sim_row["posterior_shape"], scale=1 / sim_row["posterior_rate"]
    ).rvs(size=n_samples)
    return float((samples_real > samples_sim).mean())
