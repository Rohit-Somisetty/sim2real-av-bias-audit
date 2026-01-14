"""Statistical divergence metrics between SIM and REAL domains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class FeatureShift:
    feature: str
    feature_type: str
    shift_score: float
    metrics: dict


def compute_shift_metrics(
    df: pd.DataFrame,
    *,
    continuous_features: Iterable[str],
    categorical_features: Iterable[str],
    bins: int = 40,
) -> pd.DataFrame:
    """Compute per-feature shift diagnostics and return a tidy dataframe."""

    results: List[FeatureShift] = []
    for feature in continuous_features:
        results.append(summarize_feature_shift(df, feature, feature_type="continuous", bins=bins))
    for feature in categorical_features:
        results.append(summarize_feature_shift(df, feature, feature_type="categorical", bins=bins))

    records = []
    for result in results:
        row = {"feature": result.feature, "type": result.feature_type, "shift_score": result.shift_score}
        row.update(result.metrics)
        records.append(row)

    summary = pd.DataFrame(records)
    weights = summary.get("support_min", pd.Series(1, index=summary.index))
    if not summary.empty:
        overall_score = float(np.average(summary["shift_score"], weights=weights))
        summary.loc[len(summary)] = {
            "feature": "__overall__",
            "type": "summary",
            "shift_score": overall_score,
        }
    return summary


def summarize_feature_shift(
    df: pd.DataFrame,
    feature: str,
    *,
    feature_type: str,
    bins: int = 40,
) -> FeatureShift:
    sim = df[df["domain"] == "sim"][feature].dropna().to_numpy()
    real = df[df["domain"] == "real"][feature].dropna().to_numpy()
    if len(sim) == 0 or len(real) == 0:
        raise ValueError(f"Insufficient samples for feature {feature}")

    if feature_type == "continuous":
        metrics = _continuous_metrics(sim, real, bins)
        shift_score = min(1.0, 0.4 * metrics["ks_stat"] + 0.3 * metrics["wasserstein_scaled"] + 0.3 * metrics["js_divergence"])
    else:
        metrics = _categorical_metrics(sim, real)
        shift_score = min(1.0, 0.6 * metrics["cramers_v"] + 0.4 * (1 - metrics["p_value"]))

    return FeatureShift(feature=feature, feature_type=feature_type, shift_score=shift_score, metrics=metrics)


def _continuous_metrics(sim: np.ndarray, real: np.ndarray, bins: int) -> dict:
    ks_stat, ks_p = stats.ks_2samp(sim, real, alternative="two-sided", method="auto")
    wasserstein = stats.wasserstein_distance(sim, real)
    std = np.std(np.concatenate([sim, real])) or 1.0
    wasserstein_scaled = min(1.0, wasserstein / (std * 2))
    js = _js_divergence(sim, real, bins)
    mean_sim = float(np.mean(sim))
    mean_real = float(np.mean(real))
    return {
        "ks_stat": float(ks_stat),
        "ks_p_value": float(ks_p),
        "wasserstein": float(wasserstein),
        "wasserstein_scaled": float(wasserstein_scaled),
        "js_divergence": float(js),
        "mean_sim": mean_sim,
        "mean_real": mean_real,
        "mean_delta": float(mean_real - mean_sim),
        "support_sim": int(sim.size),
        "support_real": int(real.size),
        "support_min": int(min(sim.size, real.size)),
    }


def _categorical_metrics(sim: np.ndarray, real: np.ndarray) -> dict:
    sim_counts = pd.Series(sim).value_counts()
    real_counts = pd.Series(real).value_counts()
    categories = sorted(set(sim_counts.index).union(real_counts.index))
    contingency = np.array([
        [sim_counts.get(cat, 0), real_counts.get(cat, 0)] for cat in categories
    ])
    chi2, p_value, _, expected = stats.chi2_contingency(contingency)
    n = contingency.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1))) if n > 0 else 0.0
    sim_probs = pd.Series({cat: sim_counts.get(cat, 0) for cat in categories}, dtype=float)
    real_probs = pd.Series({cat: real_counts.get(cat, 0) for cat in categories}, dtype=float)
    sim_probs = sim_probs / max(sim_probs.sum(), 1.0)
    real_probs = real_probs / max(real_probs.sum(), 1.0)
    deltas = real_probs - sim_probs
    if not deltas.empty:
        dominant_category = deltas.abs().idxmax()
        dominant_domain = "REAL" if deltas[dominant_category] > 0 else "SIM"
        dominant_delta = float(deltas[dominant_category])
        share_sim = float(sim_probs.get(dominant_category, 0.0))
        share_real = float(real_probs.get(dominant_category, 0.0))
    else:
        dominant_category = None
        dominant_domain = None
        dominant_delta = 0.0
        share_sim = 0.0
        share_real = 0.0
    return {
        "chi2_stat": float(chi2),
        "p_value": float(p_value),
        "cramers_v": float(cramers_v),
        "dominant_category": dominant_category,
        "dominant_domain": dominant_domain,
        "dominant_delta": dominant_delta,
        "share_sim": share_sim,
        "share_real": share_real,
        "support_sim": int(sim.size),
        "support_real": int(real.size),
        "support_min": int(min(sim.size, real.size)),
    }


def _js_divergence(sim: np.ndarray, real: np.ndarray, bins: int) -> float:
    min_val = float(min(sim.min(), real.min()))
    max_val = float(max(sim.max(), real.max()))
    if min_val == max_val:
        return 0.0
    hist_sim, bin_edges = np.histogram(sim, bins=bins, range=(min_val, max_val), density=True)
    hist_real, _ = np.histogram(real, bins=bin_edges, density=True)
    hist_sim = _smooth(hist_sim)
    hist_real = _smooth(hist_real)
    m = 0.5 * (hist_sim + hist_real)
    js = 0.5 * (_kl_div(hist_sim, m) + _kl_div(hist_real, m))
    return float(min(1.0, js / np.log(2)))


def _smooth(d: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    d = d.astype(float)
    d[d < eps] = eps
    d /= d.sum()
    return d


def _kl_div(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * np.log(p / q)))
