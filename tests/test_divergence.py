"""Divergence and separability tests."""

from sim2real.config import GeneratorConfig, CONTINUOUS_FEATURES, CATEGORICAL_FEATURES
from sim2real.data.synthetic import generate_paired_logs
from sim2real.shift.divergence import compute_shift_metrics
from sim2real.shift.domain_classifier import train_domain_classifier


def test_divergence_detects_shift() -> None:
    df = generate_paired_logs(GeneratorConfig(trips_per_domain=5, shift_strength=1.5))
    metrics = compute_shift_metrics(
        df,
        continuous_features=CONTINUOUS_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        bins=20,
    )
    speed_row = metrics[metrics["feature"] == "ego_speed_mps"].iloc[0]
    assert speed_row["ks_stat"] > 0.05
    assert speed_row["js_divergence"] > 0.05


def test_domain_classifier_auc_exceeds_threshold() -> None:
    df = generate_paired_logs(GeneratorConfig(trips_per_domain=6, shift_strength=1.0))
    result = train_domain_classifier(
        df,
        continuous_features=CONTINUOUS_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )
    assert result.auc > 0.6
