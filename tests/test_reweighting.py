"""Reweighting behavior tests."""

from sim2real.config import GeneratorConfig, CONTINUOUS_FEATURES, CATEGORICAL_FEATURES
from sim2real.data.synthetic import generate_paired_logs
from sim2real.shift.reweighting import DensityRatioReweighter


def test_reweighting_reduces_shift_score_for_speed() -> None:
    df = generate_paired_logs(GeneratorConfig(trips_per_domain=6, shift_strength=1.2))
    reweighter = DensityRatioReweighter(
        continuous_features=CONTINUOUS_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    ).fit(df)
    report = reweighter.evaluate(df, ["ego_speed_mps"])
    delta = report.iloc[0]["improvement"]
    assert delta > 0.0
