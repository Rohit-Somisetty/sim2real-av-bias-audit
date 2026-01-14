"""Safety rate comparison tests."""

import pandas as pd

from sim2real.safety.rate_compare import compare_event_rates, compute_event_rates


def test_rate_compare_probability_balanced_counts() -> None:
    summary = pd.DataFrame(
        [
            {"domain": "sim", "event_type": "hard_brake", "count": 10, "miles": 100.0},
            {"domain": "real", "event_type": "hard_brake", "count": 10, "miles": 100.0},
        ]
    )
    rates = compute_event_rates(summary)
    compare = compare_event_rates(rates)
    prob = compare.loc[compare["event_type"] == "hard_brake", "p_real_gt_sim"].iat[0]
    assert 0.4 < prob < 0.6
