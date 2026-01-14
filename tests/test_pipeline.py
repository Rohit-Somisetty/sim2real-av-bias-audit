"""End-to-end CLI test."""

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from sim2real.cli import app


def test_cli_pipeline(tmp_path: Path) -> None:
    runner = CliRunner()
    data_path = tmp_path / "data.parquet"
    outdir = tmp_path / "outputs"

    result = runner.invoke(
        app,
        [
            "generate-data",
            "--out",
            str(data_path),
            "--trips",
            "5",
            "--seed",
            "0",
            "--shift-strength",
            "1.0",
        ],
    )
    assert result.exit_code == 0, result.stdout

    result = runner.invoke(
        app,
        [
            "analyze",
            "--data",
            str(data_path),
            "--outdir",
            str(outdir),
            "--bins",
            "15",
            "--min-slice-miles",
            "1.0",
        ],
    )
    assert result.exit_code == 0, result.stdout

    expected = [
        outdir / "shift_features.csv",
        outdir / "shift_features_reweighted.csv",
        outdir / "shift_slices.csv",
        outdir / "domain_classifier.csv",
        outdir / "event_rate_compare.csv",
        outdir / "event_rate_compare_reweighted.csv",
        outdir / "report.md",
        outdir / "reweighting_effect.csv",
        outdir / "shift_summary.json",
    ]
    for path in expected:
        assert path.exists(), f"Missing artifact {path}"

    plots_dir = outdir / "plots"
    assert plots_dir.exists()
    assert (plots_dir / "shift_score_top_features.png").exists()


def test_merge_domains_cli(tmp_path: Path) -> None:
    runner = CliRunner()
    sim_csv = tmp_path / "sim.csv"
    real_csv = tmp_path / "real.csv"
    out_path = tmp_path / "merged.parquet"

    sim_df = pd.DataFrame(
        [
            {
                "trip_id": "sim_trip",
                "timestamp_s": 0.0,
                "dt_s": 0.1,
                "ego_speed_mps": 12.0,
                "ego_accel_mps2": 0.1,
                "lane_offset_m": 0.02,
                "ttc_s": 6.0,
                "engaged": True,
                "weather": "clear",
                "time_of_day": "day",
                "traffic_density": "low",
            }
        ]
    )
    real_df = pd.DataFrame(
        [
            {
                "trip_id": "real_trip",
                "timestamp_s": 0.0,
                "dt_s": 0.1,
                "ego_speed_mps": 10.0,
                "ego_accel_mps2": -0.3,
                "lane_offset_m": 0.05,
                "ttc_s": 3.5,
                "engaged": False,
                "weather": "rain",
                "time_of_day": "night",
                "traffic_density": "high",
            }
        ]
    )
    sim_df.to_csv(sim_csv, index=False)
    real_df.to_csv(real_csv, index=False)

    result = runner.invoke(
        app,
        [
            "merge-domains",
            "--sim",
            str(sim_csv),
            "--real",
            str(real_csv),
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    merged = pd.read_parquet(out_path)
    assert set(merged["domain"].unique()) == {"sim", "real"}
