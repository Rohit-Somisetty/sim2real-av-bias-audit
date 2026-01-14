"""Typer-powered CLI for the sim2real bias audit toolkit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import typer

from sim2real.config import (
    AnalysisConfig,
    GeneratorConfig,
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    DEFAULT_ANALYSIS_CONFIG,
)
from sim2real.data import ParquetLoader, generate_paired_logs, validate_frame_schema
from sim2real.report import build_markdown_report
from sim2real.safety.events import summarize_events
from sim2real.safety.rate_compare import compare_event_rates, compute_event_rates
from sim2real.shift.divergence import compute_shift_metrics
from sim2real.shift.domain_classifier import train_domain_classifier
from sim2real.shift.reweighting import DensityRatioReweighter
from sim2real.shift.slicing import compute_slice_gaps
from sim2real.viz import (
    plot_categorical_distributions,
    plot_continuous_distributions,
    plot_event_rate_comparison,
    plot_shift_leaders,
)

app = typer.Typer(add_completion=False)


@app.command("generate-data")
def generate_data(
    out: Path = typer.Option(..., help="Output parquet path"),
    trips: int = typer.Option(200, help="Trips per domain"),
    shift_strength: float = typer.Option(1.0, help="Shift strength multiplier"),
    seed: int = typer.Option(42, help="Random seed"),
    hz: float = typer.Option(10.0, help="Sampling frequency"),
) -> None:
    cfg = GeneratorConfig(trips_per_domain=trips, shift_strength=shift_strength, seed=seed, hz=hz)
    df = generate_paired_logs(cfg, out=out)
    typer.echo(f"Generated {len(df):,} rows to {out}")


@app.command()
def analyze(
    data: Path = typer.Option(..., help="Parquet file with SIM+REAL frames"),
    outdir: Path = typer.Option(Path("outputs"), help="Directory for artifacts"),
    bins: int = typer.Option(DEFAULT_ANALYSIS_CONFIG.bins, help="Histogram bins for JS divergence"),
    min_slice_miles: float = typer.Option(
        DEFAULT_ANALYSIS_CONFIG.min_slice_miles, help="Minimum miles per domain for slice output"
    ),
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    loader = ParquetLoader(data)
    df = loader.load()

    analysis_cfg = AnalysisConfig(
        bins=bins,
        min_slice_miles=min_slice_miles,
        plots_dir=outdir / "plots",
    )
    plots_dir = analysis_cfg.plots_dir or (outdir / "plots")

    shift_metrics = compute_shift_metrics(
        df,
        continuous_features=analysis_cfg.continuous_features,
        categorical_features=analysis_cfg.categorical_features,
        bins=analysis_cfg.bins,
    )
    shift_path = outdir / "shift_features.csv"
    shift_metrics.to_csv(shift_path, index=False)

    domain_result = train_domain_classifier(
        df,
        continuous_features=analysis_cfg.continuous_features,
        categorical_features=analysis_cfg.categorical_features,
    )
    dom_records = [{"metric": "auc", "feature": "", "value": domain_result.auc}]
    dom_records.extend(
        {
            "metric": "importance",
            "feature": row.feature,
            "value": row.weight,
        }
        for row in domain_result.feature_importances.itertuples()
    )
    pd.DataFrame(dom_records).to_csv(outdir / "domain_classifier.csv", index=False)

    reweighter = DensityRatioReweighter(
        continuous_features=analysis_cfg.continuous_features,
        categorical_features=analysis_cfg.categorical_features,
    ).fit(df)
    reweighted_df = reweighter.build_reweighted_frame(df, random_state=42)
    reweight_report = reweighter.evaluate(df, analysis_cfg.continuous_features)
    reweight_report.to_csv(outdir / "reweighting_effect.csv", index=False)

    shift_metrics_reweighted = compute_shift_metrics(
        reweighted_df,
        continuous_features=analysis_cfg.continuous_features,
        categorical_features=analysis_cfg.categorical_features,
        bins=analysis_cfg.bins,
    )
    shift_metrics_reweighted.to_csv(outdir / "shift_features_reweighted.csv", index=False)

    events = summarize_events(df)
    rates = compute_event_rates(events)
    rate_compare = compare_event_rates(rates)
    rate_compare.to_csv(outdir / "event_rate_compare.csv", index=False)

    events_reweighted = summarize_events(reweighted_df)
    rates_reweighted = compute_event_rates(events_reweighted)
    rate_compare_reweighted = compare_event_rates(rates_reweighted)
    rate_compare_reweighted.to_csv(outdir / "event_rate_compare_reweighted.csv", index=False)

    slice_df = compute_slice_gaps(
        df,
        slice_cols=("weather", "time_of_day", "traffic_density"),
        continuous_features=analysis_cfg.continuous_features,
        categorical_features=analysis_cfg.categorical_features,
        min_slice_miles=analysis_cfg.min_slice_miles,
    )
    slice_df.to_csv(outdir / "shift_slices.csv", index=False)

    summary_payload = _create_shift_summary(
        domain_auc=domain_result.auc,
        shift_df=shift_metrics,
        shift_df_reweighted=shift_metrics_reweighted,
        rate_compare=rate_compare,
        rate_compare_reweighted=rate_compare_reweighted,
        slice_df=slice_df,
        reweight_df=reweight_report,
    )
    (outdir / "shift_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    key_findings = summary_payload["key_findings"]

    plot_continuous_distributions(df, analysis_cfg.continuous_features, plots_dir)
    plot_categorical_distributions(df, analysis_cfg.categorical_features, plots_dir)
    plot_shift_leaders(shift_metrics, plots_dir)
    plot_shift_leaders(shift_metrics, plots_dir, filename="shift_score_top_features.png")
    plot_event_rate_comparison(rate_compare, plots_dir)

    report_path = outdir / "report.md"
    build_markdown_report(
        report_path,
        domain_auc=domain_result.auc,
        shift_df=shift_metrics,
        reweight_df=reweight_report,
        rate_compare_df=rate_compare,
        slice_df=slice_df,
        key_findings=key_findings,
    )

    typer.echo(f"Analysis artifacts saved under {outdir}")


@app.command()
def reweight(
    data: Path = typer.Option(..., help="Parquet file with SIM+REAL frames"),
    out: Path = typer.Option(..., help="Path for weighted parquet"),
) -> None:
    df = ParquetLoader(data).load()
    reweighter = DensityRatioReweighter(
        continuous_features=CONTINUOUS_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    ).fit(df)
    weighted = reweighter.attach_weights(df)
    out.parent.mkdir(parents=True, exist_ok=True)
    weighted.to_parquet(out, index=False)
    typer.echo(f"Wrote reweighted parquet with importance weights to {out}")


@app.command("merge-domains")
def merge_domains(
    sim: Path = typer.Option(..., help="CSV or Parquet file containing simulator frames"),
    real: Path = typer.Option(..., help="CSV or Parquet file containing real-world frames"),
    out: Path = typer.Option(..., help="Destination parquet for merged data"),
) -> None:
    sim_df = _load_user_log(sim, domain_label="sim")
    real_df = _load_user_log(real, domain_label="real")
    merged = pd.concat([sim_df, real_df], ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    typer.echo(f"Merged {len(merged):,} rows into {out}")


def _create_shift_summary(
    *,
    domain_auc: float,
    shift_df: pd.DataFrame,
    shift_df_reweighted: pd.DataFrame,
    rate_compare: pd.DataFrame,
    rate_compare_reweighted: pd.DataFrame,
    slice_df: pd.DataFrame,
    reweight_df: pd.DataFrame,
) -> Dict[str, Any]:
    overall_shift_pre = _overall_shift_score(shift_df)
    overall_shift_post = _overall_shift_score(shift_df_reweighted)
    top_features = _top_shift_feature_directions(shift_df)
    safety_gap = _biggest_safety_gap(rate_compare)
    safety_gap_post = _biggest_safety_gap(rate_compare_reweighted)
    risky_slice = _top_risky_slice(slice_df)
    reweighting = _reweighting_summary(reweight_df)
    key_findings = _build_key_findings(
        domain_auc=domain_auc,
        top_features=top_features,
        safety_gap=safety_gap,
        risky_slice=risky_slice,
        reweighting=reweighting,
    )
    return {
        "domain_auc": float(domain_auc),
        "overall_shift_score_pre": overall_shift_pre,
        "overall_shift_score_post": overall_shift_post,
        "top_shifted_features": top_features,
        "biggest_safety_gap": safety_gap,
        "biggest_safety_gap_post": safety_gap_post,
        "top_risky_slice": risky_slice,
        "reweighting_effect": reweighting,
        "key_findings": key_findings,
    }


def _overall_shift_score(shift_df: pd.DataFrame) -> float:
    summary = shift_df[shift_df["feature"] == "__overall__"]
    if summary.empty:
        return 0.0
    return float(summary["shift_score"].iloc[0])


def _top_shift_feature_directions(shift_df: pd.DataFrame, limit: int = 3) -> List[Dict[str, Any]]:
    filtered = shift_df[shift_df["feature"] != "__overall__"].copy()
    records: List[Dict[str, Any]] = []
    if filtered.empty:
        return records
    for _, row in filtered.sort_values("shift_score", ascending=False).head(limit).iterrows():
        entry: Dict[str, Any] = {
            "feature": row["feature"],
            "shift_score": float(row["shift_score"]),
            "type": row["type"],
        }
        if row["type"] == "continuous":
            delta = float(row.get("mean_delta", 0.0))
            entry["direction"] = "REAL > SIM" if delta > 0 else "SIM > REAL"
            entry["delta"] = delta
            entry["detail"] = f"Δmean={delta:+.3f}"
        else:
            delta = float(row.get("dominant_delta", 0.0))
            domain = row.get("dominant_domain") or ("REAL" if delta > 0 else "SIM")
            category = row.get("dominant_category") or "category"
            entry["direction"] = "REAL > SIM" if delta > 0 else "SIM > REAL"
            entry["delta"] = delta
            entry["detail"] = f"{domain} favors {category} ({delta:+.3f})"
        records.append(entry)
    return records


def _biggest_safety_gap(rate_compare: pd.DataFrame) -> Dict[str, Any] | None:
    if rate_compare.empty:
        return None
    idx = rate_compare["delta_per_1000_miles"].abs().idxmax()
    row = rate_compare.loc[idx]
    return {
        "event_type": row["event_type"],
        "delta_per_1000_miles": float(row["delta_per_1000_miles"]),
        "rate_real": float(row["rate_real"]),
        "rate_sim": float(row["rate_sim"]),
        "p_real_gt_sim": float(row.get("p_real_gt_sim", 0.5)),
    }


def _top_risky_slice(slice_df: pd.DataFrame) -> Dict[str, Any] | None:
    if slice_df.empty or "max_event_gap_per_1000_miles" not in slice_df.columns:
        return None
    filtered = slice_df[slice_df["max_event_gap_per_1000_miles"] > 0]
    if filtered.empty:
        return None
    row = filtered.sort_values("max_event_gap_per_1000_miles", ascending=False).iloc[0]
    return {
        "slice_dim": row["slice_dim"],
        "slice_value": row["slice_value"],
        "max_event_gap_per_1000_miles": float(row["max_event_gap_per_1000_miles"]),
    }


def _reweighting_summary(reweight_df: pd.DataFrame) -> Dict[str, Any] | None:
    if reweight_df.empty:
        return None
    row = reweight_df.sort_values("improvement", ascending=False).iloc[0]
    improvement = float(row["improvement"])
    if improvement <= 0:
        return None
    before = float(row.get("wasserstein_before", 0.0))
    percent = improvement / max(before, 1e-9) * 100.0 if before > 0 else 0.0
    return {
        "feature": row["feature"],
        "wasserstein_before": before,
        "wasserstein_after": float(row.get("wasserstein_after", 0.0)),
        "absolute_reduction": improvement,
        "percent_reduction": percent,
    }


def _build_key_findings(
    *,
    domain_auc: float,
    top_features: List[Dict[str, Any]],
    safety_gap: Dict[str, Any] | None,
    risky_slice: Dict[str, Any] | None,
    reweighting: Dict[str, Any] | None,
) -> List[str]:
    findings: List[str] = []
    findings.append(
        f"Domain classifier AUC {domain_auc:.3f} — higher values indicate stronger SIM vs REAL separation."
    )

    if top_features:
        parts = [f"{item['feature']} ({item['direction']})" for item in top_features]
        findings.append("Top shifted features: " + ", ".join(parts))
    else:
        findings.append("Top shifted features: none detected.")

    if safety_gap:
        delta = safety_gap["delta_per_1000_miles"]
        direction = "REAL worse" if delta > 0 else "REAL better"
        findings.append(
            f"Biggest safety rate gap: {safety_gap['event_type']} Δ={delta:+.2f}/1k miles ({direction})."
        )
    else:
        findings.append("No significant safety rate gaps detected.")

    if risky_slice:
        findings.append(
            "Risky slice: {dim}={val} (+{gap:.2f}/1k miles REAL vs SIM).".format(
                dim=risky_slice["slice_dim"],
                val=risky_slice["slice_value"],
                gap=risky_slice["max_event_gap_per_1000_miles"],
            )
        )
    else:
        findings.append("No slice cleared the exposure bar with higher REAL risk.")

    if reweighting:
        findings.append(
            f"Reweighting cut {reweighting['feature']} divergence by {reweighting['percent_reduction']:.1f}% (Wasserstein)."
        )
    else:
        findings.append("Reweighting produced negligible divergence improvements.")

    return findings[:5]


def _load_user_log(path: Path, *, domain_label: str) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    frame = frame.copy()
    frame["domain"] = domain_label
    validated = validate_frame_schema(frame)
    return validated


if __name__ == "__main__":  # pragma: no cover
    app()
