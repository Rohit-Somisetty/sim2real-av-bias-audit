# Sample SIM vs REAL Bias Audit (Excerpt)

## Key Findings
- Domain classifier AUC **0.748** → SIM and REAL logs are distinguishable but reweighting mitigates the largest gaps.
- Largest feature shifts: `weather`, `time_of_day`, `traffic_density`.
- Biggest safety rate gap: `hard_brake` with +14.5k events / 1k miles in REAL compared to SIM.

## Top Shifted Features
| feature | shift_score |
| --- | --- |
| weather | 0.498 |
| time_of_day | 0.467 |
| traffic_density | 0.455 |

## Reweighting Impact (Wasserstein)
| feature | before | after | improvement |
| --- | --- | --- | --- |
| ego_speed_mps | 1.197 | 0.508 | 0.689 |
| ego_accel_mps2 | 2.305 | 2.293 | 0.012 |
| ttc_s | 0.715 | 0.325 | 0.389 |

## Safety Event Comparison
| event_type | rate_sim | rate_real | delta_per_1000_miles |
| --- | --- | --- | --- |
| disengagement | 3309.6 | 3378.9 | +69.3 |
| hard_brake | 10361.8 | 24911.2 | +14549.4 |
| near_miss | 0.0 | 6128.4 | +6128.4 |

## Risky Slices (REAL worse)
| slice_dim | slice_value | miles_sim | miles_real | max_event_gap_per_1000_miles |
| --- | --- | --- | --- | --- |
| weather | clear | 24.4 | 19.6 | 15295.1 |
| traffic_density | medium | 13.5 | 12.7 | 14772.8 |
| time_of_day | day | 20.9 | 21.9 | 11668.6 |

👉 Run `python -m sim2real.cli analyze --data outputs/data.parquet --outdir outputs` to generate the full report in `outputs/report.md`.
