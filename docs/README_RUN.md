# Runbook: sim2real-av-bias-audit

This runbook reproduces the full audit from a clean checkout. Commands assume a POSIX shell; replace `source .venv/bin/activate` with `.venv\\Scripts\\activate` on Windows PowerShell.

## 1. Environment setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## 2. Generate paired SIM/REAL logs
```bash
python -m sim2real.cli generate-data --out outputs/data.parquet --trips 200 --seed 42
```
_Expected artifacts_: `outputs/data.parquet`

## 3. Run the core audit
```bash
python -m sim2real.cli analyze --data outputs/data.parquet --outdir outputs
```
_Expected artifacts_:
- `outputs/metrics_overall.csv`, `metrics_slices.csv`, `metrics_trends.csv`
- `outputs/anomalies.csv` for high-risk slices
- `outputs/plots/*.png` mirroring the docs/figures samples
- `outputs/report.md` (full Markdown executive summary)
- `outputs/summary.json` (machine-readable key metrics)

## 4. Optional: Importance reweighting pass
```bash
python -m sim2real.cli reweight --data outputs/data.parquet --out outputs/reweighted.parquet
```
_Expected artifacts_: `outputs/reweighted.parquet` containing density ratios via `weight`

## 5. Bring your own logs (optional)
```bash
python -m sim2real.cli merge-domains --sim docs/sample_sim.csv --real docs/sample_real.csv --out outputs/merged.parquet
python -m sim2real.cli analyze --data outputs/merged.parquet --outdir outputs
```
_Expected artifacts_: `outputs/merged.parquet` plus the full audit bundle above.

## 6. Run the test suite
```bash
pytest -q
```
_Expected outcome_: `8 passed` in under a minute on a modern laptop.
