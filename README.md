# Flight Delay Prediction

Personal machine-learning portfolio project that classifies U.S. carrier/airport/
month flight records as "high delay" (more than 25% of flights delayed by 15+
minutes) or "normal operations". The repo delivers a leakage-safe training
pipeline, a reproducible Streamlit dashboard, and a Markdown case study.

> Deployment pending — the live Streamlit Community Cloud URL will replace
> this notice after publication.

## Highlights

- Leakage-safe chronological evaluation: training 2003–2022, validation 2023,
  final test 2024, drift check 2025-Jan through 2025-Jul.
- Three candidate models compared on validation F1; one fitted pipeline
  persisted to `models/model_pipeline.joblib`.
- Small, deterministic dashboard dataset for reliable Streamlit deployment.
- Reproducible data acquisition script that validates the official BTS
  snapshot's checksum, row count, and column contract.

## Repository layout

```
data/                 validated snapshot metadata and dashboard artifact
docs/case-study.md    full methodology and results
docs/assets/          model comparison and confusion matrix figures
models/               fitted pipeline + metadata JSON
notebooks/            reproducible narrative
scripts/              CLI entrypoints (download, build, train)
src/ml/               reusable data, feature, training, evaluation modules
src/app.py            Streamlit dashboard
tests/                pytest suite covering contract, training, app, scripts
```

## Quick start

```bash
python3.11 -m venv venv
venv/bin/pip install -r requirements-dev.txt

# Acquire the BTS snapshot manually from
# https://www.transtats.bts.gov/ot_delay/ot_delaycause1.asp?pn=1
# for June 2003 through July 2025 and save as a CSV.

venv/bin/python scripts/download_data.py --source-file <downloaded.csv>
venv/bin/python scripts/build_dashboard_data.py
venv/bin/python scripts/train_model.py
venv/bin/streamlit run src/app.py
```

## Tests

```bash
venv/bin/python -m pytest -q
```

## Data

The raw dataset is not committed. `scripts/download_data.py` validates the
SHA-256, row count, and column contract of a manually downloaded BTS snapshot
before importing it. `scripts/build_dashboard_data.py` produces a deterministic
2020–July 2025 dataset used by the deployed app.

## Limitations

This is a portfolio project. The model is built on historical U.S. domestic
data, cannot capture weather or air-traffic events, and is not suitable for
operational airline decisions. See `docs/case-study.md` for details.

## License

MIT. See `LICENSE`.

## Author

Shahboz Munirov — <https://github.com/shakhbozmn>.