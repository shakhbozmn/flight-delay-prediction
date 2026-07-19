# Flight Delay Prediction — Technical Case Study

## Overview

This case study documents the methodology and results of a personal machine-learning
portfolio project that predicts whether a U.S. carrier/airport/month record will
experience a high delay rate (>25% of flights delayed by 15 minutes or more).

## Data provenance

- Source: U.S. Department of Transportation, Bureau of Transportation Statistics
  ([Airline On-Time Statistics and Delay Causes](https://www.transtats.bts.gov/ot_delay/ot_delaycause1.asp?pn=1)).
- License: U.S. Government work (17 U.S.C. §105); public-domain reuse per BTS policy.
- Pinned snapshot: 409,612 monthly carrier/airport records covering June 2003
  through July 2025 (SHA-256
  `383fb1ae404cc46aa9380bbc8156fdf6e2e4bd5af7ae1197717a639a92378134`).
- Reproducible acquisition: download the raw CSV from the BTS dashboard for the
  period June 2003 to July 2025, then run
  `python scripts/download_data.py --source-file <downloaded.csv>`.

## Problem framing

The binary target is whether a carrier/airport/month row meets the definition of
"high delay month". Rows with zero flights are excluded because the target is
undefined. The training partition uses years 2003–2022; validation is 2023;
final held-out test is 2024; a supplemental drift check covers January–July 2025.

## Feature engineering

The deterministic feature builder (`src/ml/features.py`) computes calendar and
operational metrics for each row:

- Calendar: `quarter`, `is_winter`, `is_summer`, `is_peak_travel`.
- Operational: `flights_per_day` (using calendar days in the month),
  `cancellation_rate`, `total_disruptions`.
- Raw inputs: `year`, `month`, `carrier`, `airport`, `arr_flights`,
  `arr_cancelled`, `arr_diverted`.

The transformer is stateless: it learns no statistics in `fit`, so it is safe
to call before the train/test split. Categorical encoding and numeric scaling
happen inside the model pipeline after the split, which removes the leakage
present in the original notebook.

## Candidate models

Three scikit-learn/XGBoost candidates share the same preprocessing:

| Model | Notes |
|-------|-------|
| Logistic Regression | Linear baseline with balanced class weights and 2000 max iterations. |
| Random Forest | 300 trees, minimum leaf size 5, balanced class weights. |
| XGBoost | 300 trees, max depth 6, learning rate 0.1, scale_pos_weight from training class ratio. |

Selection picks the candidate with the highest validation F1, breaking ties by
PR-AUC, then ROC-AUC, then name.

## Evaluation

The pipeline reports six metrics — accuracy, F1, precision, recall, ROC-AUC,
and PR-AUC — for validation (2023), test (2024), and the 2025 drift snapshot.
The 2024 metrics are the held-out headline result; the 2025 partial-period
metrics are presented as informational drift only.

## App architecture

The Streamlit dashboard (`src/app.py`) loads a deterministic, smaller
dashboard dataset (`data/dashboard_data.csv.gz`) plus the fitted pipeline and
metadata. It exposes three pages — Overview, Data Exploration, and Model
Prediction — that match the metrics reported in `models/model_metadata.json`.

The "Model Prediction" page is a scenario estimator: the user enters the year,
month, carrier, airport, and forecast operational volumes, and the fitted
pipeline returns the predicted probability of exceeding a 25% delay rate. The
page surfaces a clear validation error if the user enters cancellations plus
diversions greater than total flights.

## Limitations

- The model is built from historical U.S. domestic data and does not capture
  weather, air-traffic events, or economic conditions.
- Predictions cover monthly aggregates, not daily forecasts. A 25% threshold is
  a working definition; alternatives (e.g., 20% or 30%) change the class balance
  and decision trade-offs.
- Carrier and airport encoders are learned from the training data; rare or
  unseen carriers default to a low-frequency prior via the one-hot encoder's
  `handle_unknown="ignore"` setting.
- Results are illustrative and not suitable for operational airline planning.

## Ethical and intended use

This project is intended as a portfolio demonstration of leakage-safe machine
learning on public government data. It must not be used as the basis for
financial, operational, or scheduling decisions affecting passengers or staff.

## Reproducibility

```bash
python scripts/download_data.py --source-file <bts-csv>
python scripts/build_dashboard_data.py
python scripts/train_model.py
pytest -q
```

The `scripts/train_model.py` entrypoint prints a JSON summary with selected
model and held-out metrics.

## Future work

- Calibrated probabilities for threshold tuning.
- Time-series features capturing year-over-year trends.
- External data integrations (weather, NAS events, holidays).
- Multi-class delay severity prediction (low/medium/high).

## References

- U.S. Bureau of Transportation Statistics. Airline On-Time Statistics and
  Delay Causes. <https://www.transtats.bts.gov/ot_delay/ot_delaycause1.asp?pn=1>
- Pedregosa et al. Scikit-learn: Machine Learning in Python. *JMLR*, 2011.
- Chen & Guestrin. XGBoost: A Scalable Tree Boosting System. *KDD*, 2016.