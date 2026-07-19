"""Reproducible training entrypoint.

Reads the pinned BTS snapshot, runs the leakage-safe selection workflow,
and writes the fitted pipeline plus model metadata. The script never
downloads the dataset; the snapshot must already live under ``data/raw/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.ml.artifacts import build_metadata
from src.ml.data import build_target, split_by_period, validate_dataset
from src.ml.evaluation import save_confusion_matrix_figure, save_model_comparison_figure
from src.ml.training import (
    library_versions,
    run_validation,
    save_model_bundle,
    select_best_validation_result,
    _candidate_pipelines,
)

AUTHOR_NAME = "Shahboz Munirov"
SOURCE_SHA256 = "383fb1ae404cc46aa9380bbc8156fdf6e2e4bd5af7ae1197717a639a92378134"
TARGET_THRESHOLD = 0.25

PARTITION_LABELS: dict[str, str] = {
    "train": "2003-2022",
    "validation": "2023",
    "test": "2024",
    "drift": "2025-01..2025-07",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/Airline_Delay_Cause.csv"),
        help="Validated BTS snapshot path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models"),
        help="Directory for the model artifact and metadata.",
    )
    parser.add_argument(
        "--assets-output",
        type=Path,
        default=Path("docs/assets"),
        help="Directory for comparison and confusion matrix figures.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.data.exists():
        print(f"snapshot not found: {args.data}")
        return 1

    frame = pd.read_csv(args.data, low_memory=False)
    validate_dataset(frame)
    features, target = build_target(frame, threshold=TARGET_THRESHOLD)
    splits = split_by_period(features, target)

    validation_results = run_validation(splits, random_state=args.random_state)
    selected_model = select_best_validation_result(validation_results)
    pipelines = _candidate_pipelines(splits.y_train, random_state=args.random_state)
    chosen = pipelines[selected_model]

    fit_frame = pd.concat(
        [splits.x_train, splits.x_validation], ignore_index=True
    )
    fit_target = pd.concat(
        [splits.y_train, splits.y_validation], ignore_index=True
    )
    chosen.fit(fit_frame, fit_target)

    test_predictions = chosen.predict(splits.x_test)
    test_scores = chosen.predict_proba(splits.x_test)[:, 1]

    from src.ml.evaluation import metric_bundle
    import numpy as np

    test_metrics = metric_bundle(
        np.asarray(splits.y_test), test_predictions, test_scores
    )

    drift_metrics: dict[str, float] = {}
    if not splits.x_drift.empty:
        drift_predictions = chosen.predict(splits.x_drift)
        drift_scores = chosen.predict_proba(splits.x_drift)[:, 1]
        drift_metrics = metric_bundle(
            np.asarray(splits.y_drift), drift_predictions, drift_scores
        )

    metadata = build_metadata(
        author=AUTHOR_NAME,
        source_sha256=SOURCE_SHA256,
        target_threshold=TARGET_THRESHOLD,
        partitions=PARTITION_LABELS,
        selected_model=selected_model,
        validation_results=validation_results,
        test_metrics=test_metrics,
        drift_metrics=drift_metrics,
        library_versions=library_versions(),
    )

    output_dir = args.output.expanduser().resolve()
    paths = save_model_bundle(chosen, metadata, output_dir)

    assets_dir = args.assets_output.expanduser().resolve()
    save_model_comparison_figure(validation_results, assets_dir / "model-comparison.png")
    save_confusion_matrix_figure(
        np.asarray(splits.y_test),
        test_predictions,
        assets_dir / "confusion-matrix.png",
    )

    summary = {
        "selected_model": selected_model,
        "test_metrics": test_metrics,
        "drift_metrics": drift_metrics,
        "pipeline_path": str(paths["pipeline_path"]),
        "metadata_path": str(paths["metadata_path"]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())