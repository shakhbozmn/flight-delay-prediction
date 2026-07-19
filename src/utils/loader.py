"""Project-root aware loaders for the dashboard data and trained model."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DASHBOARD_PATH = REPO_ROOT / "data" / "dashboard_data.csv.gz"
DASHBOARD_METADATA_PATH = REPO_ROOT / "data" / "dashboard_metadata.json"
PIPELINE_PATH = REPO_ROOT / "models" / "model_pipeline.joblib"
METADATA_PATH = REPO_ROOT / "models" / "model_metadata.json"


def load_dashboard_dataset(
    path: Path | None = None, metadata_path: Path | None = None
) -> Optional[pd.DataFrame]:
    """Load the deployment dashboard data with checksum validation."""

    target = path or DASHBOARD_PATH
    meta = metadata_path or DASHBOARD_METADATA_PATH

    try:
        with gzip.open(target, "rb") as handle:
            frame = pd.read_csv(handle)
    except FileNotFoundError:
        _notify_error(f"Dashboard data not found at {target}")
        return None
    except (OSError, gzip.BadGzipFile) as exc:
        _notify_error(f"Failed to read dashboard data: {exc}")
        return None

    expected = _expected_dashboard_sha(meta)
    if expected is None:
        return frame
    actual = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            actual.update(chunk)
    if actual.hexdigest() != expected:
        _notify_error(
            "Dashboard data checksum mismatch; rebuild via "
            "scripts/build_dashboard_data.py"
        )
        return None
    return frame


def load_model_resources(
    model_path: Path | None = None,
    metadata_path: Path | None = None,
) -> tuple[Optional[object], Optional[dict]]:
    """Load the fitted pipeline and validated metadata."""

    target_pipeline = model_path or PIPELINE_PATH
    target_metadata = metadata_path or METADATA_PATH

    try:
        pipeline = joblib.load(target_pipeline)
    except FileNotFoundError:
        _notify_error(f"Model pipeline not found at {target_pipeline}")
        return None, None
    except Exception as exc:  # joblib raises broad exception types
        _notify_error(f"Failed to load model pipeline: {exc}")
        return None, None

    if not target_metadata.exists():
        _notify_error(f"Model metadata not found at {target_metadata}")
        return pipeline, None

    try:
        from src.ml.artifacts import load_metadata

        metadata = load_metadata(target_metadata)
    except (json.JSONDecodeError, ValueError, FileNotFoundError) as exc:
        _notify_error(f"Failed to load model metadata: {exc}")
        return pipeline, None

    return pipeline, metadata


def get_dataset_info(df: pd.DataFrame) -> dict:
    """Return summary statistics used by the Overview page."""

    return {
        "total_records": df.shape[0],
        "total_features": df.shape[1],
        "time_period": f"{df['year'].min()}-{df['year'].max()}",
        "total_airlines": df["carrier"].nunique(),
        "total_airports": df["airport"].nunique(),
    }


def _notify_error(message: str) -> None:
    try:
        import streamlit as st

        st.error(message)
    except Exception:
        print(message)


def _expected_dashboard_sha(metadata_path: Path) -> Optional[str]:
    if not metadata_path.exists():
        return None
    try:
        payload = json.loads(metadata_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return payload.get("output_sha256")