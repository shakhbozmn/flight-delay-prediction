"""Artifact metadata validation and serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path

REQUIRED_METADATA_KEYS: tuple[str, ...] = (
    "schema_version",
    "author",
    "source_sha256",
    "target_threshold",
    "partitions",
    "selected_model",
    "validation_results",
    "test_metrics",
    "drift_metrics",
    "library_versions",
)

SUPPORTED_SCHEMA_VERSIONS: tuple[int, ...] = (1,)


def build_metadata(
    *,
    author: str,
    source_sha256: str,
    target_threshold: float,
    partitions: dict[str, str],
    selected_model: str,
    validation_results: dict[str, dict[str, float]],
    test_metrics: dict[str, float],
    drift_metrics: dict[str, float],
    library_versions: dict[str, str],
) -> dict[str, object]:
    """Assemble the metadata dictionary and validate it against the contract."""

    metadata = {
        "schema_version": 1,
        "author": author,
        "source_sha256": source_sha256,
        "target_threshold": target_threshold,
        "partitions": partitions,
        "selected_model": selected_model,
        "validation_results": validation_results,
        "test_metrics": test_metrics,
        "drift_metrics": drift_metrics,
        "library_versions": library_versions,
    }
    validate_metadata(metadata)
    return metadata


def validate_metadata(metadata: dict) -> None:
    """Raise ``ValueError`` when the metadata violates the artifact contract."""

    missing = [key for key in REQUIRED_METADATA_KEYS if key not in metadata]
    if missing:
        raise ValueError(
            "metadata is missing required keys: " + ", ".join(missing)
        )

    schema_version = metadata["schema_version"]
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"unsupported metadata schema_version: {schema_version}"
        )

    if not metadata["validation_results"]:
        raise ValueError("metadata.validation_results is empty")
    if not metadata["test_metrics"]:
        raise ValueError("metadata.test_metrics is empty")
    if not metadata["library_versions"]:
        raise ValueError("metadata.library_versions is empty")


def load_metadata(path: Path) -> dict:
    """Load and validate metadata from disk."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"metadata file not found: {path}")
    metadata = json.loads(path.read_text())
    validate_metadata(metadata)
    return metadata