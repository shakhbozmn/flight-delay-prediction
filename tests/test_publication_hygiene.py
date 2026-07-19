"""Tracked-tree hygiene guard.

The block-list below is assembled from string fragments so prohibited exact
identifiers never appear in the test source itself. The test scans tracked
text files for the assembled terms and fails the build if any are present.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

STUDENT_NUMBER = "16" + "395"
BLOCKED_TERMS: tuple[str, ...] = (
    STUDENT_NUMBER,
    "000" + STUDENT_NUMBER,
    "Student" + " ID",
    "ML" + "DA",
    "CW" + "-1",
    "course" + "work",
    "module " + "course" + "work",
    "" + STUDENT_NUMBER,
    "" + STUDENT_NUMBER,
    "flight-delay-" + STUDENT_NUMBER,
    "Clau" + "de",
    "Anthro" + "pic",
    "Co-" + "Authored-" + "By",
    "Generated with " + "Clau" + "de",
    "AI-" + "generated",
)

IGNORED_PATH_SUFFIXES: tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pkl",
    ".joblib",
    ".gz",
    ".zip",
    ".parquet",
)


def _tracked_text_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        capture_output=True,
        cwd=str(REPO_ROOT),
    )
    raw = completed.stdout.decode("utf-8")
    paths = [Path(name) for name in raw.split("\0") if name]
    text_paths = [
        path
        for path in paths
        if not path.name.endswith(IGNORED_PATH_SUFFIXES)
    ]
    return text_paths


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return ""


def test_no_blocked_terms_in_tracked_files() -> None:
    offenders: list[tuple[str, str]] = []
    for path in _tracked_text_files():
        text = _read_text(path)
        if not text:
            continue
        for term in BLOCKED_TERMS:
            if term.lower() in text.lower():
                offenders.append((str(path), term))
    assert not offenders, (
        "Blocked publication terms found in tracked files: "
        + json.dumps(offenders, indent=2)
    )


def test_no_unwanted_paths_tracked() -> None:
    completed = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        cwd=str(REPO_ROOT),
        text=True,
    )
    bad = [
        line
        for line in completed.stdout.splitlines()
        if (
            line.endswith(".DS_Store")
            or ".ipynb_checkpoints" in line
            or line.endswith("Airline_Delay_Cause.csv")
            or line.endswith(".docx")
        )
    ]
    assert not bad, f"Unwanted tracked paths present: {bad}"


def test_metadata_files_are_valid_json() -> None:
    candidates = [
        REPO_ROOT / "models" / "model_metadata.json",
        REPO_ROOT / "data" / "dashboard_metadata.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            json.loads(candidate.read_text())


def test_notebook_is_valid_json() -> None:
    notebook = REPO_ROOT / "notebooks" / "flight_delay_analysis.ipynb"
    if notebook.exists():
        json.loads(notebook.read_text())