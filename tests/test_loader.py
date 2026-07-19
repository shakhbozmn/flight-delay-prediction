"""Tests for the data-import and dashboard-build scripts."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
PYTHON = sys.executable


def _make_snapshot(tmp_path: Path, *, row_count: int = 2200) -> Path:
    rows = []
    carriers = [f"C{i}" for i in range(8)]
    airports = [f"A{i}" for i in range(12)]
    year = 2003
    month = 1
    for index in range(row_count):
        rows.append(
            {
                "year": year,
                "month": month,
                "carrier": carriers[index % len(carriers)],
                "carrier_name": f"Carrier {carriers[index % len(carriers)]}",
                "airport": airports[index % len(airports)],
                "airport_name": f"Airport {airports[index % len(airports)]}",
                "arr_flights": 100,
                "arr_del15": (index % 4) * 30,
                "arr_cancelled": 1,
                "arr_diverted": 1,
                "arr_delay": 50.0,
                "carrier_delay": 10.0,
                "weather_delay": 5.0,
                "nas_delay": 8.0,
                "security_delay": 0.0,
                "late_aircraft_delay": 6.0,
            }
        )
        month += 1
        if month > 12:
            month = 1
            year += 1
    df = pd.DataFrame(rows)
    df = df.loc[df["year"].le(2025)]
    df = df.loc[~((df["year"].eq(2025)) & (df["month"] > 7))].reset_index(drop=True)
    snapshot = tmp_path / "snapshot.csv"
    df.to_csv(snapshot, index=False)
    return snapshot


def _run(script: Path, *args: str, cwd: Path) -> subprocess.CompletedProcess:
    env_path = SCRIPTS_DIR.parent
    return subprocess.run(
        [PYTHON, str(script), *args],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env={"PYTHONPATH": str(env_path), "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )


def test_download_data_prints_manual_instructions(tmp_path: Path, capsys) -> None:
    completed = _run(
        SCRIPTS_DIR / "download_data.py",
        "--destination",
        str(tmp_path / "out.csv"),
        "--metadata-output",
        str(tmp_path / "metadata.json"),
        cwd=tmp_path,
    )
    assert completed.returncode == 2
    output = completed.stdout + completed.stderr
    assert "Manual download required" in output


def test_download_data_rejects_bad_checksum(tmp_path: Path) -> None:
    snapshot = _make_snapshot(tmp_path)
    completed = _run(
        SCRIPTS_DIR / "download_data.py",
        "--source-file",
        str(snapshot),
        "--destination",
        str(tmp_path / "out.csv"),
        "--metadata-output",
        str(tmp_path / "metadata.json"),
        "--expected-sha256",
        "0" * 64,
        cwd=tmp_path,
    )
    assert completed.returncode == 4


def test_download_data_validates_and_copies(tmp_path: Path) -> None:
    snapshot = _make_snapshot(tmp_path, row_count=2200)
    digest = hashlib.sha256()
    with snapshot.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    expected_sha = digest.hexdigest()
    expected_rows = int(pd.read_csv(snapshot).shape[0])
    completed = _run(
        SCRIPTS_DIR / "download_data.py",
        "--source-file",
        str(snapshot),
        "--destination",
        str(tmp_path / "out.csv"),
        "--metadata-output",
        str(tmp_path / "metadata.json"),
        "--expected-sha256",
        expected_sha,
        "--expected-row-count",
        str(expected_rows),
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (tmp_path / "out.csv").exists()
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["source_url"].startswith("https://www.transtats.bts.gov")
    assert metadata["row_count"] == expected_rows


def test_build_dashboard_data_reports_rows(tmp_path: Path) -> None:
    snapshot = _make_snapshot(tmp_path, row_count=2200)
    completed = _run(
        SCRIPTS_DIR / "build_dashboard_data.py",
        "--input",
        str(snapshot),
        "--output",
        str(tmp_path / "dashboard.csv.gz"),
        "--metadata-output",
        str(tmp_path / "dashboard_meta.json"),
        "--dataset-metadata",
        str(tmp_path / "ignored_metadata.json"),
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    with gzip.open(tmp_path / "dashboard.csv.gz", "rb") as handle:
        dashboard = pd.read_csv(handle)
    assert int(dashboard["year"].min()) >= 2020
    assert int(dashboard["year"].max()) <= 2025
    drift_months = dashboard.loc[dashboard["year"].eq(2025), "month"]
    if not drift_months.empty:
        assert int(drift_months.max()) <= 7
    metadata = json.loads((tmp_path / "dashboard_meta.json").read_text())
    assert metadata["row_count"] == len(dashboard)