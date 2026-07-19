"""Validate and import a BTS snapshot downloaded manually from the official source.

The script never reaches out to the network on its own. The user must
download the snapshot from the BTS Airline On-Time Statistics and Delay Causes
page and provide its path via ``--source-file``. The script then validates the
file's checksum, row count, and required columns before copying it under
``data/raw/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml.data import (
    DRIFT_YEAR,
    REQUIRED_COLUMNS,
    validate_dataset,
)

EXPECTED_SHA256 = "383fb1ae404cc46aa9380bbc8156fdf6e2e4bd5af7ae1197717a639a92378134"
EXPECTED_ROW_COUNT = 409612
EXPECTED_MIN_YEAR = 2003
EXPECTED_MAX_YEAR = DRIFT_YEAR
EXPECTED_MAX_MONTH = 7

BTS_SOURCE_URL = (
    "https://www.transtats.bts.gov/ot_delay/ot_delaycause1.asp?pn=1"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-file",
        type=Path,
        help="Path to a manually downloaded BTS Airline_Delay_Cause.csv file.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/raw/Airline_Delay_Cause.csv"),
        help="Destination path for the validated snapshot.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("data/dataset_metadata.json"),
        help="Path to write the dataset metadata JSON.",
    )
    parser.add_argument(
        "--expected-sha256",
        default=EXPECTED_SHA256,
        help="Override the pinned SHA-256 value used for validation.",
    )
    parser.add_argument(
        "--expected-row-count",
        type=int,
        default=EXPECTED_ROW_COUNT,
        help="Override the pinned row count used for validation.",
    )
    return parser.parse_args()


def _print_manual_instructions() -> None:
    print(
        "Manual download required. Visit the BTS Airline On-Time Statistics and\n"
        f"Delay Causes page ({BTS_SOURCE_URL}) and download the raw data for\n"
        "the period June 2003 through July 2025 as CSV. Then rerun this\n"
        "command with --source-file <path>."
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_metadata(
    *,
    source_path: Path,
    destination_path: Path,
    sha256_value: str,
    row_count: int,
    earliest_year: int,
    latest_year: int,
    latest_month: int,
) -> dict:
    return {
        "source_url": BTS_SOURCE_URL,
        "local_source_path": str(source_path),
        "destination_path": str(destination_path),
        "schema_columns": list(REQUIRED_COLUMNS),
        "row_count": row_count,
        "earliest_year": earliest_year,
        "latest_year": latest_year,
        "latest_month": latest_month,
        "sha256": sha256_value,
        "license": (
            "U.S. Government work (17 U.S.C. §105); public-domain reuse "
            "per Bureau of Transportation Statistics policy."
        ),
    }


def main() -> int:
    args = _parse_args()

    if args.source_file is None:
        _print_manual_instructions()
        return 2

    source_path: Path = args.source_file.expanduser().resolve()
    if not source_path.exists():
        print(f"source file not found: {source_path}")
        return 3

    sha256_value = _sha256(source_path)
    if sha256_value != args.expected_sha256:
        print(
            "SHA-256 mismatch for "
            f"{source_path}: expected {args.expected_sha256}, got {sha256_value}"
        )
        return 4

    frame = pd.read_csv(source_path, low_memory=False)
    if len(frame) != args.expected_row_count:
        print(
            f"row count mismatch: expected {args.expected_row_count}, "
            f"got {len(frame)}"
        )
        return 5

    validate_dataset(frame)

    earliest_year = int(frame["year"].min())
    latest_year = int(frame["year"].max())
    latest_month = int(frame.loc[frame["year"].eq(latest_year), "month"].max())

    if earliest_year < EXPECTED_MIN_YEAR:
        print(f"earliest year {earliest_year} is before {EXPECTED_MIN_YEAR}")
        return 6
    if latest_year > EXPECTED_MAX_YEAR:
        print(f"latest year {latest_year} is after {EXPECTED_MAX_YEAR}")
        return 7
    if latest_year == EXPECTED_MAX_YEAR and latest_month > EXPECTED_MAX_MONTH:
        print(
            f"latest month in {latest_year} is {latest_month}, "
            f"exceeds {EXPECTED_MAX_MONTH}"
        )
        return 8

    destination_path: Path = args.destination.expanduser().resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)

    metadata = _build_metadata(
        source_path=source_path,
        destination_path=destination_path,
        sha256_value=sha256_value,
        row_count=len(frame),
        earliest_year=earliest_year,
        latest_year=latest_year,
        latest_month=latest_month,
    )
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    print(
        "Snapshot validated and imported. "
        f"sha256={sha256_value} rows={len(frame)} "
        f"years={earliest_year}-{latest_year}({latest_month})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())