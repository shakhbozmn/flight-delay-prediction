"""Generate the deterministic dashboard dataset for deployment."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DASHBOARD_START_YEAR: int = 2020
DASHBOARD_END_YEAR: int = 2025
DASHBOARD_END_MONTH: int = 7

DASHBOARD_COLUMNS: tuple[str, ...] = (
    "year",
    "month",
    "carrier",
    "carrier_name",
    "airport",
    "airport_name",
    "arr_flights",
    "arr_del15",
    "arr_cancelled",
    "arr_diverted",
    "arr_delay",
    "carrier_delay",
    "weather_delay",
    "nas_delay",
    "security_delay",
    "late_aircraft_delay",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/Airline_Delay_Cause.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/dashboard_data.csv.gz"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("data/dashboard_metadata.json"),
    )
    parser.add_argument(
        "--dataset-metadata",
        type=Path,
        default=Path("data/dataset_metadata.json"),
        help="Source metadata JSON produced by scripts/download_data.py",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    args = _parse_args()

    if not args.input.exists():
        print(f"input file not found: {args.input}")
        return 1

    frame = pd.read_csv(args.input, low_memory=False)
    before_rows = len(frame)

    mask = frame["year"].between(DASHBOARD_START_YEAR, DASHBOARD_END_YEAR)
    filtered = frame.loc[mask, list(DASHBOARD_COLUMNS)].copy()
    drift_mask = filtered["year"].eq(DASHBOARD_END_YEAR) & (
        filtered["month"] > DASHBOARD_END_MONTH
    )
    filtered = filtered.loc[~drift_mask]
    filtered.sort_values(
        ["year", "month", "carrier", "airport"], inplace=True
    )
    filtered.reset_index(drop=True, inplace=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_payload = filtered.to_csv(index=False).encode("utf-8")
    with gzip.open(args.output, "wb") as handle:
        handle.write(csv_payload)

    output_sha = _sha256(args.output)

    source_metadata: dict = {}
    if args.dataset_metadata.exists():
        source_metadata = json.loads(args.dataset_metadata.read_text())

    metadata = {
        "schema_version": 1,
        "source_metadata_path": str(args.dataset_metadata),
        "source_sha256": source_metadata.get("sha256"),
        "dashboard_columns": list(DASHBOARD_COLUMNS),
        "row_count": int(len(filtered)),
        "source_row_count": int(before_rows),
        "year_window": {
            "start_year": DASHBOARD_START_YEAR,
            "end_year": DASHBOARD_END_YEAR,
            "end_month": DASHBOARD_END_MONTH,
        },
        "output_path": str(args.output),
        "output_sha256": output_sha,
        "output_size_bytes": args.output.stat().st_size,
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    print(
        "Dashboard dataset built. "
        f"rows={len(filtered)} from source_rows={before_rows} "
        f"sha256={output_sha}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())