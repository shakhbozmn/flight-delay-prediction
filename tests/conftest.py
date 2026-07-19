"""Compact deterministic fixtures for unit tests."""

from __future__ import annotations

import pandas as pd
import pytest


def _row(
    *,
    year: int,
    month: int,
    carrier: str,
    airport: str,
    arr_flights: float,
    arr_del15: float,
    arr_cancelled: float = 0.0,
    arr_diverted: float = 0.0,
    arr_delay: float = 0.0,
    carrier_delay: float = 0.0,
    weather_delay: float = 0.0,
    nas_delay: float = 0.0,
    security_delay: float = 0.0,
    late_aircraft_delay: float = 0.0,
) -> dict[str, object]:
    return {
        "year": year,
        "month": month,
        "carrier": carrier,
        "carrier_name": f"Carrier {carrier}",
        "airport": airport,
        "airport_name": f"Airport {airport}",
        "arr_flights": arr_flights,
        "arr_del15": arr_del15,
        "arr_cancelled": arr_cancelled,
        "arr_diverted": arr_diverted,
        "arr_delay": arr_delay,
        "carrier_delay": carrier_delay,
        "weather_delay": weather_delay,
        "nas_delay": nas_delay,
        "security_delay": security_delay,
        "late_aircraft_delay": late_aircraft_delay,
    }


@pytest.fixture
def bts_sample() -> pd.DataFrame:
    """Fixture with rows from 2022, 2023, 2024, and early 2025."""

    rows = [
        _row(
            year=2022,
            month=6,
            carrier="AA",
            airport="AAA",
            arr_flights=100,
            arr_del15=20,
        ),
        _row(
            year=2022,
            month=7,
            carrier="AA",
            airport="AAA",
            arr_flights=100,
            arr_del15=30,
        ),
        _row(
            year=2023,
            month=1,
            carrier="BB",
            airport="BBB",
            arr_flights=100,
            arr_del15=25,
        ),
        _row(
            year=2023,
            month=11,
            carrier="BB",
            airport="BBB",
            arr_flights=100,
            arr_del15=26,
        ),
        _row(
            year=2024,
            month=3,
            carrier="CC",
            airport="CCC",
            arr_flights=200,
            arr_del15=80,
        ),
        _row(
            year=2024,
            month=9,
            carrier="CC",
            airport="CCC",
            arr_flights=200,
            arr_del15=10,
        ),
        _row(
            year=2025,
            month=1,
            carrier="DD",
            airport="DDD",
            arr_flights=300,
            arr_del15=90,
        ),
        _row(
            year=2025,
            month=7,
            carrier="DD",
            airport="DDD",
            arr_flights=300,
            arr_del15=60,
        ),
    ]
    return pd.DataFrame(rows)