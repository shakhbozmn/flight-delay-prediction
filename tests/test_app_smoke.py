"""Headless smoke test for the Streamlit entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_app_starts_without_exception() -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(REPO_ROOT / "src" / "app.py"))
    app.run(timeout=30)
    assert not app.exception, f"Streamlit app raised: {app.exception}"