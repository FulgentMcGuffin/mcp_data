"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_sqlite.backends import SQLiteBackend
from mcp_sqlite.data.seed import seed_database


@pytest.fixture()
def seeded_db(tmp_path: Path) -> Path:
    return seed_database(tmp_path / "test.db")


@pytest.fixture()
def backend(seeded_db: Path) -> SQLiteBackend:
    be = SQLiteBackend(seeded_db)
    yield be
    be.close()
