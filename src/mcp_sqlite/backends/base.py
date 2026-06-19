"""Generic, storage-agnostic database backend contract.

Everything above this layer (the Hamilton pipeline and the MCP tools) speaks
only to :class:`DataBackend`. Adding a new storage engine (e.g. a Redis cache)
is therefore just a matter of writing another class that satisfies this
protocol; no server or client code needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import polars as pl


class QueryError(RuntimeError):
    """Raised when a query is rejected or fails to execute."""


@dataclass(frozen=True)
class ColumnInfo:
    """A single column in a table schema."""

    name: str
    type: str
    nullable: bool = True
    primary_key: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "nullable": self.nullable,
            "primary_key": self.primary_key,
        }


@dataclass(frozen=True)
class TableSchema:
    """Schema description for one table/relation."""

    table: str
    columns: list[ColumnInfo] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "columns": [c.to_dict() for c in self.columns],
        }


# Statement leading keywords that only read data. Anything else is rejected so a
# backend opened in read-only mode never even receives a mutating statement.
_READ_ONLY_PREFIXES = ("select", "with", "pragma", "explain")


def is_read_only_sql(sql: str) -> bool:
    """Best-effort check that ``sql`` is a single read-only statement.

    Guards against mutations (INSERT/UPDATE/DELETE/DROP/...) and against
    stacking multiple statements separated by ``;``.
    """

    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        return False
    # Disallow multiple statements (e.g. "select 1; drop table t").
    if ";" in stripped:
        return False
    first_word = stripped.split(None, 1)[0].lower()
    return first_word in _READ_ONLY_PREFIXES


@runtime_checkable
class DataBackend(Protocol):
    """Storage-agnostic read interface used by the server tools."""

    @property
    def name(self) -> str:
        """Human-readable backend identifier (e.g. ``"sqlite"``)."""
        ...

    def list_tables(self) -> list[str]:
        """Return the names of queryable tables/relations."""
        ...

    def get_schema(self, table: str) -> TableSchema:
        """Return the column schema for ``table``."""
        ...

    def run_query(self, sql: str) -> pl.DataFrame:
        """Execute a read-only query and return the result as a polars frame."""
        ...

    def close(self) -> None:
        """Release any underlying resources."""
        ...
