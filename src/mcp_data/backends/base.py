"""Generic, storage-agnostic database backend contract.

Everything above this layer (the Hamilton pipeline and the MCP tools) speaks
only to :class:`DataBackend`. Adding a new storage engine (e.g. a Redis cache)
is therefore just a matter of writing another class that satisfies this
protocol; no server or client code needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import polars as pl
from tqdm.auto import tqdm


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


def _polars_dtype_to_sql(dtype: pl.DataType) -> str:
    """Map a Polars dtype to a generic SQL type (INTEGER, REAL, or TEXT)."""
    name = type(dtype).__name__
    if name in {
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "Boolean",
    }:
        return "INTEGER"
    if name in {"Float32", "Float64"}:
        return "REAL"
    return "TEXT"


@runtime_checkable
class DataBackend(Protocol):
    """Storage-agnostic *read* interface used by the server tools.

    This is the seam that keeps the project engine-agnostic: a new store (e.g. a
    Redis cache) only needs to satisfy this protocol for the MCP server and query
    pipeline to use it -- no write capabilities required.
    """

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


class DataSink(ABC):
    """Write-side contract for creating tables and ingesting/mutating rows.

    Kept deliberately separate from :class:`DataBackend` (the read side) so that
    a read-only engine, or one whose write model does not map onto relational
    DDL/DML (e.g. Redis), is not forced to implement operations it cannot
    support. A concrete store typically implements both interfaces.

    ``create_table_from_polars`` is provided as a template method built on the
    abstract primitives, so every implementation gets DataFrame ingestion for
    free once it implements ``create_table`` and ``insert``.
    """

    @abstractmethod
    def create_table(
        self,
        table_name: str,
        schema: dict[str, str],
        overwrite_if_exists: bool = False,
    ) -> bool:
        """Create a table with the given ``{column: type/constraint}`` schema.

        Returns ``True`` if created, ``False`` if it already existed and
        *overwrite_if_exists* was ``False``.
        """

    @abstractmethod
    def select(
        self,
        table_name: str,
        columns: list[str] | None = None,
        where: dict | None = None,
    ) -> list[dict]:
        """Fetch rows as ``{column: value}`` dicts (``columns=None`` -> all)."""

    @abstractmethod
    def insert(self, table_name: str, data: dict | list[dict]) -> int:
        """Insert one or more rows. Returns the number of rows inserted."""

    @abstractmethod
    def update(self, table_name: str, data: dict, where: dict) -> int:
        """Update rows matching *where*. Returns rows affected."""

    @abstractmethod
    def delete(self, table_name: str, where: dict) -> int:
        """Delete rows matching *where*. Returns rows deleted."""

    @abstractmethod
    def execute(self, query: str, params: tuple = ()) -> list[dict]:
        """Execute an arbitrary statement; SELECT-like ones return row dicts."""

    def create_table_from_polars(
        self,
        table_name: str,
        df: pl.DataFrame,
        overwrite_if_exists: bool = False,
        num_splits: int = 20,
    ) -> bool:
        """Create a table from a Polars DataFrame and populate it with its rows.

        The schema is derived from the DataFrame's dtypes (integer/boolean ->
        ``INTEGER``, float -> ``REAL``, everything else -> ``TEXT``). Data is
        inserted only when the table is actually created; if it already exists
        and *overwrite_if_exists* is ``False`` this is a no-op returning
        ``False``.
        """
        schema = {col: _polars_dtype_to_sql(dtype) for col, dtype in df.schema.items()}
        created = self.create_table(table_name, schema, overwrite_if_exists)
        if created and len(df) > 0:
            if df.height > 1e6:
                df_splits = [
                    df.slice(idx[0], len(idx))  # second arg is the slice length, NOT the end index
                    for idx in np.array_split(np.arange(df.height), num_splits)
                    if len(idx) > 0
                ]
                for df_split in tqdm(df_splits, desc="Inserting data into SQLite"):
                    self.insert(table_name, df_split.to_dicts())
            else:
                self.insert(table_name, df.to_dicts())
        return created
