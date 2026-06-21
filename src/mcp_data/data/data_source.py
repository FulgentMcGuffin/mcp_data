import os, re, sys
from abc import ABC, abstractmethod

import polars as pl
import numpy as np
import sqlite3
from tqdm.auto import tqdm
from pathlib import Path
import asyncio

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

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


class DataSource(ABC):
    """Abstract interface for interacting with an underlying data store."""

    def create_table_from_polars(
        self,
        table_name: str,
        df: pl.DataFrame,
        overwrite_if_exists: bool = False,
        num_splits: int = 20
    ) -> bool:
        """Create a table from a Polars DataFrame and populate it with its rows.

        The column schema is derived from the DataFrame's dtypes:
        integer/boolean columns become ``INTEGER``, float columns become
        ``REAL``, and everything else becomes ``TEXT``.

        Data is inserted only when the table is actually created.  If the
        table already exists and *overwrite_if_exists* is ``False``, this
        method is a no-op and returns ``False``.

        Args:
            table_name: Name of the table to create.
            df: Polars DataFrame whose schema and rows are used.
            overwrite_if_exists: When True, drop and recreate the table if it
                already exists.

        Returns:
            ``True`` if the table was created (and data inserted), ``False``
            if the table already existed and was left unchanged.
        """
        schema = {col: _polars_dtype_to_sql(dtype) for col, dtype in df.schema.items()}
        created = self.create_table(table_name, schema, overwrite_if_exists)
        if created and len(df) > 0:
            if df.height > 1E6:
                df_splits = [
                    df.slice(idx[0], len(idx)) # second argumentis the length of the slice, NOT the index
                    for idx in np.array_split(np.arange(df.height), num_splits)
                    if len(idx) > 0
                ]
                for df_split in tqdm(df_splits, desc="Inserting data into SQLite"):
                    self.insert(table_name, df_split.to_dicts())
            else:
                self.insert(table_name, df.to_dicts())
        return created

    @abstractmethod
    def create_table(
        self,
        table_name: str,
        schema: dict[str, str],
        overwrite_if_exists: bool = False,
    ) -> bool:
        """Create a table with the given schema.

        Args:
            table_name: Name of the table to create.
            schema: Mapping of column name to its full type/constraint definition,
                e.g. ``{"id": "INTEGER PRIMARY KEY AUTOINCREMENT", "name": "TEXT NOT NULL"}``.
            overwrite_if_exists: When True, drop and recreate the table if it
                already exists.

        Returns:
            ``True`` if the table was created, ``False`` if it already existed
            and *overwrite_if_exists* was ``False``.
        """

    @abstractmethod
    def select(
        self,
        table_name: str,
        columns: list[str] | None = None,
        where: dict | None = None,
    ) -> list[dict]:
        """Fetch rows from *table_name*.

        Args:
            table_name: Table to query.
            columns: Columns to return; ``None`` returns all columns.
            where: Equality conditions applied as ``col = value`` joined by ``AND``.

        Returns:
            List of rows, each represented as a ``{column: value}`` dict.
        """

    @abstractmethod
    def insert(
        self,
        table_name: str,
        data: dict | list[dict],
    ) -> int:
        """Insert one or more rows into *table_name*.

        Args:
            table_name: Target table.
            data: A single row dict or a list of row dicts.

        Returns:
            Number of rows inserted.
        """

    @abstractmethod
    def update(
        self,
        table_name: str,
        data: dict,
        where: dict,
    ) -> int:
        """Update rows in *table_name* that match *where*.

        Args:
            table_name: Target table.
            data: Columns and their new values.
            where: Equality conditions that identify rows to update.

        Returns:
            Number of rows affected.
        """

    @abstractmethod
    def delete(
        self,
        table_name: str,
        where: dict,
    ) -> int:
        """Delete rows from *table_name* that match *where*.

        Args:
            table_name: Target table.
            where: Equality conditions that identify rows to delete.

        Returns:
            Number of rows deleted.
        """

    @abstractmethod
    def execute(
        self,
        query: str,
        params: tuple = (),
    ) -> list[dict]:
        """Execute an arbitrary SQL statement.

        Args:
            query: Raw SQL string, using ``?`` placeholders for parameters.
            params: Positional parameter values bound to the placeholders.

        Returns:
            For SELECT-like statements, a list of row dicts; otherwise an empty list.
        """

class SQLiteSource(DataSource):
    """DataSource implementation backed by a SQLite database.

    Supports use as a context manager to automatically open and close the
    underlying connection::

        with SQLiteSource("company.db") as db:
            db.create_table("employees", schema)
            db.insert("employees", {"name": "Alice", "role": "engineer", "salary": 90000.0})
            rows = db.select("employees", where={"role": "engineer"})
    """

    DEFAULT_DB_DIR = os.getenv('MCP_DB_PATH', None)
    if DEFAULT_DB_DIR is None:
        raise ValueError("MCP_DB_PATH environment variable is not set")
    DEFAULT_DB_NAME = os.path.basename(DEFAULT_DB_DIR) # "input_data.db"
    DEFAULT_DB_DIR = os.path.dirname(DEFAULT_DB_DIR)
    if not DEFAULT_DB_NAME.endswith(".db"):
        raise ValueError(f"MCP_DB_PATH environment variable must end with .db but found {DEFAULT_DB_NAME}")

    @classmethod
    def get_full_db_path(cls, db_name: str = DEFAULT_DB_NAME) -> str:
        return os.path.join(
            cls.DEFAULT_DB_DIR,
            f"{db_name}.db" if not db_name.endswith(".db") else db_name,
        )

    def __init__(self, db_path: str | None = None) -> None:
        """
        Args:
            db_path: Path to the SQLite database file.
                Use ``":memory:"`` for a temporary in-memory database.
        """
        if db_path is None:
            db_path = os.path.join(self.DEFAULT_DB_DIR, self.DEFAULT_DB_NAME)
        self._db_path = db_path
        self._connection: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open the database connection, creating the database file and any missing parent directories if needed."""
        if self._db_path != ":memory:":
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._connection = sqlite3.connect(self._db_path)
        self._connection.row_factory = sqlite3.Row

    def close(self) -> None:
        """Close the database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> "SQLiteSource":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _conn(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError(
                "No active connection. Call connect() or use SQLiteSource as a context manager."
            )
        return self._connection

    # ------------------------------------------------------------------
    # DataSource interface
    # ------------------------------------------------------------------

    def create_table(
        self,
        table_name: str,
        schema: dict[str, str],
        overwrite_if_exists: bool = False,
    ) -> bool:
        columns_sql = ",\n    ".join(
            f"{col} {definition}" for col, definition in schema.items()
        )
        conn = self._conn()
        already_exists = (
            conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            ).fetchone()
            is not None
        )

        if already_exists and not overwrite_if_exists:
            return False

        if overwrite_if_exists:
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")

        conn.execute(f"CREATE TABLE {table_name} (\n    {columns_sql}\n);")
        conn.commit()
        return True

    def select(
        self,
        table_name: str,
        columns: list[str] | None = None,
        where: dict | None = None,
    ) -> list[dict]:
        cols_sql = ", ".join(columns) if columns else "*"
        query = f"SELECT {cols_sql} FROM {table_name}"
        params: list = []
        if where:
            conditions = " AND ".join(f"{col} = ?" for col in where)
            query += f" WHERE {conditions}"
            params = list(where.values())
        cursor = self._conn().execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def insert(
        self,
        table_name: str,
        data: dict | list[dict],
    ) -> int:
        if isinstance(data, dict):
            data = [data]
        if not data:
            return 0
        columns_sql = ", ".join(data[0].keys())
        placeholders = ", ".join("?" for _ in data[0])
        query = f"INSERT INTO {table_name} ({columns_sql}) VALUES ({placeholders})"
        self._conn().executemany(query, [list(row.values()) for row in data])
        self._conn().commit()
        return len(data)

    def update(
        self,
        table_name: str,
        data: dict,
        where: dict,
    ) -> int:
        set_clause = ", ".join(f"{col} = ?" for col in data)
        where_clause = " AND ".join(f"{col} = ?" for col in where)
        query = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"
        params = list(data.values()) + list(where.values())
        cursor = self._conn().execute(query, params)
        self._conn().commit()
        return cursor.rowcount

    def delete(
        self,
        table_name: str,
        where: dict,
    ) -> int:
        where_clause = " AND ".join(f"{col} = ?" for col in where)
        query = f"DELETE FROM {table_name} WHERE {where_clause}"
        cursor = self._conn().execute(query, list(where.values()))
        self._conn().commit()
        return cursor.rowcount

    def execute(
        self,
        query: str,
        params: tuple = (),
    ) -> list[dict]:
        cursor = self._conn().execute(query, params)
        self._conn().commit()
        if cursor.description:
            return [dict(row) for row in cursor.fetchall()]
        return []

# if __name__ == "__main__":
#     employees_schema = {
#         "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
#         "name": "TEXT NOT NULL",
#         "role": "TEXT NOT NULL",
#         "salary": "REAL",
#         "hire_date": "TEXT DEFAULT CURRENT_TIMESTAMP",
#     }

#     with SQLiteSource("company.db") as db:
#         db.create_table("employees", employees_schema)
#         print("Table 'employees' created successfully!")
