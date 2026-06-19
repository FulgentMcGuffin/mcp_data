"""Create and seed a small example SQLite database.

Run via the ``db-mcp-seed`` console script (or ``python -m mcp_sqlite.data.seed``)
so the server has something to query out of the box. Re-running is idempotent:
the existing tables are dropped and recreated.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from mcp_sqlite.config import get_settings

CUSTOMERS = [
    (1, "Ada Lovelace", "ada@example.com", "UK"),
    (2, "Alan Turing", "alan@example.com", "UK"),
    (3, "Grace Hopper", "grace@example.com", "US"),
    (4, "Katherine Johnson", "katherine@example.com", "US"),
    (5, "Edsger Dijkstra", "edsger@example.com", "NL"),
]

# (id, customer_id, product, amount, created_at)
ORDERS = [
    (1, 1, "Analytical Engine", 1200.00, "2026-01-05"),
    (2, 1, "Punch Cards", 49.99, "2026-01-09"),
    (3, 2, "Turing Machine", 999.00, "2026-02-14"),
    (4, 3, "COBOL Manual", 25.50, "2026-02-20"),
    (5, 3, "Nanosecond Wire", 5.00, "2026-03-01"),
    (6, 4, "Orbital Calculator", 350.75, "2026-03-11"),
    (7, 5, "Goto Eraser", 12.00, "2026-03-15"),
    (8, 5, "Shortest Path Map", 75.25, "2026-04-02"),
]


def seed_database(db_path: Path) -> Path:
    """(Re)create the example schema and data at ``db_path``."""

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            DROP TABLE IF EXISTS orders;
            DROP TABLE IF EXISTS customers;

            CREATE TABLE customers (
                id      INTEGER PRIMARY KEY,
                name    TEXT    NOT NULL,
                email   TEXT    NOT NULL UNIQUE,
                country TEXT    NOT NULL
            );

            CREATE TABLE orders (
                id          INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL REFERENCES customers(id),
                product     TEXT    NOT NULL,
                amount      REAL    NOT NULL,
                created_at  TEXT    NOT NULL
            );
            """
        )
        conn.executemany(
            "INSERT INTO customers (id, name, email, country) VALUES (?, ?, ?, ?)",
            CUSTOMERS,
        )
        conn.executemany(
            "INSERT INTO orders (id, customer_id, product, amount, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ORDERS,
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Seed the example SQLite database.")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=settings.db_path,
        help=f"Target database file (default: {settings.db_path}).",
    )
    args = parser.parse_args()
    path = seed_database(args.db_path)
    print(f"Seeded example database at {path}")


if __name__ == "__main__":
    main()
