"""Per-data-source semantic layer.

A *semantic profile* is a declarative data dictionary for a single dataset: it
describes what each table and column means, a controlled vocabulary that maps
business terms to column names / stored values (e.g. ``Italy`` -> ``ITA``,
``5Y`` -> ``Y005p0``), conventions (date formats, etc.), and a handful of
curated natural-language -> SQL examples.

The profile is co-located with the data source and exposed by the MCP server
(see ``describe_dataset``), so any client can fetch it and inject it into an
LLM's context to translate domain natural language into correct SQL.

Profiles live as YAML files named ``<dataset>.yaml`` inside a semantics
directory (default ``<project root>/semantics``). When no profile exists for a
dataset, the LLM simply falls back to live schema introspection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ColumnSemantics:
    """Meaning of a single column."""

    name: str
    type: str = ""
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "type": self.type, "description": self.description}


@dataclass(frozen=True)
class TableSemantics:
    """Meaning of a single table and its columns."""

    name: str
    description: str = ""
    columns: list[ColumnSemantics] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "columns": [c.to_dict() for c in self.columns],
        }


@dataclass(frozen=True)
class QueryExample:
    """A curated natural-language question paired with its SQL answer."""

    q: str
    sql: str

    def to_dict(self) -> dict[str, Any]:
        return {"q": self.q, "sql": self.sql}


@dataclass(frozen=True)
class SemanticProfile:
    """Declarative business-logic description for one dataset."""

    dataset: str
    backend: str = "sqlite"
    description: str = ""
    vocabulary: dict[str, Any] = field(default_factory=dict)
    conventions: dict[str, Any] = field(default_factory=dict)
    tables: list[TableSemantics] = field(default_factory=list)
    examples: list[QueryExample] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "backend": self.backend,
            "description": self.description,
            "vocabulary": self.vocabulary,
            "conventions": self.conventions,
            "tables": [t.to_dict() for t in self.tables],
            "examples": [e.to_dict() for e in self.examples],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticProfile":
        tables: list[TableSemantics] = []
        for name, tbl in _iter_named(data.get("tables")):
            tbl = tbl or {}
            columns = [
                ColumnSemantics(
                    name=col_name,
                    type=str((col or {}).get("type", "")),
                    description=str((col or {}).get("description", "")),
                )
                for col_name, col in _iter_named(tbl.get("columns"))
            ]
            tables.append(
                TableSemantics(
                    name=name,
                    description=str(tbl.get("description", "")),
                    columns=columns,
                )
            )

        examples = [
            QueryExample(q=str(ex.get("q", "")), sql=str(ex.get("sql", "")))
            for ex in (data.get("examples") or [])
        ]

        return cls(
            dataset=str(data.get("dataset", "")),
            backend=str(data.get("backend", "sqlite")),
            description=str(data.get("description", "")),
            vocabulary=dict(data.get("vocabulary") or {}),
            conventions=dict(data.get("conventions") or {}),
            tables=tables,
            examples=examples,
        )


def _iter_named(value: Any) -> list[tuple[str, dict[str, Any]]]:
    """Yield ``(name, mapping)`` pairs from either YAML mapping form or the
    ``to_dict`` list form (``[{"name": ..., ...}, ...]``).

    This lets profiles authored as YAML mappings *and* serialized dicts both
    round-trip through :meth:`SemanticProfile.from_dict`.
    """
    if not value:
        return []
    if isinstance(value, dict):
        return [(str(k), v or {}) for k, v in value.items()]
    pairs: list[tuple[str, dict[str, Any]]] = []
    for item in value:
        item = item or {}
        name = str(item.get("name", ""))
        rest = {k: v for k, v in item.items() if k != "name"}
        pairs.append((name, rest))
    return pairs


def profile_path(dataset: str, semantics_dir: str | Path) -> Path:
    """Return the expected YAML path for ``dataset`` within ``semantics_dir``."""
    return Path(semantics_dir) / f"{dataset}.yaml"


def load_profile(dataset: str, semantics_dir: str | Path) -> SemanticProfile | None:
    """Load the semantic profile for ``dataset``.

    Returns ``None`` if no profile file exists, so callers can transparently
    fall back to live schema introspection.
    """
    path = profile_path(dataset, semantics_dir)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Semantic profile {path} must be a YAML mapping.")
    data.setdefault("dataset", dataset)
    return SemanticProfile.from_dict(data)


def render_profile_prompt(profile: SemanticProfile) -> str:
    """Render a compact, LLM-friendly text block describing the dataset.

    Designed to be prepended to a system prompt so the model understands table
    and column meaning, the controlled vocabulary, and example queries.
    """
    lines: list[str] = []
    lines.append(f"# Dataset: {profile.dataset} (backend: {profile.backend})")
    if profile.description:
        lines.append(profile.description)

    if profile.vocabulary:
        lines.append("\n## Vocabulary (map business terms to stored values/columns)")
        for group, mapping in profile.vocabulary.items():
            if isinstance(mapping, dict):
                pairs = ", ".join(f"{k} -> {v}" for k, v in mapping.items())
                lines.append(f"- {group}: {pairs}")
            else:
                lines.append(f"- {group}: {mapping}")

    if profile.conventions:
        lines.append("\n## Conventions")
        for key, value in profile.conventions.items():
            lines.append(f"- {key}: {value}")

    if profile.tables:
        lines.append("\n## Tables")
        for table in profile.tables:
            header = f"### {table.name}"
            if table.description:
                header += f" — {table.description}"
            lines.append(header)
            for col in table.columns:
                type_part = f" ({col.type})" if col.type else ""
                desc_part = f": {col.description}" if col.description else ""
                lines.append(f"  - {col.name}{type_part}{desc_part}")

    if profile.examples:
        lines.append("\n## Example questions and SQL")
        for ex in profile.examples:
            lines.append(f"- Q: {ex.q}")
            lines.append(f"  SQL: {ex.sql}")

    return "\n".join(lines)


__all__ = [
    "ColumnSemantics",
    "TableSemantics",
    "QueryExample",
    "SemanticProfile",
    "profile_path",
    "load_profile",
    "render_profile_prompt",
]
