"""Tests for the semantic layer: profile loading, prompt rendering, the
``describe_dataset`` description builder, and LLMPlanner profile injection."""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_data.backends import SQLiteSource
from mcp_data.client.planner import LLMPlanner, ToolCall
from mcp_data.config import Settings
from mcp_data.semantics import (
    SemanticProfile,
    load_profile,
    profile_path,
    render_profile_prompt,
)
from mcp_data.server.app import _build_dataset_description

SAMPLE_YAML = """\
dataset: demo
backend: sqlite
description: A tiny demo dataset.
vocabulary:
  countries:
    Italy: ITA
    Germany: DEU
  tenors:
    "5Y": Y005p0
conventions:
  date: "TEXT YYYY-MM-DD"
tables:
  zero_rates:
    description: Zero rates per source and date.
    columns:
      date: {type: TEXT, description: "Observation date."}
      source: {type: TEXT, description: "Country code."}
      Y005p0: {type: REAL, description: "5-year zero rate."}
examples:
  - q: "5Y rate for Italy"
    sql: "SELECT Y005p0 FROM zero_rates WHERE source='ITA'"
"""


@pytest.fixture()
def semantics_dir(tmp_path: Path) -> Path:
    (tmp_path / "demo.yaml").write_text(SAMPLE_YAML, encoding="utf-8")
    return tmp_path


def test_load_profile_returns_none_when_missing(tmp_path: Path) -> None:
    assert load_profile("nope", tmp_path) is None


def test_profile_path(tmp_path: Path) -> None:
    assert profile_path("demo", tmp_path) == tmp_path / "demo.yaml"


def test_load_profile_parses_structure(semantics_dir: Path) -> None:
    profile = load_profile("demo", semantics_dir)
    assert isinstance(profile, SemanticProfile)
    assert profile.dataset == "demo"
    assert profile.backend == "sqlite"
    assert profile.vocabulary["countries"]["Italy"] == "ITA"
    assert profile.vocabulary["tenors"]["5Y"] == "Y005p0"

    assert len(profile.tables) == 1
    table = profile.tables[0]
    assert table.name == "zero_rates"
    col_names = [c.name for c in table.columns]
    assert col_names == ["date", "source", "Y005p0"]

    assert len(profile.examples) == 1
    assert profile.examples[0].sql.startswith("SELECT Y005p0")


def test_render_profile_prompt_contains_key_info(semantics_dir: Path) -> None:
    profile = load_profile("demo", semantics_dir)
    prompt = render_profile_prompt(profile)
    assert "Dataset: demo" in prompt
    assert "Italy -> ITA" in prompt
    assert "5Y -> Y005p0" in prompt
    assert "zero_rates" in prompt
    assert "Y005p0" in prompt
    assert "SELECT Y005p0 FROM zero_rates" in prompt


def test_to_dict_roundtrips(semantics_dir: Path) -> None:
    profile = load_profile("demo", semantics_dir)
    rebuilt = SemanticProfile.from_dict(profile.to_dict())
    assert rebuilt == profile


def test_load_profile_rejects_non_mapping(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_profile("bad", tmp_path)


# ---------------------------------------------------------------------------
# describe_dataset description builder
# ---------------------------------------------------------------------------


def test_describe_dataset_uses_curated_profile(
    backend: SQLiteSource, semantics_dir: Path
) -> None:
    settings = Settings(dataset="demo", semantics_dir=semantics_dir)
    desc = _build_dataset_description(backend, settings)
    assert desc["has_curated_profile"] is True
    assert desc["profile"]["dataset"] == "demo"
    assert "Italy -> ITA" in desc["prompt"]


def test_describe_dataset_falls_back_to_live_schema(
    backend: SQLiteSource, tmp_path: Path
) -> None:
    # No YAML for this dataset -> synthesised from the live schema.
    settings = Settings(dataset="missing", semantics_dir=tmp_path)
    desc = _build_dataset_description(backend, settings)
    assert desc["has_curated_profile"] is False
    table_names = [t["name"] for t in desc["profile"]["tables"]]
    assert "customers" in table_names
    assert "orders" in table_names


# ---------------------------------------------------------------------------
# LLMPlanner profile injection
# ---------------------------------------------------------------------------


class _FakeBound:
    def __init__(self, recorder: dict) -> None:
        self._recorder = recorder

    def invoke(self, messages):
        # Record the system prompt text for assertions.
        self._recorder["system"] = messages[0].content

        class _Resp:
            tool_calls = [{"name": "run_sql", "args": {"sql": "SELECT 1"}}]

        return _Resp()


class _FakeModel:
    def __init__(self) -> None:
        self.recorder: dict = {}

    def bind_tools(self, tools, tool_choice=None):
        return _FakeBound(self.recorder)


def test_llm_planner_injects_profile_prompt() -> None:
    model = _FakeModel()
    planner = LLMPlanner(
        model=model,
        profile_prompt="# Dataset: demo\nItaly -> ITA\n5Y -> Y005p0",
    )
    calls = planner.plan("5Y rate for Italy", ["run_sql"])
    assert calls == [ToolCall("run_sql", {"sql": "SELECT 1"})]
    assert "Italy -> ITA" in model.recorder["system"]
    assert "5Y -> Y005p0" in model.recorder["system"]


def test_llm_planner_without_profile_uses_base_prompt() -> None:
    model = _FakeModel()
    planner = LLMPlanner(model=model)
    planner.plan("anything", ["run_sql"])
    assert "Dataset:" not in model.recorder["system"]
    assert "database assistant" in model.recorder["system"]
