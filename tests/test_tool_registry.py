"""Tests for ToolSpec / ToolRegistry (app/tools/registry.py)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from app.tools.registry import ToolRegistry, ToolSpec, tool_registry


class _DummyArgs(BaseModel):
    user_id: str = Field(default="u1")
    foo: int = 0


class TestToolSpecRendering:
    def test_signature_uses_pydantic_schema(self) -> None:
        spec = tool_registry.get("get_activities")
        sig = spec.signature()
        assert sig.startswith("get_activities(")
        assert "date_from: YYYY-MM-DD" in sig
        assert "date_to: YYYY-MM-DD" in sig
        # user_id is hidden — pipeline fills it
        assert "user_id" not in sig

    def test_to_openai_tool_strips_hidden_fields(self) -> None:
        spec = tool_registry.get("get_activities")
        tool = spec.to_openai_tool()
        assert tool["type"] == "function"
        fn = tool["function"]
        assert fn["name"] == "get_activities"
        assert "description" in fn
        params = fn["parameters"]
        assert params["type"] == "object"
        # user_id and tool discriminator must not leak into the LLM-visible schema
        assert "user_id" not in params["properties"]
        assert "tool" not in params["properties"]
        # required list (if present) does not mention hidden fields
        assert "user_id" not in params.get("required", [])

    def test_to_openai_tool_keeps_visible_fields(self) -> None:
        spec = tool_registry.get("get_activities")
        params = spec.to_openai_tool()["function"]["parameters"]
        assert "date_from" in params["properties"]
        assert "date_to" in params["properties"]


class TestRegistryDefaults:
    def test_default_registry_has_planner_tools(self) -> None:
        names = set(tool_registry.names())
        assert {
            "get_activities",
            "get_daily_facts",
            "get_user_profile",
            "compute_recovery",
            "check_overtraining",
            "rag_retrieve",
        }.issubset(names)

    def test_always_include_set_for_cheap_tools(self) -> None:
        assert tool_registry.get("get_user_profile").always_include is True
        assert tool_registry.get("rag_retrieve").always_include is True
        # Heavy data tools are not pinned
        assert tool_registry.get("get_activities").always_include is False


class TestRegistryRegistration:
    def test_register_and_get(self) -> None:
        reg = ToolRegistry()
        spec = ToolSpec(
            name="dummy",
            description="dummy tool",
            args_model=_DummyArgs,
        )
        reg.register(spec)
        assert "dummy" in reg
        assert reg.get("dummy") is spec
        assert reg.names() == ["dummy"]
        assert reg.all() == [spec]

    def test_duplicate_registration_raises(self) -> None:
        reg = ToolRegistry()
        reg.register(ToolSpec(name="dummy", description="x", args_model=_DummyArgs))
        with pytest.raises(ValueError):
            reg.register(ToolSpec(name="dummy", description="y", args_model=_DummyArgs))
