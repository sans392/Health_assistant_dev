"""Tests for ToolSelector (app/tools/selector.py)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.tools.registry import ToolRegistry, ToolSpec
from app.tools.selector import AllToolsSelector, KeywordToolSelector


class _Args(BaseModel):
    user_id: str = Field(default="u1")


def _make_registry() -> ToolRegistry:
    """Small isolated registry so tests are independent of the global one."""
    reg = ToolRegistry()
    reg.register(ToolSpec(
        name="get_activities", description="trainings", args_model=_Args,
        keywords=("тренировк", "пробежк", "run"),
    ))
    reg.register(ToolSpec(
        name="get_daily_facts", description="daily metrics", args_model=_Args,
        keywords=("сон", "пульс", "hrv", "sleep"),
    ))
    reg.register(ToolSpec(
        name="rag_retrieve", description="kb chunks", args_model=_Args,
        keywords=("почему", "норм", "why"),
        always_include=True,
    ))
    reg.register(ToolSpec(
        name="get_user_profile", description="profile", args_model=_Args,
        always_include=True,
    ))
    return reg


class TestAllToolsSelector:
    def test_returns_every_tool_in_registration_order(self) -> None:
        reg = _make_registry()
        sel = AllToolsSelector(reg)
        names = [s.name for s in sel.select("anything")]
        assert names == [
            "get_activities", "get_daily_facts", "rag_retrieve", "get_user_profile",
        ]

    def test_max_tools_truncates_from_the_end(self) -> None:
        reg = _make_registry()
        sel = AllToolsSelector(reg)
        names = [s.name for s in sel.select("anything", max_tools=2)]
        assert names == ["get_activities", "get_daily_facts"]


class TestKeywordToolSelector:
    def test_pinned_tools_come_first_then_keyword_matches(self) -> None:
        sel = KeywordToolSelector(_make_registry())
        names = [s.name for s in sel.select("какая у меня была пробежка вчера?")]
        # always_include tools first, in registration order
        assert names[0] == "rag_retrieve"
        assert names[1] == "get_user_profile"
        # then keyword-matched
        assert "get_activities" in names
        # unrelated tool dropped
        assert "get_daily_facts" not in names

    def test_higher_keyword_score_wins(self) -> None:
        sel = KeywordToolSelector(_make_registry())
        # query mentions HRV + сон → 2 hits for daily_facts, 0 for activities
        names = [s.name for s in sel.select("плохой сон и низкий HRV")]
        non_pinned = [n for n in names if n not in {"rag_retrieve", "get_user_profile"}]
        assert non_pinned == ["get_daily_facts"]

    def test_no_keyword_hits_returns_only_pinned(self) -> None:
        sel = KeywordToolSelector(_make_registry())
        # Query that matches nothing keyword-wise — pinned tools still come
        # through; data tools are dropped so the prompt stays small.
        names = [s.name for s in sel.select("absolutely unrelated banana")]
        assert names == ["rag_retrieve", "get_user_profile"]

    def test_no_matches_and_no_pinned_falls_back_to_all(self) -> None:
        # Registry without any always_include tools: empty match would leave
        # the planner with zero tools, so fallback returns the full set.
        reg = ToolRegistry()
        reg.register(ToolSpec(
            name="a", description="x", args_model=_Args, keywords=("foo",),
        ))
        reg.register(ToolSpec(
            name="b", description="x", args_model=_Args, keywords=("bar",),
        ))
        sel = KeywordToolSelector(reg)
        names = [s.name for s in sel.select("nothing matches here")]
        assert names == ["a", "b"]

    def test_max_tools_caps_result(self) -> None:
        sel = KeywordToolSelector(_make_registry())
        names = [s.name for s in sel.select("сон HRV пробежка", max_tools=2)]
        assert len(names) == 2
        # always_include comes first, so cap=2 means the two pinned tools
        assert names[0] == "rag_retrieve"
        assert names[1] == "get_user_profile"


