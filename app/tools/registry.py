"""Tool registry: single source of truth for planner tools.

Each `ToolSpec` carries everything needed to:
- render a textual signature for the current prompt-based planner;
- emit an OpenAI-style `tools` entry (when we move to native tool-calling);
- match the tool against a user query (keywords / future embeddings).

Adding a new tool: create the Pydantic args model in `schemas.py`, then call
`tool_registry.register(ToolSpec(...))` once at module import. Planner picks it
up automatically through the selector — no edits in `planner.py` required.

The registry is intentionally separate from `pipeline/tool_executor.py`. The
executor knows how to *run* tools; the registry knows how to *describe* them.
Mixing the two would force every renderer (admin UI, planner prompt, OpenAI
tool spec) to import the executor's runtime side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from app.tools.schemas import (
    CheckOvertrainingArgs,
    ComputeRecoveryArgs,
    GetActivitiesArgs,
    GetDailyFactsArgs,
    GetUserProfileArgs,
    RagRetrieveArgs,
    tool_to_prompt_signature,
)


# Fields the planner LLM never sets — pipeline fills them. Hidden from both the
# textual signature and the OpenAI JSON schema so the model is not tempted to
# pass them.
_HIDDEN_FIELDS: frozenset[str] = frozenset({"user_id", "tool"})


@dataclass(frozen=True)
class ToolSpec:
    """Description of a single planner-callable tool."""

    name: str
    description: str
    args_model: type[BaseModel]
    # Canonical example. Concrete dates/sport_types help small LLMs copy the
    # shape rather than infer it from the schema.
    example_args: dict[str, Any] = field(default_factory=dict)
    # Keywords (substring match, lowercased) used by the keyword selector.
    # Embeddings-based selector will ignore this and use `description`.
    keywords: tuple[str, ...] = field(default_factory=tuple)
    # If True, tool is included regardless of selector matching. Use for
    # cheap, almost-always-useful tools (e.g. rag_retrieve, get_user_profile).
    always_include: bool = False

    def signature(self) -> str:
        """Compact one-line signature: `name(arg1: type, arg2?: type, ...)`."""
        return tool_to_prompt_signature(self.name)

    def to_openai_tool(self) -> dict[str, Any]:
        """Render as OpenAI Chat Completions `tools` entry.

        Strips `_HIDDEN_FIELDS` from the JSON schema so the model does not see
        `user_id` / `tool` discriminators.
        """
        schema = self.args_model.model_json_schema()
        props = dict(schema.get("properties") or {})
        required = list(schema.get("required") or [])
        for hidden in _HIDDEN_FIELDS:
            props.pop(hidden, None)
            if hidden in required:
                required.remove(hidden)
        parameters: dict[str, Any] = {
            "type": "object",
            "properties": props,
        }
        if required:
            parameters["required"] = required
        if "$defs" in schema:
            parameters["$defs"] = schema["$defs"]
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }


class ToolRegistry:
    """In-memory registry of `ToolSpec` keyed by tool name."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Tool {spec.name!r} is already registered")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        return self._specs[name]

    def all(self) -> list[ToolSpec]:
        """All registered specs in registration order."""
        return list(self._specs.values())

    def names(self) -> list[str]:
        return list(self._specs.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._specs


# ---------------------------------------------------------------------------
# Default registry — populated at import time with the planner tools that
# already exist in `app/tools/`. Keep descriptions short; the canonical example
# does the heavy lifting for small-LLM imitation.
# ---------------------------------------------------------------------------

tool_registry = ToolRegistry()


# Description text goes verbatim into the planner system prompt, so it stays
# in Russian to match the rest of the prompt. Code comments / docstrings
# remain English per the project style guide.
tool_registry.register(
    ToolSpec(
        name="get_activities",
        description="тренировки пользователя за интервал дат",
        args_model=GetActivitiesArgs,
        example_args={
            "date_from": "2026-04-20",
            "date_to": "2026-04-26",
            "sport_type": "running",
            "min_distance_meters": 10000,
        },
        keywords=(
            "тренировк", "пробежк", "бега", "бег", "run", "ride", "swim",
            "велотрен", "плаван", "training", "workout", "activity", "sessions",
            "интервалк", "long run", "марафон",
        ),
    )
)

tool_registry.register(
    ToolSpec(
        name="get_daily_facts",
        description="дневные метрики здоровья (HRV, ЧСС, сон, шаги) за интервал дат",
        args_model=GetDailyFactsArgs,
        example_args={
            "date_from": "2026-04-20",
            "date_to": "2026-04-26",
            "metrics": ["hrv", "sleep", "recovery"],
        },
        keywords=(
            "сон", "спал", "пульс", "чсс", "hrv", "вариабельност", "вес",
            "шаги", "восстановл", "recovery", "sleep", "heart rate", "steps",
            "weight", "spo2", "hydration", "вода",
        ),
    )
)

tool_registry.register(
    ToolSpec(
        name="get_user_profile",
        description="профиль пользователя",
        args_model=GetUserProfileArgs,
        example_args={},
        keywords=(
            "профил", "возраст", "вес", "рост", "цел", "опыт", "травм",
            "profile", "goal", "experience", "personal",
        ),
        always_include=True,
    )
)

tool_registry.register(
    ToolSpec(
        name="compute_recovery",
        description="recovery score за окно window_days",
        args_model=ComputeRecoveryArgs,
        example_args={"window_days": 14},
        keywords=(
            "восстановл", "recovery", "готовност", "ready", "readiness",
            "свеж", "fresh",
        ),
    )
)

tool_registry.register(
    ToolSpec(
        name="check_overtraining",
        description="признаки перетренированности за окно window_days",
        args_model=CheckOvertrainingArgs,
        example_args={"window_days": 14},
        keywords=(
            "перетрен", "overtrain", "burnout", "усталост", "fatigue",
            "выгоран", "overload", "exhaust",
        ),
    )
)

tool_registry.register(
    ToolSpec(
        name="rag_retrieve",
        description=(
            "знания из базы. Категории: physiology_norms | training_principles | "
            "recovery_science | sport_specific | nutrition_basics"
        ),
        args_model=RagRetrieveArgs,
        example_args={
            "query_text": "оптимальная зона ЧСС для аэробного бега",
            "category": "training_principles",
            "top_k": 3,
        },
        keywords=(
            "почему", "зачем", "норм", "оптимал", "правильн", "как ", "сколько",
            "why", "how", "norm", "optimal", "should", "best", "recommended",
        ),
        always_include=True,
    )
)
