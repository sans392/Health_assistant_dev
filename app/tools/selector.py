"""Tool selection: pick a relevant subset of tools for a given user query.

Why this exists:
The planner system prompt currently lists every tool every time. As the
registry grows that bloat hurts small LLMs and the upcoming vLLM throughput
budget. The selector lets us shrink the visible tool set per request.

Two implementations are provided here:

- `AllToolsSelector` — passthrough, returns every tool. Default, preserves
  existing behaviour exactly.
- `KeywordToolSelector` — substring match against `ToolSpec.keywords` plus
  `always_include`. Cheap and dependency-free; useful as a stub and as a
  guardrail layer once embeddings ship.

Embedding-based selector (planned): an `EmbeddingToolSelector` will embed the
query plus a per-tool description sentence and pick top-K. It will live next
to this file so the rest of the pipeline keeps a single import surface
(`from app.tools.selector import tool_selector`).
"""

from __future__ import annotations

from typing import Protocol

from app.tools.registry import ToolRegistry, ToolSpec, tool_registry


class ToolSelector(Protocol):
    """Anything that can pick `ToolSpec`s for a user query."""

    def select(
        self,
        query: str,
        *,
        max_tools: int | None = None,
    ) -> list[ToolSpec]:
        """Return tools relevant to `query`. Order matters — higher first."""
        ...


class AllToolsSelector:
    """Returns every registered tool. Equivalent to today's planner behaviour."""

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self._registry = registry or tool_registry

    def select(
        self,
        query: str,
        *,
        max_tools: int | None = None,
    ) -> list[ToolSpec]:
        specs = self._registry.all()
        if max_tools is not None:
            specs = specs[:max_tools]
        return specs


class KeywordToolSelector:
    """Substring keyword matching, with `always_include` tools pinned on top.

    Scoring: number of distinct keywords from `ToolSpec.keywords` that occur
    in the lowercased query. Ties are broken by registration order. Tools with
    score 0 and `always_include=False` are dropped — unless the result would
    be empty, in which case we fall back to all tools (a small LLM with no
    tools is worse than one with a few extras).
    """

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self._registry = registry or tool_registry

    def select(
        self,
        query: str,
        *,
        max_tools: int | None = None,
    ) -> list[ToolSpec]:
        q = (query or "").lower()
        scored: list[tuple[int, int, ToolSpec]] = []
        for idx, spec in enumerate(self._registry.all()):
            score = sum(1 for kw in spec.keywords if kw and kw.lower() in q)
            scored.append((score, idx, spec))

        always_pinned = [spec for _, _, spec in scored if spec.always_include]
        matched = sorted(
            (item for item in scored if item[0] > 0 and not item[2].always_include),
            key=lambda item: (-item[0], item[1]),
        )

        ordered: list[ToolSpec] = []
        seen: set[str] = set()
        for spec in always_pinned:
            if spec.name not in seen:
                ordered.append(spec)
                seen.add(spec.name)
        for _, _, spec in matched:
            if spec.name not in seen:
                ordered.append(spec)
                seen.add(spec.name)

        if not ordered:
            ordered = self._registry.all()

        if max_tools is not None:
            ordered = ordered[:max_tools]
        return ordered


# Default selector: passthrough. Swap to KeywordToolSelector (or future
# embedding-based one) when we want to start trimming the prompt.
tool_selector: ToolSelector = AllToolsSelector()
