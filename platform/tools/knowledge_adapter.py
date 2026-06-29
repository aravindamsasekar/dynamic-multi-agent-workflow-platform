"""KnowledgeAdapter — IToolAdapter that searches the knowledge layer."""

from __future__ import annotations

from platform.core.interfaces.tool import IToolAdapter
from platform.core.models.tool import ToolCall, ToolResult
from platform.knowledge.service import KnowledgeService


class KnowledgeAdapter(IToolAdapter):
    """Executes a semantic search over configured knowledge collections.

    Configured via adapter_config in tools.yaml:
        collections: [list-of-collection-names]
        top_k: 5  # optional

    ToolCall.input must contain:
        query: str  — the natural-language search query
    """

    def __init__(
        self,
        service: KnowledgeService,
        collections: list[str],
        top_k: int = 5,
    ) -> None:
        self._service = service
        self._collections = collections
        self._top_k = top_k

    async def execute(self, call: ToolCall) -> ToolResult:
        query: str | None = call.input.get("query")
        if not query:
            return ToolResult(
                tool_use_id=call.tool_use_id,
                content="Missing required field 'query'. Provide a non-empty search query.",
                is_error=True,
            )

        try:
            results = await self._service.search(query, self._collections, self._top_k)
        except Exception as exc:
            return ToolResult(
                tool_use_id=call.tool_use_id,
                content=f"Knowledge search failed: {exc}",
                is_error=True,
            )

        if not results:
            return ToolResult(
                tool_use_id=call.tool_use_id,
                content=f'No results found for query: "{query}"',
            )

        lines: list[str] = [f'Found {len(results)} result(s) for: "{query}"\n']
        for i, r in enumerate(results, 1):
            lines.append(
                f"[{i}] Source: {r.source_file} | Collection: {r.collection} | Score: {r.score:.4f}"
            )
            lines.append(r.text)
            lines.append("")

        return ToolResult(
            tool_use_id=call.tool_use_id,
            content="\n".join(lines).rstrip(),
        )
