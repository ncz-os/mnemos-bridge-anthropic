from __future__ import annotations

from typing import Any

import pytest
from mnemos_bridge_core import ContentBlock, ToolResult

from mnemos_bridge_anthropic import MnemosAnthropicAdapter


class FakeMcpClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        self.calls.append((name, args))
        return ToolResult(
            content=[
                ContentBlock(
                    type="text",
                    text="Found memories about infrastructure planning and rollout.",
                )
            ],
            is_error=False,
        )


@pytest.mark.asyncio
async def test_handle_tool_use_dispatches_and_returns_tool_result_block() -> None:
    client = FakeMcpClient()
    adapter = MnemosAnthropicAdapter(client)
    tool_use = {
        "id": "toolu_abc",
        "name": "search_memories",
        "input": {"query": "infrastructure"},
    }

    result = await adapter.handle_tool_use(tool_use)

    assert client.calls == [("search_memories", {"query": "infrastructure"})]
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "toolu_abc"
    assert isinstance(result["content"], list)
    assert result["content"][0]["type"] == "text"
    assert result["content"][0]["text"]
