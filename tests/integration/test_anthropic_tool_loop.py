from __future__ import annotations

import os
from typing import Any

import pytest

from mnemos_bridge_anthropic import MnemosAnthropicAdapter


def _dump_block(block: Any) -> dict[str, Any]:
    if hasattr(block, "model_dump"):
        return block.model_dump(exclude_none=True)
    if isinstance(block, dict):
        return block
    raise TypeError(f"Cannot serialize Anthropic content block {type(block)!r}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_anthropic_tool_loop() -> None:
    if not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("MNEMOS_TEST_BASE"):
        pytest.skip("requires ANTHROPIC_API_KEY and MNEMOS_TEST_BASE")

    from anthropic import AsyncAnthropic

    mcp_token = os.getenv("MNEMOS_MCP_TOKEN", "")
    adapter = await MnemosAnthropicAdapter.connect(os.environ["MNEMOS_TEST_BASE"], mcp_token)
    anthropic = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    try:
        tools = await adapter.anthropic_tools()
        user_message = "Search MNEMOS for memories about infrastructure"

        response = await anthropic.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            tools=tools,
            messages=[{"role": "user", "content": user_message}],
        )

        tool_uses = [block for block in response.content if getattr(block, "type", None) == "tool_use"]
        assert tool_uses

        tool_result = await adapter.handle_tool_use(tool_uses[0])
        final = await anthropic.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            tools=tools,
            messages=[
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": [_dump_block(block) for block in response.content]},
                {"role": "user", "content": [tool_result]},
            ],
        )

        final_text = "\n".join(
            block.text for block in final.content if getattr(block, "type", None) == "text"
        )
        assert "search" in final_text.lower() or "mnemos" in final_text.lower()
    finally:
        await adapter.aclose()
