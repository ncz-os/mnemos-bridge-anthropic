from __future__ import annotations

import asyncio
import base64
from typing import Any

import pytest
from mnemos_bridge_core import ContentBlock, ToolResult

from mnemos_bridge_anthropic import MnemosAnthropicAdapter
from mnemos_bridge_anthropic import adapter as adapter_module


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
    assert result["is_error"] is False


@pytest.mark.asyncio
async def test_handle_tool_use_preserves_tool_error_state() -> None:
    client = FakeMcpClient()

    async def call_tool(name: str, args: dict[str, Any]) -> ToolResult:
        return ToolResult(content=[ContentBlock(type="text", text="permission denied")], is_error=True)

    client.call_tool = call_tool  # type: ignore[method-assign]
    adapter = MnemosAnthropicAdapter(client)

    result = await adapter.handle_tool_use({"id": "toolu_error", "name": "write", "input": {}})

    assert result["is_error"] is True
    assert result["content"] == [{"type": "text", "text": "permission denied"}]


@pytest.mark.asyncio
async def test_handle_tool_use_converts_image_content_to_anthropic_source() -> None:
    client = FakeMcpClient()

    async def call_tool(name: str, args: dict[str, Any]) -> ToolResult:
        return ToolResult(
            content=[ContentBlock(type="image", data=b"image bytes", mime_type="image/png")]
        )

    client.call_tool = call_tool  # type: ignore[method-assign]
    adapter = MnemosAnthropicAdapter(client)

    result = await adapter.handle_tool_use({"id": "toolu_image", "name": "chart", "input": {}})

    assert result["content"] == [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(b"image bytes").decode("ascii"),
            },
        }
    ]


@pytest.mark.asyncio
async def test_connect_closes_client_when_tool_discovery_is_cancelled(monkeypatch: Any) -> None:
    class CancellingClient:
        def __init__(self) -> None:
            self.exited = False

        async def __aenter__(self) -> CancellingClient:
            return self

        async def list_tools(self) -> list[Any]:
            raise asyncio.CancelledError

        async def __aexit__(self, *_: Any) -> None:
            self.exited = True

    client = CancellingClient()
    monkeypatch.setattr(adapter_module.McpClient, "from_url", lambda *args, **kwargs: client)

    with pytest.raises(asyncio.CancelledError):
        await MnemosAnthropicAdapter.connect("https://mcp.test", "token")

    assert client.exited is True


@pytest.mark.asyncio
async def test_connect_cleans_up_partial_enter_failure(monkeypatch: Any) -> None:
    class PartiallyOpeningClient:
        def __init__(self) -> None:
            self.exited = False

        async def __aenter__(self) -> PartiallyOpeningClient:
            raise RuntimeError("session initialization failed")

        async def __aexit__(self, *_: Any) -> None:
            self.exited = True

    client = PartiallyOpeningClient()
    monkeypatch.setattr(adapter_module.McpClient, "from_url", lambda *args, **kwargs: client)

    with pytest.raises(RuntimeError, match="session initialization failed"):
        await MnemosAnthropicAdapter.connect("https://mcp.test", "token")

    assert client.exited is True
