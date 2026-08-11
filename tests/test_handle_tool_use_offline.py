from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace
from typing import Any

import pytest
from mnemos_bridge_core import ContentBlock, McpClient, ToolResult

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
async def test_handle_tool_use_recognizes_mnemos_failure_envelope() -> None:
    client = FakeMcpClient()

    async def call_tool(name: str, args: dict[str, Any]) -> ToolResult:
        return ToolResult(
            content=[
                ContentBlock(
                    type="text",
                    text='{"success": false, "error": "Rate limit exceeded"}',
                )
            ],
            is_error=False,
        )

    client.call_tool = call_tool  # type: ignore[method-assign]
    adapter = MnemosAnthropicAdapter(client)

    result = await adapter.handle_tool_use(
        {"id": "toolu_failure", "name": "create_memory", "input": {}}
    )

    assert result["is_error"] is True


@pytest.mark.asyncio
async def test_handle_tool_use_preserves_empty_success() -> None:
    client = FakeMcpClient()

    async def call_tool(name: str, args: dict[str, Any]) -> ToolResult:
        return ToolResult(content=[], is_error=False)

    client.call_tool = call_tool  # type: ignore[method-assign]
    adapter = MnemosAnthropicAdapter(client)

    result = await adapter.handle_tool_use(
        {"id": "toolu_empty", "name": "read_resource", "input": {}}
    )

    assert result["is_error"] is False
    assert "content" not in result


@pytest.mark.asyncio
async def test_handle_tool_use_translates_embedded_text_through_real_core_mapping() -> None:
    if not hasattr(McpClient, "_map_content"):
        pytest.skip("requires the real mnemos-bridge-core package")

    class CoreMappingClient(FakeMcpClient):
        async def call_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
            self.calls.append((name, args))
            item = SimpleNamespace(
                type="resource",
                resource=SimpleNamespace(
                    uri="memory://record/123",
                    mimeType="text/plain",
                    text="embedded memory text",
                ),
            )
            block = McpClient._map_content(item)
            assert block is not None
            return ToolResult(content=[block], is_error=False)

    client = CoreMappingClient()
    adapter = MnemosAnthropicAdapter(client)

    result = await adapter.handle_tool_use(
        {"id": "toolu_resource", "name": "read_resource", "input": {}}
    )

    assert result["is_error"] is False
    assert result["content"] == [{"type": "text", "text": "embedded memory text"}]


@pytest.mark.asyncio
async def test_handle_tool_use_translates_embedded_text_resource(monkeypatch: Any) -> None:
    client = FakeMcpClient()
    monkeypatch.setattr(
        adapter_module.ResultRenderer,
        "to_anthropic_message",
        lambda result, tool_use_id: {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": [
                {
                    "type": "resource",
                    "resource": {
                        "uri": "memory://record/123",
                        "mimeType": "text/plain",
                        "text": "embedded memory text",
                    },
                }
            ],
            "is_error": False,
        },
    )
    adapter = MnemosAnthropicAdapter(client)

    result = await adapter.handle_tool_use(
        {"id": "toolu_resource", "name": "read_resource", "input": {}}
    )

    assert result["is_error"] is False
    assert result["content"] == [{"type": "text", "text": "embedded memory text"}]


@pytest.mark.asyncio
async def test_handle_tool_use_translates_embedded_pdf_resource(monkeypatch: Any) -> None:
    client = FakeMcpClient()
    encoded_pdf = base64.b64encode(b"%PDF-1.7").decode("ascii")
    monkeypatch.setattr(
        adapter_module.ResultRenderer,
        "to_anthropic_message",
        lambda result, tool_use_id: {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": [
                {
                    "type": "resource",
                    "resource": {
                        "uri": "memory://report/123",
                        "mimeType": "application/pdf",
                        "blob": encoded_pdf,
                    },
                }
            ],
            "is_error": False,
        },
    )
    adapter = MnemosAnthropicAdapter(client)

    result = await adapter.handle_tool_use(
        {"id": "toolu_resource", "name": "read_resource", "input": {}}
    )

    assert result["is_error"] is False
    assert result["content"] == [
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": encoded_pdf,
            },
        }
    ]


@pytest.mark.asyncio
async def test_handle_tool_use_rejects_audio_as_explicit_error(monkeypatch: Any) -> None:
    client = FakeMcpClient()
    monkeypatch.setattr(
        adapter_module.ResultRenderer,
        "to_anthropic_message",
        lambda result, tool_use_id: {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": [{"type": "audio", "mimeType": "audio/wav", "data": "AAAA"}],
            "is_error": False,
        },
    )
    adapter = MnemosAnthropicAdapter(client)

    result = await adapter.handle_tool_use(
        {"id": "toolu_audio", "name": "speech", "input": {}}
    )

    assert result["is_error"] is True
    assert "audio" in result["content"][0]["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "media_type",
    ["image/jpeg", "image/png", "image/gif", "image/webp"],
)
async def test_handle_tool_use_converts_image_content_to_anthropic_source(
    media_type: str,
) -> None:
    client = FakeMcpClient()

    async def call_tool(name: str, args: dict[str, Any]) -> ToolResult:
        return ToolResult(
            content=[ContentBlock(type="image", data=b"image bytes", mime_type=media_type)]
        )

    client.call_tool = call_tool  # type: ignore[method-assign]
    adapter = MnemosAnthropicAdapter(client)

    result = await adapter.handle_tool_use({"id": "toolu_image", "name": "chart", "input": {}})

    assert result["content"] == [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64.b64encode(b"image bytes").decode("ascii"),
            },
        }
    ]


@pytest.mark.asyncio
async def test_handle_tool_use_preserves_image_already_rendered_by_core(
    monkeypatch: Any,
) -> None:
    client = FakeMcpClient()
    rendered_image = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.b64encode(b"jpeg bytes").decode("ascii"),
        },
    }
    monkeypatch.setattr(
        adapter_module.ResultRenderer,
        "to_anthropic_message",
        lambda result, tool_use_id: {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": [rendered_image],
                    "is_error": False,
                }
            ],
        },
    )
    adapter = MnemosAnthropicAdapter(client)

    result = await adapter.handle_tool_use(
        {"id": "toolu_image", "name": "chart", "input": {}}
    )

    assert result["content"] == [rendered_image]


@pytest.mark.asyncio
async def test_handle_tool_use_rejects_unsupported_image_mime_type() -> None:
    client = FakeMcpClient()

    async def call_tool(name: str, args: dict[str, Any]) -> ToolResult:
        return ToolResult(
            content=[ContentBlock(type="image", data=b"svg", mime_type="image/svg+xml")]
        )

    client.call_tool = call_tool  # type: ignore[method-assign]
    adapter = MnemosAnthropicAdapter(client)

    result = await adapter.handle_tool_use(
        {"id": "toolu_svg", "name": "chart", "input": {}}
    )

    assert result["is_error"] is True
    assert result["content"][0]["type"] == "text"
    assert "image/svg+xml" in result["content"][0]["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_input", [[], "", 0, False])
async def test_handle_tool_use_rejects_falsey_non_dict_input(invalid_input: Any) -> None:
    client = FakeMcpClient()
    adapter = MnemosAnthropicAdapter(client)

    with pytest.raises(TypeError, match="input must be a dict"):
        await adapter.handle_tool_use(
            {"id": "toolu_invalid", "name": "optional_side_effect", "input": invalid_input}
        )

    assert client.calls == []


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
