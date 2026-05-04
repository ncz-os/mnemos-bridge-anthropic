from __future__ import annotations

import inspect
import json
from types import SimpleNamespace
from typing import Any

from mnemos_bridge_core import McpClient, ResultRenderer, SchemaTranslator


class MnemosAnthropicAdapter:
    """Anthropic adapter for MNEMOS MCP tools."""

    def __init__(self, mcp_client: McpClient, tools: list[Any] | None = None) -> None:
        self._client = mcp_client
        self._tools = tools
        self._closed = False

    @classmethod
    async def connect(
        cls,
        mcp_url: str,
        mcp_token: str,
        *,
        timeout: float = 30,
    ) -> MnemosAnthropicAdapter:
        """Discover MNEMOS MCP tools, return ready adapter. Uses mnemos_bridge_core.McpClient."""
        client = McpClient.from_url(mcp_url, token=mcp_token, timeout=timeout)
        await client.__aenter__()

        try:
            tools = await client.list_tools()
        except Exception:
            await client.__aexit__(None, None, None)
            raise

        return cls(client, tools)

    async def anthropic_tools(self) -> list[dict[str, Any]]:
        """Return Anthropic-shape tool list.

        The output shape is [{"name", "description", "input_schema"}]. Translation delegates
        to SchemaTranslator.to_anthropic(), which is mostly passthrough because Anthropic
        input_schema is JSON Schema, matching MCP.
        """
        if self._tools is None:
            self._tools = await self._client.list_tools()

        return [_anthropic_tool(tool) for tool in self._tools]

    async def handle_tool_use(self, tool_use_block: Any) -> dict[str, Any]:
        """Dispatch an Anthropic ToolUseBlock or plain dict and return a tool_result block."""
        tool_use_id = _get_value(tool_use_block, "id")
        name = _get_value(tool_use_block, "name")
        arguments = _get_value(tool_use_block, "input") or {}

        if not tool_use_id:
            raise ValueError("tool_use_block must include an id")
        if not name:
            raise ValueError("tool_use_block must include a name")
        if not isinstance(arguments, dict):
            raise TypeError("tool_use_block input must be a dict")

        result = await self._client.call_tool(name, arguments)
        rendered = ResultRenderer.to_anthropic_message(result, tool_use_id)
        return _tool_result_block(rendered, tool_use_id)

    async def aclose(self) -> None:
        """Close the underlying MCP client session."""
        if self._closed:
            return

        close = getattr(self._client, "aclose", None)
        if close is not None:
            maybe_awaitable = close()
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable
        else:
            await self._client.__aexit__(None, None, None)

        self._closed = True

    async def __aenter__(self) -> MnemosAnthropicAdapter:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()


def _anthropic_tool(tool: Any) -> dict[str, Any]:
    translated = SchemaTranslator.to_anthropic(_normalize_tool(tool))
    return {
        "name": translated["name"],
        "description": translated.get("description") or "",
        "input_schema": translated.get("input_schema") or {"type": "object"},
    }


def _normalize_tool(tool: Any) -> Any:
    if not isinstance(tool, dict):
        return tool

    return SimpleNamespace(
        name=tool.get("name", ""),
        description=tool.get("description") or "",
        input_schema=tool.get("input_schema") or tool.get("inputSchema") or {"type": "object"},
    )


def _get_value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _tool_result_block(rendered: dict[str, Any], tool_use_id: str) -> dict[str, Any]:
    block = rendered
    if rendered.get("type") != "tool_result":
        block = _first_tool_result_content(rendered, tool_use_id)

    return {
        "type": "tool_result",
        "tool_use_id": str(block.get("tool_use_id") or tool_use_id),
        "content": [{"type": "text", "text": _render_text(block.get("content"))}],
    }


def _first_tool_result_content(rendered: dict[str, Any], tool_use_id: str) -> dict[str, Any]:
    content = rendered.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "tool_result":
                if item.get("tool_use_id") in {None, tool_use_id}:
                    return item

    raise ValueError("ResultRenderer.to_anthropic_message() did not return a tool_result block")


def _render_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [_render_text(item) for item in content]
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        if content.get("type") == "text":
            return str(content.get("text") or "")
        if "text" in content:
            return str(content["text"])
        return json.dumps(content, sort_keys=True)
    return str(content)
