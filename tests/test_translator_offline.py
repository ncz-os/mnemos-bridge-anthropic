from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

import pytest

from mnemos_bridge_anthropic import MnemosAnthropicAdapter


class FakeMcpClient:
    def __init__(self, tools: list[Any]) -> None:
        self._tools = tools

    async def list_tools(self) -> list[Any]:
        return self._tools


@pytest.mark.asyncio
async def test_anthropic_tools_are_minimal_mcp_passthrough_shape() -> None:
    tools = [
        {
            "name": "search_memories",
            "description": "Search memory records.",
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            "annotations": {"readOnlyHint": True},
        },
        SimpleNamespace(
            name="read_memory",
            description="Read one memory.",
            input_schema={
                "type": "object",
                "properties": {"memory_id": {"type": "string"}},
                "required": ["memory_id"],
            },
        ),
        {
            "name": "list_projects",
            "description": "List known projects.",
            "input_schema": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "minimum": 1}},
            },
        },
    ]

    adapter = MnemosAnthropicAdapter(FakeMcpClient(tools), tools)

    anthropic_tools = await adapter.anthropic_tools()

    assert len(anthropic_tools) == len(tools)
    for tool in anthropic_tools:
        assert set(tool) == {"name", "description", "input_schema"}
        assert isinstance(tool["name"], str)
        assert isinstance(tool["description"], str)
        assert isinstance(tool["input_schema"], dict)
        assert tool["input_schema"]["type"] == "object"


@pytest.mark.asyncio
async def test_anthropic_tools_alias_unsafe_names_and_dispatch_original_name() -> None:
    unsafe_names = [
        "admin.tools.list",
        "x" * 65,
        "admin_tools_list",
    ]
    tools = [
        {"name": name, "description": "Tool", "inputSchema": {"type": "object"}}
        for name in unsafe_names
    ]
    client = FakeMcpClient(tools)
    calls: list[str] = []

    async def call_tool(name: str, args: dict[str, Any]) -> Any:
        calls.append(name)
        return SimpleNamespace(content=[], is_error=True)

    client.call_tool = call_tool  # type: ignore[attr-defined]
    adapter = MnemosAnthropicAdapter(client, tools)

    anthropic_tools = await adapter.anthropic_tools()
    aliases = [tool["name"] for tool in anthropic_tools]

    assert len(set(aliases)) == len(unsafe_names)
    assert all(re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", alias) for alias in aliases)
    assert aliases[2] == "admin_tools_list"

    await adapter.handle_tool_use({"id": "toolu_alias", "name": aliases[0], "input": {}})

    assert calls == ["admin.tools.list"]


@pytest.mark.asyncio
async def test_handle_tool_use_allowlist_resolves_aliases_and_rejects_unadvertised_alias() -> None:
    allowed_tools = [
        {
            "name": "namespace/read",
            "description": "Allowed tool",
            "inputSchema": {"type": "object"},
        }
    ]
    client = FakeMcpClient(allowed_tools)
    calls: list[str] = []

    async def call_tool(name: str, args: dict[str, Any]) -> Any:
        calls.append(name)
        return SimpleNamespace(content=[], is_error=False)

    client.call_tool = call_tool  # type: ignore[attr-defined]
    adapter = MnemosAnthropicAdapter(client, allowed_tools)
    allowed_alias = (await adapter.anthropic_tools())[0]["name"]

    denied_tools = [
        {
            "name": "namespace/write",
            "description": "Denied tool",
            "inputSchema": {"type": "object"},
        }
    ]
    denied_alias = (
        await MnemosAnthropicAdapter(FakeMcpClient(denied_tools), denied_tools).anthropic_tools()
    )[0]["name"]

    await adapter.handle_tool_use({"id": "toolu_allowed_alias", "name": allowed_alias, "input": {}})
    with pytest.raises(ValueError, match="not in the advertised Anthropic tool set"):
        await adapter.handle_tool_use(
            {"id": "toolu_denied_alias", "name": denied_alias, "input": {}}
        )

    assert allowed_alias != "namespace/read"
    assert denied_alias != "namespace/write"
    assert calls == ["namespace/read"]
