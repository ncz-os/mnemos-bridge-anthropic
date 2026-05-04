from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any


try:
    import mnemos_bridge_core  # noqa: F401
except ModuleNotFoundError:
    core = types.ModuleType("mnemos_bridge_core")

    @dataclass
    class ContentBlock:
        type: str
        text: str | None = None
        data: bytes | None = None
        mime_type: str | None = None
        uri: str | None = None

        def model_dump(self, *, exclude_none: bool = False) -> dict[str, Any]:
            data = {
                "type": self.type,
                "text": self.text,
                "data": self.data,
                "mime_type": self.mime_type,
                "uri": self.uri,
            }
            if exclude_none:
                return {key: value for key, value in data.items() if value is not None}
            return data

    @dataclass
    class ToolResult:
        content: list[ContentBlock]
        is_error: bool = False

    @dataclass
    class ToolSchema:
        name: str
        description: str
        input_schema: dict[str, Any]

    class McpClient:
        def __init__(self, url: str = "", *, token: str = "", timeout: float = 30) -> None:
            self.url = url
            self.token = token
            self.timeout = timeout

        @classmethod
        def from_url(cls, url: str, *, token: str, timeout: float = 30) -> McpClient:
            return cls(url, token=token, timeout=timeout)

        async def __aenter__(self) -> McpClient:
            return self

        async def __aexit__(self, *_: Any) -> None:
            return None

        async def list_tools(self) -> list[ToolSchema]:
            return []

        async def call_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
            return ToolResult([ContentBlock(type="text", text=f"{name}: {args}")])

    class SchemaTranslator:
        @staticmethod
        def to_anthropic(t: ToolSchema) -> dict[str, Any]:
            return {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }

    class ResultRenderer:
        @staticmethod
        def to_anthropic_message(r: ToolResult, tool_use_id: str) -> dict[str, Any]:
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": [
                            block.model_dump(exclude_none=True) for block in r.content if block.text
                        ],
                    }
                ],
            }

    core.ContentBlock = ContentBlock
    core.McpClient = McpClient
    core.ResultRenderer = ResultRenderer
    core.SchemaTranslator = SchemaTranslator
    core.ToolResult = ToolResult
    core.ToolSchema = ToolSchema

    sys.modules["mnemos_bridge_core"] = core
