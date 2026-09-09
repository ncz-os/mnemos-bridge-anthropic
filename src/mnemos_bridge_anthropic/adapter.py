from __future__ import annotations

import base64
import hashlib
import inspect
import json
import re
from types import SimpleNamespace
from typing import Any

from mnemos_bridge_core import McpClient, ResultRenderer, SchemaTranslator


_ANTHROPIC_TOOL_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_ANTHROPIC_IMAGE_MEDIA_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/gif", "image/webp"}
)
_DEFAULT_MAX_CONTENT_DEPTH = 64
_MAX_CONTENT_DEPTH = 100


class _UnsupportedContentError(ValueError):
    pass


def _validate_max_content_depth(max_content_depth: int) -> None:
    if isinstance(max_content_depth, bool) or not isinstance(max_content_depth, int):
        raise TypeError("max_content_depth must be an integer")
    if not 0 <= max_content_depth <= _MAX_CONTENT_DEPTH:
        raise ValueError(f"max_content_depth must be between 0 and {_MAX_CONTENT_DEPTH}")


class MnemosAnthropicAdapter:
    """Anthropic adapter for MNEMOS MCP tools."""

    def __init__(
        self,
        mcp_client: McpClient,
        tools: list[Any] | None = None,
        *,
        max_content_depth: int = _DEFAULT_MAX_CONTENT_DEPTH,
    ) -> None:
        _validate_max_content_depth(max_content_depth)
        self._client = mcp_client
        self._tools = tools
        self._tool_alias_to_mcp: dict[str, str] = {}
        self._tool_aliases_initialized = False
        self._max_content_depth = max_content_depth
        self._closed = False

    @classmethod
    async def connect(
        cls,
        mcp_url: str,
        mcp_token: str,
        *,
        timeout: float = 30,
        max_content_depth: int = _DEFAULT_MAX_CONTENT_DEPTH,
    ) -> MnemosAnthropicAdapter:
        """Discover MNEMOS MCP tools, return ready adapter. Uses mnemos_bridge_core.McpClient."""
        _validate_max_content_depth(max_content_depth)
        client = McpClient.from_url(mcp_url, token=mcp_token, timeout=timeout)

        try:
            await client.__aenter__()
            tools = await client.list_tools()
        except BaseException:
            await client.__aexit__(None, None, None)
            raise

        return cls(client, tools, max_content_depth=max_content_depth)

    async def anthropic_tools(self) -> list[dict[str, Any]]:
        """Return Anthropic-shape tool list.

        The output shape is [{"name", "description", "input_schema"}]. Translation delegates
        to SchemaTranslator.to_anthropic(), which is mostly passthrough because Anthropic
        input_schema is JSON Schema, matching MCP.
        """
        if self._tools is None:
            self._tools = await self._client.list_tools()

        translated_tools = [_anthropic_tool(tool) for tool in self._tools]
        aliases = _tool_name_aliases([tool["name"] for tool in translated_tools])
        self._tool_alias_to_mcp = {
            alias: tool["name"] for alias, tool in zip(aliases, translated_tools, strict=True)
        }
        self._tool_aliases_initialized = True
        return [
            {**tool, "name": alias}
            for alias, tool in zip(aliases, translated_tools, strict=True)
        ]

    async def handle_tool_use(self, tool_use_block: Any) -> dict[str, Any]:
        """Dispatch an Anthropic ToolUseBlock or plain dict and return a tool_result block."""
        tool_use_id = _get_value(tool_use_block, "id")
        name = _get_value(tool_use_block, "name")
        arguments = _get_value(tool_use_block, "input")
        if arguments is None:
            arguments = {}

        if not tool_use_id:
            raise ValueError("tool_use_block must include an id")
        if not name:
            raise ValueError("tool_use_block must include a name")
        if not isinstance(arguments, dict):
            raise TypeError("tool_use_block input must be a dict")

        if not self._tool_aliases_initialized:
            await self.anthropic_tools()
        try:
            mcp_name = self._tool_alias_to_mcp[name]
        except KeyError:
            raise ValueError(f"tool {name!r} is not in the advertised Anthropic tool set") from None
        result = await self._client.call_tool(mcp_name, arguments)
        rendered = ResultRenderer.to_anthropic_message(result, tool_use_id)
        return _tool_result_block(
            rendered,
            tool_use_id,
            max_content_depth=self._max_content_depth,
        )

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


def _tool_name_aliases(names: list[str]) -> list[str]:
    if len(set(names)) != len(names):
        raise ValueError("MCP tool names must be unique")

    reserved = {name for name in names if _ANTHROPIC_TOOL_NAME.fullmatch(name)}
    aliases: list[str] = []
    for name in names:
        if name in reserved:
            aliases.append(name)
            continue

        safe_stem = re.sub(r"[^a-zA-Z0-9_-]", "_", name) or "tool"
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
        for digest_length in range(8, len(digest) + 1):
            stem_length = 64 - digest_length - 1
            candidate = (
                f"{safe_stem[:stem_length]}_{digest[:digest_length]}"
                if stem_length > 0
                else digest[:64]
            )
            if candidate not in reserved:
                break
        else:  # pragma: no cover - a full SHA-256 collision is not constructible in tests
            raise ValueError(f"Could not create a unique Anthropic alias for MCP tool {name!r}")
        reserved.add(candidate)
        aliases.append(candidate)
    return aliases


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


def _tool_result_block(
    rendered: dict[str, Any],
    tool_use_id: str,
    *,
    max_content_depth: int = _DEFAULT_MAX_CONTENT_DEPTH,
) -> dict[str, Any]:
    block = rendered
    if rendered.get("type") != "tool_result":
        block = _first_tool_result_content(rendered, tool_use_id)

    is_error = bool(block.get("is_error", False))
    try:
        raw_content = block.get("content")
        content = _anthropic_content(raw_content, max_depth=max_content_depth)
        is_error = is_error or _content_reports_mnemos_error(raw_content)
    except _UnsupportedContentError as exc:
        content = [{"type": "text", "text": str(exc)}]
        is_error = True

    result = {
        "type": "tool_result",
        "tool_use_id": str(block.get("tool_use_id") or tool_use_id),
        "is_error": is_error,
    }
    if content:
        result["content"] = content
    return result


def _first_tool_result_content(rendered: dict[str, Any], tool_use_id: str) -> dict[str, Any]:
    content = rendered.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "tool_result":
                if item.get("tool_use_id") in {None, tool_use_id}:
                    return item

    raise ValueError("ResultRenderer.to_anthropic_message() did not return a tool_result block")


def _anthropic_content(
    content: Any,
    *,
    max_depth: int = _DEFAULT_MAX_CONTENT_DEPTH,
) -> list[dict[str, Any]]:
    _validate_content_depth(content, max_depth=max_depth)
    return _anthropic_content_unchecked(content)


def _anthropic_content_unchecked(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [block for item in content for block in _anthropic_content_unchecked(item)]
    if isinstance(content, dict):
        if content.get("type") == "image":
            if isinstance(content.get("source"), dict):
                source = dict(content["source"])
                if source.get("type") == "base64":
                    source["media_type"] = _anthropic_image_media_type(
                        source.get("media_type")
                    )
                return [{**content, "source": source}]
            data = content.get("data", b"")
            if isinstance(data, (bytes, bytearray, memoryview)):
                data = base64.b64encode(data).decode("ascii")
            return [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": _anthropic_image_media_type(
                            content.get("mime_type") or content.get("mimeType")
                        ),
                        "data": str(data),
                    },
                }
            ]
        if content.get("type") == "text":
            return [{"type": "text", "text": str(content.get("text") or "")}]
        if content.get("type") in {"resource", "embedded_resource"}:
            resource = content.get("resource")
            if isinstance(resource, dict):
                if "text" in resource:
                    return [{"type": "text", "text": str(resource["text"])}]
                if "blob" in resource:
                    mime_type = resource.get("mimeType") or resource.get("mime_type")
                    data = resource["blob"]
                    if isinstance(data, (bytes, bytearray, memoryview)):
                        data = base64.b64encode(data).decode("ascii")
                    if isinstance(mime_type, str) and mime_type.startswith("image/"):
                        return [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": _anthropic_image_media_type(mime_type),
                                    "data": str(data),
                                },
                            }
                        ]
                    if mime_type == "application/pdf":
                        return [
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": str(data),
                                },
                            }
                        ]
            raise _UnsupportedContentError(
                "MCP tool returned an embedded resource that Anthropic cannot represent."
            )
        if content.get("type") == "audio":
            raise _UnsupportedContentError(
                "MCP tool returned audio content that Anthropic tool results cannot represent."
            )
    return [{"type": "text", "text": _render_text(content)}]


def _validate_content_depth(content: Any, *, max_depth: int) -> None:
    stack = [(content, 0)]
    greatest_depth_by_id: dict[int, int] = {}
    while stack:
        value, depth = stack.pop()
        if depth > max_depth:
            raise _UnsupportedContentError(
                f"MCP tool result exceeds the maximum content nesting depth of {max_depth}."
            )
        if not isinstance(value, (list, dict)):
            continue

        value_id = id(value)
        if greatest_depth_by_id.get(value_id, -1) >= depth:
            continue
        greatest_depth_by_id[value_id] = depth
        children = value if isinstance(value, list) else value.values()
        stack.extend((child, depth + 1) for child in children)


def _anthropic_image_media_type(media_type: Any) -> str:
    normalized = str(media_type or "image/png").strip().lower()
    if normalized not in _ANTHROPIC_IMAGE_MEDIA_TYPES:
        raise _UnsupportedContentError(
            f"MCP tool returned unsupported Anthropic image media type: {normalized}"
        )
    return normalized


def _content_reports_mnemos_error(content: Any) -> bool:
    if isinstance(content, list):
        return any(_content_reports_mnemos_error(item) for item in content)
    if isinstance(content, dict):
        if content.get("type") == "text":
            return _content_reports_mnemos_error(content.get("text"))
        return content.get("success") is False or bool(content.get("error"))
    if not isinstance(content, str):
        return False
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(payload, dict) and (
        payload.get("success") is False or bool(payload.get("error"))
    )


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
