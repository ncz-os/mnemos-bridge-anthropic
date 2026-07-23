# ⚠️ This is a mirror — the canonical repo lives on GitLab

### 👉 https://gitlab.com/ncz-os/mnemos-bridge-anthropic

**Source, releases, issues, merge requests, and CI all live on GitLab.** This GitHub copy is a read-only mirror and may lag. Please file issues and get releases there.

---

> # 📍 Moved to GitLab
> **The canonical, authoritative home of this project is GitLab — always:**
> ## 👉 https://gitlab.com/ncz-os/mnemos-bridge-anthropic
>
> This GitHub repository is a **frozen, read-only mirror**. All development, issues, and releases happen on GitLab. Please open issues and merge requests there. The full history of this stub is preserved on GitLab.

---

# mnemos-bridge-anthropic

`mnemos-bridge-anthropic` is the Anthropic surface adapter for the MNEMOS bridge
family. It wraps `mnemos-bridge-core` so Claude can discover and call MNEMOS MCP
tools while acting as a tool-using agent.

Anthropic's tool format is MCP-canonical: tool schemas use JSON Schema directly
as `input_schema`. That makes this the lightest adapter in the family. It mostly
passes MCP tool descriptors through `SchemaTranslator.to_anthropic()` and delegates
tool calls to `McpClient`.

## Install

```bash
pip install mnemos-bridge-anthropic
```

For local development:

```bash
pip install -e .[dev]
```

## Quick Example

```python
import os

from anthropic import AsyncAnthropic
from mnemos_bridge_anthropic import MnemosAnthropicAdapter


async def main() -> str:
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    adapter = await MnemosAnthropicAdapter.connect(
        os.environ["MNEMOS_TEST_BASE"],
        os.environ["MNEMOS_MCP_TOKEN"],
    )

    try:
        tools = await adapter.anthropic_tools()
        messages = [
            {"role": "user", "content": "Search MNEMOS for memories about infrastructure"}
        ]

        response = await client.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            tools=tools,
            messages=messages,
        )

        tool_uses = [block for block in response.content if block.type == "tool_use"]
        if tool_uses:
            tool_result = await adapter.handle_tool_use(tool_uses[0])
            assistant_content = [block.model_dump(exclude_none=True) for block in response.content]

            final = await client.messages.create(
                model="claude-opus-4-7",
                max_tokens=1024,
                tools=tools,
                messages=[
                    *messages,
                    {"role": "assistant", "content": assistant_content},
                    {"role": "user", "content": [tool_result]},
                ],
            )

            return "".join(
                block.text for block in final.content if getattr(block, "type", None) == "text"
            )

        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
    finally:
        await adapter.aclose()
```

## API

### `MnemosAnthropicAdapter.connect(mcp_url, mcp_token, *, timeout=30)`

Opens an MCP client session, discovers MNEMOS tools with `McpClient.list_tools()`,
and returns a ready adapter.

### `await adapter.anthropic_tools()`

Returns Anthropic-shape tools:

```python
[
    {
        "name": "search_memories",
        "description": "Search stored memories.",
        "input_schema": {"type": "object", "properties": {...}},
    }
]
```

### `await adapter.handle_tool_use(tool_use_block)`

Accepts either an Anthropic SDK `ToolUseBlock` or a plain dict with `id`, `name`,
and `input`. It calls the MNEMOS MCP tool and returns an Anthropic `tool_result`
content block suitable for the next user message.

### `await adapter.aclose()`

Closes the underlying MCP client session.

The adapter also supports async context manager usage.

## Integration Tests

Offline tests do not require Anthropic or a live MNEMOS service:

```bash
python -m pytest tests/test_translator_offline.py tests/test_handle_tool_use_offline.py -v
```

The live tool-loop test is skipped unless all required environment variables are
set:

```bash
export ANTHROPIC_API_KEY=...
export MNEMOS_TEST_BASE=https://...
export MNEMOS_MCP_TOKEN=...
python -m pytest tests/integration/test_anthropic_tool_loop.py -v
```

## Claw-Family Policy Note

Anthropic is forbidden as the LLM runtime for `nclawzero` and `zeroclaw`. This
package is a legitimate consumer-side adapter: it exposes MNEMOS MCP tools to a
Claude tool-using client and does not change those runtime restrictions.