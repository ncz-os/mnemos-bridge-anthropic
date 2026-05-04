# Changelog

## v0.1.0 - 2026-05-03

- Initial Anthropic adapter for the MNEMOS bridge family.
- Added MCP-canonical tool translation through `SchemaTranslator.to_anthropic()`.
- Added Anthropic `tool_use` dispatch through `McpClient.call_tool()`.
- Added offline adapter tests and skipped live Anthropic/MNEMOS integration test.
