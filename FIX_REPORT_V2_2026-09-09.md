# Fix Report V2 - 2026-09-09

## Scope and starting state

- Worked only in `/tmp/mnemos-audit-clones/mnemos-bridge-anthropic` on
  `fix/audit-findings-v2`.
- Verified the starting `HEAD` (`a2673474601cf3dff1a57bf0f87d8dc1fb18810b`) matched
  `origin/master`.
- Preserved the pre-existing untracked `SOL_REVIEW.md`; it is not part of this change.
- No repository clone, remote update, or push was performed.

## What the actual code contained

### Finding 1: unguarded nested-content recursion

The active result path is in `src/mnemos_bridge_anthropic/adapter.py`:
`handle_tool_use()` calls `_tool_result_block()`, which renders through
`_anthropic_content()`. That function recursively flattened nested lists without a depth
limit. The same path also recursively inspected error envelopes in
`_content_reports_mnemos_error()` and could recursively serialize fallback content through
`_render_text()`. The error-envelope scan occurred before the existing
`_UnsupportedContentError` handler, so a deeply nested result could raise an unhandled
`RecursionError` before rendering.

### Finding 2: aliases existed, but no allowlist existed

`anthropic_tools()` already generated Anthropic-safe names and cached an
`alias -> canonical MCP name` map. However, `handle_tool_use()` used
`self._tool_alias_to_mcp.get(name, name)`, so any caller-supplied raw name bypassed the map and
was dispatched. A raw-name allowlist would also have rejected legal advertised aliases for
canonical names such as `namespace/read`.

## Changes made

### Nested-content depth guard

- Added a configurable `max_content_depth` to both adapter construction and `connect()`, with
  a default of 64.
- Bounded accepted configuration to 0 through 100 so configuration cannot raise the limit
  high enough to re-enable Python recursion failure.
- Added an iterative full-tree depth validation before recursive content conversion. It
  covers nested lists, dictionaries, shared containers, and cycles without recursively
  validating them.
- Moved the recursive MNEMOS error-envelope scan behind the guarded conversion and inside the
  existing controlled-error path.
- Excessive nesting now returns an Anthropic `tool_result` with `is_error: true` and a clear
  maximum-depth message instead of leaking `RecursionError`.

### Alias-aware tool allowlist

- Added explicit alias-map initialization state.
- `handle_tool_use()` now initializes the advertised tool map lazily when needed.
- Dispatch requires the supplied name to be a key in the advertised alias map and calls only
  the mapped canonical MCP name.
- Advertised aliases for allowed tools work, while raw or aliased names absent from the
  advertised set fail before `call_tool()`.

## Regression tests added

1. `test_handle_tool_use_rejects_deeply_nested_content_cleanly` constructs 5,000 real nested
   lists, sends them through `handle_tool_use()` and the real result-normalization path, and
   verifies a controlled depth error result rather than `RecursionError`.
2. `test_handle_tool_use_allowlist_resolves_aliases_and_rejects_unadvertised_alias` verifies
   that an advertised alias dispatches to its allowed canonical MCP name and that an alias for
   a non-advertised tool is rejected without reaching `call_tool()`.
3. `test_adapter_rejects_unsafe_max_content_depth` verifies that a configuration above the
   conservative recursion-safe bound is rejected.

The offline fake client now supplies the tool descriptors its existing dispatch tests use so
those tests exercise the new lazy advertised-tool allowlist.

## Verification results

- After Finding 1: targeted depth test `1 passed`; full suite `22 passed, 0 failed, 2 skipped`.
- After Finding 2: targeted alias/allowlist test `1 passed`; full suite
  `23 passed, 0 failed, 2 skipped`.
- Final suite after the review fix: `24 passed, 0 failed, 2 skipped` in 0.04 seconds.
- Final skips were expected environment-gated tests:
  - live integration test requires `ANTHROPIC_API_KEY` and `MNEMOS_TEST_BASE`;
  - real-core mapping test requires the real `mnemos-bridge-core` package in the test runtime.
- `git diff --check` passed.
- Ruff reported 11 pre-existing findings on unchanged lines and no new finding in the added
  code. The repository's existing formatting baseline also remains non-clean; unrelated
  formatting changes were intentionally kept out of this scoped fix.
- Zoder author/review attempts could not run because its zeroclaw daemon/provider was
  unreachable in this environment. The required secondary adversarial review found one
  configurable-depth bypass, which was fixed by the 0-through-100 bound; re-review then
  returned `APPROVE`.

## Commit and delivery

The scoped source, tests, and this report are included in a conventional commit authored as
`Jason Perlow <jperlow@gmail.com>`. No AI-attribution footer was added. No push was performed.
