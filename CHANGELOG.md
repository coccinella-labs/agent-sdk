# Changelog

All notable changes to `coccinella-agent-sdk` are documented here.

## [Unreleased]

### Decision: hold at `0.1.0-alpha.1` (2026-09-24)

Recorded for issue #6. The crate is **not** tagged `0.1.0` and is **not**
published to crates.io.

Reasons:

1. **Harper has only exercised the `tool_call` boundary.** Harper PR #870
   re-exports `ToolCall`, `ToolCallSource`, and `parse_tool_calls`. The
   `Agent` loop, `Tool` / `dispatch`, `ModelBackend`, `StaticBackend`, and
   `OllamaBackend` are not yet consumed by Harper. The issue requires the
   first real tag after the API has been exercised by Harper.
2. **No crates.io credentials** are available in the release environment, so
   a publish cannot be completed or verified here.
3. **`OllamaBackend` is not e2e-verified** against a live daemon. The HTTP
   contract is unit-tested only.

Release prep completed in this change:

- Dual license files added (`LICENSE-MIT`, `LICENSE-APACHE`).
- `cargo package` succeeds (16 files; package verifies and builds).
- Changelog and README status updated to match the decision.

When those three blockers clear, the release path is:

1. Bump `version` to `0.1.0` in `Cargo.toml`.
2. `cargo publish`.
3. Tag `v0.1.0`.
4. Replace the `harper-core` git dependency
   (`rev = 8df2ba0...`) with `coccinella-agent-sdk = "0.1.0"` and refresh
   `Cargo.lock`.

Until then, consumers pin the git revision:

```toml
coccinella-agent-sdk = { git = "https://github.com/coccinella-labs/agent-sdk", rev = "<commit>" }
```

### Added (milestone #1 to #4)

- Provider API: `Message`, `ChatRequest`, `ModelBackend` contract,
  `OllamaBackend` non-2xx -> `Error::Http`, `StaticBackend::remaining`.
- Tool API: `Error::InvalidArguments`, `arg_str` / `arg_object`, documented
  `Tool` / `dispatch` contract (first duplicate wins; no arg validation in
  dispatch).
- Agent loop: termination / continuation / degradation contracts,
  `max_rounds == 0` guard, `run_command` normalization before dispatch,
  `history()` and `max_rounds()` accessors, role=tool messages.
- Public API docs (`src/lib.rs` crate docs), README adoption path, expanded
  offline `examples/basic_agent.rs`.
- 57 unit tests; `cargo fmt`, `clippy -D warnings`, and `cargo doc --no-deps`
  clean.

### Boundary

The SDK boundary remains: Harper consumes `tool_call` as the first shared
runtime surface. Harper keeps its own richer agent loop.
