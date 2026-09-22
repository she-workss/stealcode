# AGENTS.md

Guidance for agents when working in the `stealcode` repository.

## Project

AI coding agent.

## Building and running

- `cargo [+nightly-msvc|+nightly-gnu] r -Fall` - run as TUI application with all enabled features (`desktop`, `server`, `tui`, `voice`, `web`) (also with different toolchains).
- `cargo [+nightly-msvc|+nightly-gnu] r` - run as TUI application.
- `cargo [+nightly-msvc|+nightly-gnu] r -Fdesktop -- desktop` - run as GUI application.
- `cargo [+nightly-msvc|+nightly-gnu] r -Fserver -- serve` - run as a headless server.
- `cargo nt` - run tests (`nt` is an alias from `.cargo/config.toml`).
- `cargo bench` - run benches.
- `cargo clippy --all-targets --all-features` - run clippy.

## Verification after code changes

After writing, reviewing, or refactoring any code, always verify that compilation does not fail before finishing: `cargo +nightly-gnu b` (`cargo +nightly-msvc b`) (and `cargo nt` when tests are affected). Do not leave the workspace in a non-compiling state.

## MCP servers (project-local, auto-connected)

- `gdb` (`mcp-gdb`) - CPU debugging: breakpoints, stepping, variables. Debug a `build/debug` binary, never a release one.
- `renderdoc` (`renderdoc-mcp`) - GPU frame analysis of captures (`open_capture`, `list_draws`, `goto_event`, export render targets). Produce a capture first via RenderDoc UI / `renderdoccmd` (Vulkan layer), then analyze - do not guess GPU state from code alone.
- `context7` - up-to-date library docs. Prefer over training knowledge for API details.
- `zvec_grep` - semantic workspace search. Prefer over `grep` when location is unknown. Run `zg index` first to index all files.
- `samply` - CPU sampling profiler: record a profile (`samply_record`), then inspect hottest functions, callers/callees and threads (`samply_summarize_profile`, `samply_focus_functions`). Profile first - do not guess hotspots from code.
