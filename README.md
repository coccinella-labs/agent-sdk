# coccinella-agent-sdk

Minimal reusable agent runtime extracted from stable Harper boundaries:

- normalized `ToolCall` / `ToolCallSource` parsing (all provider shapes),
- a provider abstraction with offline and local-Ollama backends,
- a `Tool` trait plus dispatching,
- a tiny fixed-budget agent loop,
- `Error` / `Result` types for every failure path.

Status: alpha (`0.1.0-alpha.1`). The v0.1 surface below is the adoption
target; parser edge cases may still change. Not tagged or published as
`0.1.0` yet; see `CHANGELOG.md` for the release decision.

## Install

Git pin (the supported install path until crates.io publish):

```toml
[dependencies]
coccinella-agent-sdk = { git = "https://github.com/coccinella-labs/agent-sdk", rev = "<commit>" }
```

crates.io (`0.1.0` and later) is deferred; see `CHANGELOG.md`.

## Quick start

Offline, no network. A `StaticBackend` replays scripted replies so the full
parse, dispatch, and continuation path runs under test or in CI.

```rust
use coccinella_agent_sdk::{
    arg_str, Agent, Provider, Result, StaticBackend, Tool, ToolCall,
};

struct Greet;
impl Tool for Greet {
    fn name(&self) -> &'static str {
        "greet"
    }
    fn run(&self, call: &ToolCall) -> Result<String> {
        let who = arg_str(call, "name")?;
        Ok(format!("hello {who}"))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let backend = StaticBackend::new(
        Provider::OpenAI,
        vec![
            r#"[{"function":{"name":"greet","arguments":"{\"name\":\"Ada\"}"}}]"#.into(),
            r#"{"content":"Nice to meet you, Ada."}"#.into(),
        ],
    );
    let mut agent = Agent::new(&backend, vec![Box::new(Greet)], 4);
    let reply = agent.run("greet Ada").await?;
    println!("{reply}");
    Ok(())
}
```

Run the fuller walkthrough:

```bash
cargo run --example basic_agent
```

## API surface (v0.1)

Import from the crate root. Modules (`agent`, `error`, `provider`, `tool`,
`tool_call`) stay public for qualified paths; the re-exports below are the
intended surface.

| Area | Exports |
| --- | --- |
| Loop | `Agent` |
| Tool calls | `ToolCall`, `ToolCallSource`, `parse_tool_calls` |
| Tools | `Tool`, `dispatch`, `arg_str`, `arg_object` |
| Backends | `ModelBackend`, `StaticBackend`, `OllamaBackend`, `ChatRequest` |
| Wire types | `Message`, `Provider` |
| Errors | `Error`, `Result` |

Stable wire contracts:

- `Message` is exactly `{ "role": string, "content": string }`.
- `ChatRequest` is the Ollama `POST /api/chat` body (`model`, `messages`,
  `stream: false`).
- `ModelBackend::chat` returns the raw assistant reply so
  `parse_tool_calls` sees provider shapes unchanged.

## Backends

Two `ModelBackend` implementations ship with the crate. Pick by whether you
need a real model or a deterministic script.

| Backend | Network | Purpose |
| --- | --- | --- |
| `StaticBackend` | none | Deterministic replies for tests, examples, and offline demos. Each `chat` pops the next scripted reply; the last repeats. An empty script yields `Ok("")`. Use `remaining()` to assert how much of the script is left. |
| `OllamaBackend` | local HTTP | Talks to an Ollama daemon at `POST {base}/api/chat` with `stream: false`. Non-2xx responses become `Error::Http` with the URL and status. Never send secrets through this path; it expects a local daemon. |

Same agent code drives both: construct the backend, pass `&backend` to
`Agent::new`, and call `run`. The parse source is derived from
`backend.provider()` unless you override it with `Agent::with_source`.

```bash
# offline (default in the example)
cargo run --example basic_agent

# real local model
OLLAMA_BASE_URL=http://localhost:11434 OLLAMA_MODEL=llama3 cargo run --example basic_agent
```

## Tools

`Tool` is `name()` plus `run(&ToolCall) -> Result<String>`. Arguments are a
`serde_json::Value`; results are plain UTF-8 strings the loop feeds back to
the model.

- Use `arg_str` / `arg_object` for shape-checked reads. They return
  `Error::InvalidArguments` on missing keys, wrong JSON types, or non-object
  payloads.
- `dispatch` matches by exact name (first match wins) and returns
  `Error::UnknownTool` when nothing matches. It does not validate argument
  shapes; that belongs in the tool.
- The trait surface is only `ToolCall` plus `serde_json`. No Harper types
  are part of this API.

## Loop

`Agent::run` is a fixed-budget tool loop (`max_rounds`, minimum 1):

- A reply with zero tool calls is trimmed and returned as the final answer
  (plain text and malformed JSON both take this path).
- On the last round, tool outputs are not fed back; the return value is
  `Reached max rounds (N); last tool outputs:` plus one line per call.
- Unknown tools and tool errors become `[name] error: ...` in the tool
  message; the loop continues so the model can recover.
- Backend transport/HTTP failures propagate as `Err`.
- `max_rounds == 0` returns `Err` without calling the backend.

`run_command` calls are normalized before dispatch. Inspect `history()` for
the user / assistant / tool turns after a run. The loop is a reference
shape; Harper keeps its own richer loop.

## Errors

`Error` is the single failure type (`Result<T>` aliases `Result<T, Error>`):

| Variant | When |
| --- | --- |
| `Parse` | A reply could not be handled as tool calls or a message. |
| `Http` | Transport failure or non-2xx status from a backend (includes URL and status). |
| `UnknownTool` | `dispatch` found no tool with that name. |
| `InvalidArguments { tool, message }` | `arg_str` / `arg_object` shape check failed. |
| `Other` | Everything else, including tool execution failures and `max_rounds == 0`. |

Inside the loop, tool-level errors are degraded to `[name] error: ...`
lines rather than aborting `run`. `run` itself only returns `Err` for
backend failures and a zero round budget.

## Example

```bash
cargo run --example basic_agent          # offline, StaticBackend
OLLAMA_BASE_URL=http://localhost:11434 OLLAMA_MODEL=llama3 cargo run --example basic_agent
```

The example defines two tools, drives one scripted tool round, prints the
final reply, and shows `history()` after the run. Set `OLLAMA_BASE_URL` to
point the same agent at a local daemon.

## License

MIT OR Apache-2.0.
