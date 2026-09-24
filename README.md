# coccinella-agent-sdk

Minimal reusable agent runtime extracted from stable Harper boundaries:

- normalized `ToolCall` / `ToolCallSource` parsing (all provider shapes),
- a small provider abstraction with a local `OllamaBackend`,
- a `Tool` trait plus dispatching,
- a tiny, independently testable agent loop,
- clean `Error` / `Result` types.

Status: spike (`0.1.0-alpha.1`). Not production-ready.

## Backends

Two `ModelBackend` implementations ship with the crate. Pick by whether you
need a real model or a deterministic script.

| Backend | Network | Purpose |
| --- | --- | --- |
| `StaticBackend` | none | Deterministic replies for tests, examples, and offline demos. Each `chat` pops the next scripted reply; the last repeats. An empty script yields `Ok("")`. |
| `OllamaBackend` | local HTTP | Talks to an Ollama daemon at `POST {base}/api/chat` with `stream: false`. Non-2xx responses become `Error::Http` with the URL and status. |

`Message` is the stable wire type (`role` + `content`). `ChatRequest` is the
stable Ollama request body (`model`, `messages`, `stream`). `ModelBackend`
returns the raw assistant reply string so `parse_tool_calls` sees provider
shapes unchanged.

## Tools

`Tool` is `name()` plus `run(&ToolCall) -> Result<String>`. Arguments are a
`serde_json::Value`; results are plain UTF-8 strings the loop feeds back to
the model. Use `arg_str` / `arg_object` for shape-checked reads (they return
`Error::InvalidArguments`). `dispatch` matches by exact name (first match
wins) and returns `Error::UnknownTool` when nothing matches; it does not
validate argument shapes.

## Use

```rust
use coccinella_agent_sdk::{Agent, Tool, ToolCall, Result, arg_str};
use serde_json::json;

struct Greet;
impl Tool for Greet {
    fn name(&self) -> &'static str { "greet" }
    fn run(&self, call: &ToolCall) -> Result<String> {
        let who = arg_str(call, "name").unwrap_or("world");
        Ok(format!("hello {who}"))
    }
}
```

## Example

```bash
cargo run --example basic_agent          # offline, StaticBackend
OLLAMA_BASE_URL=http://localhost:11434 OLLAMA_MODEL=llama3 cargo run --example basic_agent
```

## License

MIT OR Apache-2.0.
