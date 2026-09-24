# coccinella-agent-sdk

Minimal reusable agent runtime extracted from stable Harper boundaries:

- normalized `ToolCall` / `ToolCallSource` parsing (all provider shapes),
- a small provider abstraction with a local `OllamaBackend`,
- a `Tool` trait plus dispatching,
- a tiny, independently testable agent loop,
- clean `Error` / `Result` types.

Status: spike (`0.1.0-alpha.1`). Not production-ready.

## Use

```rust
use coccinella_agent_sdk::{Agent, Tool, ToolCall, Result};
use serde_json::json;

struct Greet;
impl Tool for Greet {
    fn name(&self) -> &'static str { "greet" }
    fn run(&self, call: &ToolCall) -> Result<String> {
        let who = call.arguments.get("name").and_then(|v| v.as_str()).unwrap_or("world");
        Ok(format!("hello {who}"))
    }
}
```

## Example

```bash
cargo run --example basic_agent          # offline, scripted backend
OLLAMA_BASE_URL=http://localhost:11434 OLLAMA_MODEL=llama3 cargo run --example basic_agent
```

## License

MIT OR Apache-2.0.