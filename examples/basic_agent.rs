// Copyright 2026 coccinella-labs
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Minimal working example: a scripted backend drives the loop through one
//! tool call and a final text reply. Runs fully offline.
//!
//! To talk to a real local model instead, point `OllamaBackend` at a daemon:
//!   `OLLAMA_BASE_URL=http://localhost:11434 OLLAMA_MODEL=llama3 cargo run --example basic_agent`

use coccinella_agent_sdk::{Agent, Provider, Result, StaticBackend, Tool, ToolCall};

struct GreetTool;
impl Tool for GreetTool {
    fn name(&self) -> &'static str {
        "greet"
    }
    fn run(&self, call: &ToolCall) -> Result<String> {
        let who = call
            .arguments
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("world");
        Ok(format!("hello {who}"))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // One real local-backend path is exercised by default (need an Ollama daemon).
    if let Ok(url) = std::env::var("OLLAMA_BASE_URL") {
        let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "llama3".into());
        let backend = coccinella_agent_sdk::OllamaBackend::new(url, model);
        let mut agent = Agent::new(&backend, vec![Box::new(GreetTool)], 4);
        let reply = agent.run("greet Ada").await?;
        println!("ollama reply: {reply}");
        return Ok(());
    }

    // Offline fallback: scripted replies drive the same loop.
    let backend = StaticBackend::new(
        Provider::OpenAI,
        vec![
            r#"[{"function":{"name":"greet","arguments":"{\"name\":\"Ada\"}"}}]"#.into(),
            r#"{"content":"Added to the guest book. Anything else?"}"#.into(),
        ],
    );
    let mut agent = Agent::new(&backend, vec![Box::new(GreetTool)], 4);
    let reply = agent.run("greet Ada").await?;
    println!("final reply: {reply}");
    Ok(())
}