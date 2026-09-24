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

//! Offline-first walkthrough: a scripted `StaticBackend` drives two tools
//! through the agent loop, then the final reply and history are printed.
//! Runs with no network.
//!
//! Point the same agent at a local Ollama daemon instead:
//!   `OLLAMA_BASE_URL=http://localhost:11434 OLLAMA_MODEL=llama3 cargo run --example basic_agent`

use coccinella_agent_sdk::{arg_str, Agent, Provider, Result, StaticBackend, Tool, ToolCall};

struct GreetTool;

impl Tool for GreetTool {
    fn name(&self) -> &'static str {
        "greet"
    }

    fn run(&self, call: &ToolCall) -> Result<String> {
        let who = arg_str(call, "name")?;
        Ok(format!("hello {who}"))
    }
}

struct RememberTool;

impl Tool for RememberTool {
    fn name(&self) -> &'static str {
        "remember"
    }

    fn run(&self, call: &ToolCall) -> Result<String> {
        let note = arg_str(call, "note")?;
        Ok(format!("noted: {note}"))
    }
}

fn tools() -> Vec<Box<dyn Tool>> {
    vec![Box::new(GreetTool), Box::new(RememberTool)]
}

#[tokio::main]
async fn main() -> Result<()> {
    if let Ok(url) = std::env::var("OLLAMA_BASE_URL") {
        let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "llama3".into());
        let backend = coccinella_agent_sdk::OllamaBackend::new(url, model);
        let mut agent = Agent::new(&backend, tools(), 4);
        let reply = agent
            .run("greet Ada and remember that she prefers tea")
            .await?;
        println!("ollama reply: {reply}");
        print_history(&agent);
        return Ok(());
    }

    let backend = StaticBackend::new(
        Provider::OpenAI,
        vec![
            r#"[{"id":"c1","function":{"name":"greet","arguments":"{\"name\":\"Ada\"}"}},{"id":"c2","function":{"name":"remember","arguments":"{\"note\":\"prefers tea\"}"}}]"#.into(),
            r#"{"content":"Noted. Anything else?"}"#.into(),
        ],
    );
    let mut agent = Agent::new(&backend, tools(), 4);
    let reply = agent
        .run("greet Ada and remember that she prefers tea")
        .await?;
    println!("final reply: {reply}");
    println!("script remaining: {}", backend.remaining());
    print_history(&agent);
    Ok(())
}

fn print_history(agent: &Agent<'_>) {
    println!("history ({} turns):", agent.history().len());
    for message in agent.history() {
        let preview: String = message.content.chars().take(60).collect();
        println!("  [{}] {preview}", message.role);
    }
}
