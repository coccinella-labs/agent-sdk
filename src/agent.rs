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

use crate::provider::{Message, ModelBackend};
use crate::tool::dispatch;
use crate::{parse_tool_calls, Result, Tool, ToolCallSource};

/// A deliberately small agent loop: prompt the backend, parse tool calls,
/// dispatch them, feed results back, repeat up to `max_rounds`.
///
/// This exists only to be independently testable. Harper keeps its own richer
/// loop; the SDK loop is a reference shape, not a rewrite target.
pub struct Agent<'a> {
    backend: &'a dyn ModelBackend,
    tools: Vec<Box<dyn Tool>>,
    history: Vec<Message>,
    source: ToolCallSource,
    max_rounds: usize,
}

impl<'a> Agent<'a> {
    /// Create an agent with an explicit parse source.
    pub fn with_source(
        backend: &'a dyn ModelBackend,
        tools: Vec<Box<dyn Tool>>,
        source: ToolCallSource,
        max_rounds: usize,
    ) -> Self {
        Self {
            backend,
            tools,
            history: Vec::new(),
            source,
            max_rounds,
        }
    }

    /// Create an agent whose parse source is derived from the backend
    /// provider (e.g. `Ollama` for a local daemon).
    pub fn new(backend: &'a dyn ModelBackend, tools: Vec<Box<dyn Tool>>, max_rounds: usize) -> Self {
        let provider = backend.provider();
        let source = ToolCallSource::from_provider(&provider);
        Self::with_source(backend, tools, source, max_rounds)
    }

    /// Run the loop against a user prompt and return the final reply.
    pub async fn run(&mut self, user_prompt: &str) -> Result<String> {
        self.history.push(Message {
            role: "user".into(),
            content: user_prompt.into(),
        });

        for round in 0..self.max_rounds {
            let reply = self.backend.chat(&self.history).await?;
            let calls = parse_tool_calls(&reply, self.source);

            if calls.is_empty() {
                let text = reply.trim().to_string();
                self.history.push(Message {
                    role: "assistant".into(),
                    content: text.clone(),
                });
                return Ok(text);
            }

            let mut outputs = Vec::new();
            for call in calls {
                match dispatch(&call, &self.tools) {
                    Ok(out) => outputs.push(format!("[{}] {}", call.name, out)),
                    Err(e) => outputs.push(format!("[{}] error: {e}", call.name)),
                }
            }

            if round + 1 == self.max_rounds {
                let msg = format!(
                    "Reached max rounds ({}); last tool outputs:\n{}",
                    self.max_rounds,
                    outputs.join("\n")
                );
                self.history.push(Message {
                    role: "assistant".into(),
                    content: msg.clone(),
                });
                return Ok(msg);
            }

            self.history.push(Message {
                role: "assistant".into(),
                content: reply,
            });
            self.history.push(Message {
                role: "tool".into(),
                content: outputs.join("\n"),
            });
        }

        Ok("reached max rounds".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OllamaBackend, Provider, StaticBackend, ToolCall};

    struct EchoTool;
    impl Tool for EchoTool {
        fn name(&self) -> &'static str {
            "echo"
        }
        fn run(&self, call: &ToolCall) -> Result<String> {
            Ok(format!("said:{}", call.arguments.get("message").unwrap()))
        }
    }

    #[tokio::test]
    async fn loop_returns_final_text_when_no_more_tools() {
        let backend = StaticBackend::new(
            Provider::OpenAI,
            vec![
                r#"[{"function":{"name":"echo","arguments":"{\"message\":\"hi\"}"}}]"#.into(),
                "All done.".into(),
            ],
        );
        let mut agent = Agent::with_source(
            &backend,
            vec![Box::new(EchoTool)],
            ToolCallSource::OpenAi,
            4,
        );
        let reply = agent.run("hello").await.unwrap();
        assert_eq!(reply, "All done.");
    }

    #[tokio::test]
    async fn loop_stops_at_max_rounds() {
        let backend = StaticBackend::new(
            Provider::OpenAI,
            vec![
                r#"[{"function":{"name":"echo","arguments":"{\"message\":\"a\"}"}}]"#.into(),
                r#"[{"function":{"name":"echo","arguments":"{\"message\":\"b\"}"}}]"#.into(),
            ],
        );
        let mut agent = Agent::with_source(
            &backend,
            vec![Box::new(EchoTool)],
            ToolCallSource::OpenAi,
            1,
        );
        let reply = agent.run("hello").await.unwrap();
        assert!(reply.contains("Reached max rounds (1)"));
    }

    #[tokio::test]
    async fn agent_source_derives_from_provider() {
        let backend = OllamaBackend::new("http://localhost:11434", "llama3");
        let source = ToolCallSource::from_provider(&backend.provider());
        assert_eq!(source, ToolCallSource::Ollama);
    }
}