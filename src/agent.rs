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
///
/// # Termination
///
/// [`run`](Agent::run) returns `Ok(final_text)` on every non-backend path:
///
/// * **Final reply**: a model response that parses to zero tool calls is
///   trimmed and returned as the assistant answer. Plain text and malformed
///   JSON both land here (`parse_tool_calls` yields an empty vec).
/// * **Max rounds**: on the last allowed round, tool outputs are not fed
///   back. The return value is `Reached max rounds (N); last tool outputs:`
///   followed by one line per call (`[name] output` or `[name] error: ...`).
/// * **Backend failure**: `chat` errors propagate as `Err` unchanged.
/// * **`max_rounds == 0`**: returns `Err(Error::Other)` without calling the
///   backend. Use at least 1.
///
/// # Continuation
///
/// Each non-final round appends the raw assistant reply (`role=assistant`)
/// and the joined tool outputs (`role=tool`) to history, then calls the
/// backend again. Parsed `run_command` calls are normalized before dispatch
/// (same candidate cleanup as `ToolCall::normalize_run_command`).
///
/// # Degradation
///
/// Unknown tools and tool failures never abort the loop. They become a
/// `[name] error: ...` line in the tool message so the model can recover on
/// the next round. Argument-shape errors surface the same way.
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
    pub fn new(
        backend: &'a dyn ModelBackend,
        tools: Vec<Box<dyn Tool>>,
        max_rounds: usize,
    ) -> Self {
        let provider = backend.provider();
        let source = ToolCallSource::from_provider(&provider);
        Self::with_source(backend, tools, source, max_rounds)
    }

    /// Messages exchanged so far (user, assistant, and tool turns).
    pub fn history(&self) -> &[Message] {
        &self.history
    }

    /// Configured round budget.
    pub fn max_rounds(&self) -> usize {
        self.max_rounds
    }

    /// Run the loop against a user prompt and return the final reply.
    ///
    /// See the type-level docs for termination, continuation, and degradation
    /// contracts. Backend transport/HTTP errors are the only `Err` paths
    /// besides `max_rounds == 0`.
    pub async fn run(&mut self, user_prompt: &str) -> Result<String> {
        if self.max_rounds == 0 {
            return Err(crate::Error::Other("max_rounds must be at least 1".into()));
        }

        self.history.push(Message::user(user_prompt));

        for round in 0..self.max_rounds {
            let reply = self.backend.chat(&self.history).await?;
            let mut calls = parse_tool_calls(&reply, self.source);
            for call in &mut calls {
                call.normalize_run_command();
            }

            if calls.is_empty() {
                let text = reply.trim().to_string();
                self.history.push(Message::assistant(text.clone()));
                return Ok(text);
            }

            let mut outputs = Vec::new();
            for call in &calls {
                match dispatch(call, &self.tools) {
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
                self.history.push(Message::assistant(msg.clone()));
                return Ok(msg);
            }

            self.history.push(Message::assistant(reply));
            self.history.push(Message {
                role: "tool".into(),
                content: outputs.join("\n"),
            });
        }

        // Unreachable when max_rounds >= 1: every iteration returns.
        Ok("reached max rounds".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Error, OllamaBackend, Provider, StaticBackend, ToolCall};
    use serde_json::json;

    struct EchoTool;
    impl Tool for EchoTool {
        fn name(&self) -> &'static str {
            "echo"
        }
        fn run(&self, call: &ToolCall) -> Result<String> {
            let message = crate::arg_str(call, "message").unwrap_or("?");
            Ok(format!("said:{message}"))
        }
    }

    struct FailTool;
    impl Tool for FailTool {
        fn name(&self) -> &'static str {
            "fail"
        }
        fn run(&self, _call: &ToolCall) -> Result<String> {
            Err(Error::Other("boom".into()))
        }
    }

    fn openai_tool_call(name: &str, args_json: &str) -> String {
        format!(r#"[{{"id":"c1","function":{{"name":"{name}","arguments":{args_json}}}}}]"#)
    }

    #[tokio::test]
    async fn loop_returns_final_text_when_no_more_tools() {
        let backend = StaticBackend::new(
            Provider::OpenAI,
            vec![
                openai_tool_call("echo", r#" "{\"message\":\"hi\"}" "#),
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
        let roles: Vec<&str> = agent.history().iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, ["user", "assistant", "tool", "assistant"]);
        assert!(agent.history()[2].content.contains("[echo] said:hi"));
        assert_eq!(agent.max_rounds(), 4);
    }

    #[tokio::test]
    async fn loop_stops_at_max_rounds_with_tool_outputs() {
        let backend = StaticBackend::new(
            Provider::OpenAI,
            vec![
                openai_tool_call("echo", r#" "{\"message\":\"a\"}" "#),
                openai_tool_call("echo", r#" "{\"message\":\"b\"}" "#),
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
        assert!(reply.contains("[echo] said:a"));
        // Tool message is not appended on the final round.
        let roles: Vec<&str> = agent.history().iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, ["user", "assistant"]);
    }

    #[tokio::test]
    async fn loop_zero_max_rounds_errors_without_backend_call() {
        let backend = StaticBackend::new(Provider::OpenAI, vec!["never".into()]);
        let mut agent = Agent::with_source(&backend, vec![], ToolCallSource::OpenAi, 0);
        let err = agent.run("hello").await.unwrap_err();
        assert!(matches!(err, Error::Other(msg) if msg.contains("max_rounds")));
        assert!(agent.history().is_empty());
        assert_eq!(backend.remaining(), 1);
    }

    #[tokio::test]
    async fn loop_treats_plain_text_as_final_reply() {
        let backend = StaticBackend::new(Provider::OpenAI, vec!["  Just a plain answer.  ".into()]);
        let mut agent = Agent::with_source(
            &backend,
            vec![Box::new(EchoTool)],
            ToolCallSource::OpenAi,
            3,
        );
        let reply = agent.run("hi").await.unwrap();
        assert_eq!(reply, "Just a plain answer.");
        assert_eq!(agent.history().len(), 2);
        assert_eq!(agent.history()[1].role, "assistant");
    }

    #[tokio::test]
    async fn loop_treats_malformed_json_without_tool_calls_as_final() {
        // Valid JSON object but not a tool-call shape: parse yields empty.
        let backend = StaticBackend::new(
            Provider::OpenAI,
            vec![r#"{"content":"I cannot do that"}"#.into()],
        );
        let mut agent = Agent::with_source(
            &backend,
            vec![Box::new(EchoTool)],
            ToolCallSource::OpenAi,
            3,
        );
        let reply = agent.run("hi").await.unwrap();
        assert_eq!(reply, r#"{"content":"I cannot do that"}"#);
    }

    #[tokio::test]
    async fn loop_degrades_unknown_tool_and_continues() {
        let backend = StaticBackend::new(
            Provider::OpenAI,
            vec![
                openai_tool_call("nope", r#" {} "#),
                "Recovered after unknown tool.".into(),
            ],
        );
        let mut agent = Agent::with_source(
            &backend,
            vec![Box::new(EchoTool)],
            ToolCallSource::OpenAi,
            4,
        );
        let reply = agent.run("hello").await.unwrap();
        assert_eq!(reply, "Recovered after unknown tool.");
        let tool_msg = agent
            .history()
            .iter()
            .find(|m| m.role == "tool")
            .expect("tool message");
        assert!(tool_msg
            .content
            .contains("[nope] error: unknown tool: nope"));
    }

    #[tokio::test]
    async fn loop_degrades_tool_error_and_continues() {
        let backend = StaticBackend::new(
            Provider::OpenAI,
            vec![
                openai_tool_call("fail", r#" {} "#),
                "Recovered after tool failure.".into(),
            ],
        );
        let mut agent = Agent::with_source(
            &backend,
            vec![Box::new(FailTool)],
            ToolCallSource::OpenAi,
            4,
        );
        let reply = agent.run("hello").await.unwrap();
        assert_eq!(reply, "Recovered after tool failure.");
        let tool_msg = agent
            .history()
            .iter()
            .find(|m| m.role == "tool")
            .expect("tool message");
        assert!(tool_msg.content.contains("[fail] error: boom"));
    }

    #[tokio::test]
    async fn loop_normalizes_run_command_before_dispatch() {
        struct CaptureCommand;
        impl Tool for CaptureCommand {
            fn name(&self) -> &'static str {
                "run_command"
            }
            fn run(&self, call: &ToolCall) -> Result<String> {
                Ok(crate::arg_str(call, "command")
                    .unwrap_or("missing")
                    .to_string())
            }
        }

        let backend = StaticBackend::new(
            Provider::OpenAI,
            vec![
                openai_tool_call("run_command", r#" "{\"command\":\"the git status\"}" "#),
                "done".into(),
            ],
        );
        let mut agent = Agent::with_source(
            &backend,
            vec![Box::new(CaptureCommand)],
            ToolCallSource::OpenAi,
            4,
        );
        let reply = agent.run("status please").await.unwrap();
        assert_eq!(reply, "done");
        let tool_msg = agent
            .history()
            .iter()
            .find(|m| m.role == "tool")
            .expect("tool message");
        assert!(tool_msg.content.contains("[run_command] git status"));
        assert!(!tool_msg.content.contains("the git status"));
    }

    #[tokio::test]
    async fn loop_propagates_backend_error() {
        // Point at a closed local port to force a transport failure.
        let backend = crate::OllamaBackend::new("http://127.0.0.1:9", "llama3");
        let mut agent = Agent::with_source(
            &backend,
            vec![Box::new(EchoTool)],
            ToolCallSource::OpenAi,
            2,
        );
        let err = agent.run("hello").await.unwrap_err();
        assert!(matches!(err, Error::Http(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn agent_source_derives_from_provider() {
        let backend = OllamaBackend::new("http://localhost:11434", "llama3");
        let source = ToolCallSource::from_provider(&backend.provider());
        assert_eq!(source, ToolCallSource::Ollama);
    }

    #[tokio::test]
    async fn loop_multi_round_feeds_tool_results_back() {
        let backend = StaticBackend::new(
            Provider::OpenAI,
            vec![
                openai_tool_call("echo", r#" "{\"message\":\"one\"}" "#),
                openai_tool_call("echo", r#" "{\"message\":\"two\"}" "#),
                "final".into(),
            ],
        );
        let mut agent = Agent::with_source(
            &backend,
            vec![Box::new(EchoTool)],
            ToolCallSource::OpenAi,
            5,
        );
        let reply = agent.run("go").await.unwrap();
        assert_eq!(reply, "final");
        let roles: Vec<&str> = agent.history().iter().map(|m| m.role.as_str()).collect();
        assert_eq!(
            roles,
            [
                "user",
                "assistant",
                "tool",
                "assistant",
                "tool",
                "assistant"
            ]
        );
        assert_eq!(backend.remaining(), 0);
    }

    #[test]
    fn tool_call_normalize_strips_article() {
        let mut call = ToolCall {
            id: None,
            name: "run_command".into(),
            arguments: json!({"command": "the git status"}),
            source: crate::ToolCallSource::OpenAi,
        };
        call.normalize_run_command();
        assert_eq!(call.arguments["command"], "git status");
    }
}
