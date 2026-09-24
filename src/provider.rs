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

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt;
use std::sync::Mutex;

/// Supported AI model providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Provider {
    OpenAI,
    Sambanova,
    Gemini,
    Ollama,
    OpenRouter,
    Zen,
}

impl fmt::Display for Provider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Provider::OpenAI => "OpenAI",
            Provider::Sambanova => "Sambanova",
            Provider::Gemini => "Gemini",
            Provider::Ollama => "Ollama",
            Provider::OpenRouter => "OpenRouter",
            Provider::Zen => "Zen",
        };
        f.write_str(name)
    }
}

/// A single chat message exchanged with the model backend.
///
/// Wire shape is stable: `{ "role": string, "content": string }`. Backends
/// serialize this type directly; no provider-specific fields are hidden here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    /// Build a system-role message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
        }
    }

    /// Build a user-role message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }

    /// Build an assistant-role message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
        }
    }
}

/// Contract for a model backend. The `chat` method returns the raw reply
/// string, exactly matching the contract Harper's loop relies on, so
/// `parse_tool_calls` in `tool_call.rs` sees the same shapes it always has.
///
/// Implementations must return `Err(Error::Http)` for transport failures and
/// non-success HTTP statuses, and must not buffer or rewrite the success body
/// beyond reading it as UTF-8 text.
#[async_trait::async_trait]
pub trait ModelBackend: Send + Sync {
    /// Which provider this backend talks to.
    fn provider(&self) -> Provider;
    /// Run a chat completion and return the raw assistant reply.
    async fn chat(&self, messages: &[Message]) -> Result<String>;
}

/// Request body sent to Ollama `POST /api/chat`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub stream: bool,
}

/// Map an HTTP status plus response body onto `Result<String>`.
///
/// Success statuses return the body unchanged. Anything else becomes
/// `Error::Http` with the URL and status so callers can branch on it.
pub(crate) fn map_http_response(url: &str, status: u16, body: String) -> Result<String> {
    if (200..300).contains(&status) {
        Ok(body)
    } else {
        Err(Error::Http(format!("{url}: HTTP {status}")))
    }
}

/// Local Ollama backend. Speaks the chat shape Ollama serves at
/// `/api/chat`: top-level or `message.tool_calls`. Non-2xx responses are
/// rejected with `Error::Http` before the body is treated as a reply.
pub struct OllamaBackend {
    base_url: String,
    model: String,
    client: reqwest::Client,
}

impl OllamaBackend {
    /// Create a backend pointed at an Ollama daemon.
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            model: model.into(),
            client: reqwest::Client::new(),
        }
    }

    /// The `model` name as configured.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The configured base URL (trailing slash preserved as given).
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Absolute `POST` endpoint for `/api/chat`, with trailing slashes on
    /// `base_url` collapsed so `http://host:11434/` and `http://host:11434`
    /// produce the same URL.
    pub fn chat_url(&self) -> String {
        format!("{}/api/chat", self.base_url.trim_end_matches('/'))
    }

    /// Build the wire request for a chat turn.
    pub fn chat_request(&self, messages: &[Message]) -> ChatRequest {
        ChatRequest {
            model: self.model.clone(),
            messages: messages.to_vec(),
            stream: false,
        }
    }
}

#[async_trait::async_trait]
impl ModelBackend for OllamaBackend {
    fn provider(&self) -> Provider {
        Provider::Ollama
    }

    async fn chat(&self, messages: &[Message]) -> Result<String> {
        let url = self.chat_url();
        let body = self.chat_request(messages);
        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Http(format!("{url}: {e}")))?;
        let status = resp.status().as_u16();
        let text = resp
            .text()
            .await
            .map_err(|e| Error::Http(format!("{url}: {e}")))?;
        map_http_response(&url, status, text)
    }
}

/// A scripted backend for tests and examples. Each `chat` call pops the next
/// reply; the last reply repeats once the queue empties. An empty reply list
/// yields `Ok("")` on every call. Never performs I/O.
pub struct StaticBackend {
    provider: Provider,
    state: Mutex<(VecDeque<String>, Option<String>)>,
}

impl StaticBackend {
    /// Create a backend that replays the given raw replies in order.
    pub fn new(provider: Provider, replies: Vec<String>) -> Self {
        Self {
            provider,
            state: Mutex::new((VecDeque::from(replies), None)),
        }
    }

    /// Number of scripted replies still queued (not counting the repeated last).
    pub fn remaining(&self) -> usize {
        self.state.lock().map(|state| state.0.len()).unwrap_or(0)
    }
}

#[async_trait::async_trait]
impl ModelBackend for StaticBackend {
    fn provider(&self) -> Provider {
        self.provider
    }

    async fn chat(&self, _messages: &[Message]) -> Result<String> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::Other("static backend poisoned".into()))?;
        let (queue, last) = &mut *state;
        if let Some(reply) = queue.pop_front() {
            *last = Some(reply.clone());
            return Ok(reply);
        }
        Ok(last.clone().unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_display_is_stable() {
        assert_eq!(Provider::OpenAI.to_string(), "OpenAI");
        assert_eq!(Provider::Sambanova.to_string(), "Sambanova");
        assert_eq!(Provider::Gemini.to_string(), "Gemini");
        assert_eq!(Provider::Ollama.to_string(), "Ollama");
        assert_eq!(Provider::OpenRouter.to_string(), "OpenRouter");
        assert_eq!(Provider::Zen.to_string(), "Zen");
    }

    #[test]
    fn message_helpers_set_role() {
        assert_eq!(
            Message::system("s"),
            Message {
                role: "system".into(),
                content: "s".into()
            }
        );
        assert_eq!(Message::user("u").role, "user");
        assert_eq!(Message::assistant("a").role, "assistant");
    }

    #[test]
    fn message_wire_shape_is_role_and_content() {
        let msg = Message::user("hello");
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["role"], "user");
        assert_eq!(json["content"], "hello");
        let round: Message = serde_json::from_value(json).unwrap();
        assert_eq!(round, msg);
    }

    #[test]
    fn chat_request_shape_is_explicit() {
        let backend = OllamaBackend::new("http://localhost:11434/", "llama3");
        let req = backend.chat_request(&[Message::user("hi")]);
        assert_eq!(req.model, "llama3");
        assert!(!req.stream);
        assert_eq!(req.messages.len(), 1);
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["stream"], false);
        assert_eq!(json["messages"][0]["role"], "user");
    }

    #[test]
    fn ollama_chat_url_collapses_trailing_slash() {
        let with = OllamaBackend::new("http://localhost:11434/", "llama3");
        let without = OllamaBackend::new("http://localhost:11434", "llama3");
        assert_eq!(with.chat_url(), "http://localhost:11434/api/chat");
        assert_eq!(without.chat_url(), "http://localhost:11434/api/chat");
        assert_eq!(with.provider(), Provider::Ollama);
        assert_eq!(with.model(), "llama3");
        assert_eq!(with.base_url(), "http://localhost:11434/");
    }

    #[test]
    fn map_http_response_accepts_2xx_and_rejects_rest() {
        let url = "http://localhost:11434/api/chat";
        assert_eq!(map_http_response(url, 200, "ok".into()).unwrap(), "ok");
        assert_eq!(map_http_response(url, 204, String::new()).unwrap(), "");
        let err = map_http_response(url, 404, "missing".into()).unwrap_err();
        assert!(matches!(&err, Error::Http(msg) if msg.contains("404") && msg.contains(url)));
        let err = map_http_response(url, 500, "boom".into()).unwrap_err();
        assert!(matches!(&err, Error::Http(msg) if msg.contains("500")));
    }

    #[tokio::test]
    async fn static_backend_cycles_and_reports_remaining() {
        let backend = StaticBackend::new(
            Provider::Ollama,
            vec![
                "first".to_string(),
                "second".to_string(),
                "last".to_string(),
            ],
        );
        assert_eq!(backend.remaining(), 3);
        assert_eq!(backend.chat(&[]).await.unwrap(), "first");
        assert_eq!(backend.remaining(), 2);
        assert_eq!(backend.chat(&[]).await.unwrap(), "second");
        assert_eq!(backend.chat(&[]).await.unwrap(), "last");
        assert_eq!(backend.remaining(), 0);
        assert_eq!(backend.chat(&[]).await.unwrap(), "last");
        assert_eq!(backend.provider(), Provider::Ollama);
    }

    #[tokio::test]
    async fn static_backend_empty_script_returns_empty_string() {
        let backend = StaticBackend::new(Provider::OpenAI, vec![]);
        assert_eq!(backend.chat(&[Message::user("hi")]).await.unwrap(), "");
        assert_eq!(backend.remaining(), 0);
    }

    #[tokio::test]
    async fn static_backend_error_is_other_when_poisoned() {
        let backend = StaticBackend::new(Provider::OpenAI, vec!["x".into()]);
        // Poison the mutex by panicking while holding it in another scope.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = backend.state.lock().unwrap();
            panic!("poison");
        }));
        let err = backend.chat(&[]).await.unwrap_err();
        assert!(matches!(err, Error::Other(_)));
    }
}
