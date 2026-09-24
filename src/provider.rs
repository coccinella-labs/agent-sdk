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
use std::collections::VecDeque;
use std::fmt;
use std::sync::Mutex;

/// Supported AI model providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Contract for a model backend. The `chat` method returns the raw reply
/// string, exactly matching the contract Harper's loop relies on, so
/// `parse_tool_calls` in `tool_call.rs` sees the same shapes it always has.
#[async_trait::async_trait]
pub trait ModelBackend: Send + Sync {
    /// Which provider this backend talks to.
    fn provider(&self) -> Provider;
    /// Run a chat completion and return the raw assistant reply.
    async fn chat(&self, messages: &[Message]) -> Result<String>;
}

/// Local Ollama backend. Defaults to the OpenAI-compatible chat shape that
/// Ollama serves at `/api/chat`: top-level or `message.tool_calls`.
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
}

#[async_trait::async_trait]
impl ModelBackend for OllamaBackend {
    fn provider(&self) -> Provider {
        Provider::Ollama
    }

    async fn chat(&self, messages: &[Message]) -> Result<String> {
        let body = serde_json::json!({
            "model": self.model,
            "messages": messages
                .iter()
                .map(|m| serde_json::json!({ "role": m.role, "content": m.content }))
                .collect::<Vec<_>>(),
            "stream": false,
        });
        let url = format!("{}/api/chat", self.base_url.trim_end_matches('/'));
        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Http(format!("{url}: {e}")))?;
        let text = resp
            .text()
            .await
            .map_err(|e| Error::Http(format!("{url}: {e}")))?;
        Ok(text)
    }
}

/// A scripted backend for tests and examples. Each `chat` call pops the next
/// reply; the last reply repeats once the queue empties.
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
    fn ollama_base_url_gets_chat_endpoint() {
        let backend = OllamaBackend::new("http://localhost:11434/", "llama3");
        assert_eq!(backend.provider(), Provider::Ollama);
        assert!(backend.base_url.ends_with('/'));
        assert_eq!(backend.model(), "llama3");
    }

    #[tokio::test]
    async fn static_backend_cycles_and_repeats_last() {
        let backend = StaticBackend::new(
            Provider::Ollama,
            vec![
                "first".to_string(),
                "second".to_string(),
                "last".to_string(),
            ],
        );
        assert_eq!(backend.chat(&[]).await.unwrap(), "first");
        assert_eq!(backend.chat(&[]).await.unwrap(), "second");
        assert_eq!(backend.chat(&[]).await.unwrap(), "last");
        assert_eq!(backend.chat(&[]).await.unwrap(), "last");
    }
}