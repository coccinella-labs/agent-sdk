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

//! Minimal reusable agent runtime: tool-call parsing, provider backends,
//! tool dispatch, and a tiny fixed-budget loop.
//!
//! # v0.1 API surface
//!
//! Import from the crate root. The modules below stay public for qualified
//! paths; the root re-exports are the intended surface.
//!
//! | Area | Types |
//! | --- | --- |
//! | Loop | [`Agent`] |
//! | Tool calls | [`ToolCall`], [`ToolCallSource`], [`parse_tool_calls`] |
//! | Tools | [`Tool`], [`dispatch`], [`arg_str`], [`arg_object`] |
//! | Backends | [`ModelBackend`], [`StaticBackend`], [`OllamaBackend`], [`ChatRequest`] |
//! | Wire types | [`Message`], [`Provider`] |
//! | Errors | [`Error`], [`Result`] |
//!
//! # Adopt
//!
//! 1. Implement [`Tool`] for each tool the model may call.
//! 2. Pick a [`ModelBackend`]: [`StaticBackend`] (offline, scripted) or
//!    [`OllamaBackend`] (local HTTP daemon).
//! 3. Build an [`Agent`] and call [`Agent::run`].
//! 4. Branch on [`Error`] when `run` returns `Err`.
//!
//! A complete offline walkthrough lives in `examples/basic_agent.rs` and in
//! the crate README.
//!
//! # Stability
//!
//! v0.1 freezes the table above, the `Message` wire shape
//! (`{ "role", "content" }`), the `ChatRequest` Ollama body, and the
//! contracts documented on [`Agent`], [`Tool`], [`ModelBackend`], and
//! [`Error`]. Parser edge cases and `pub(crate)` helpers may change without
//! a major bump.

pub mod agent;
pub mod error;
pub mod provider;
pub mod tool;
pub mod tool_call;

pub use agent::Agent;
pub use error::Error;
pub use provider::{ChatRequest, Message, ModelBackend, OllamaBackend, Provider, StaticBackend};
pub use tool::{arg_object, arg_str, dispatch, Tool};
pub use tool_call::{parse_tool_calls, ToolCall, ToolCallSource};

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, Error>;
