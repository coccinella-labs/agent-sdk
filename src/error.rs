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

use std::string::FromUtf8Error;

/// Errors surfaced by the SDK.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The raw reply could not be handled as a tool call or message.
    #[error("unparseable model reply: {0}")]
    Parse(String),
    /// A model request failed at the transport layer or returned a non-2xx
    /// HTTP status. The payload includes the request URL and, for HTTP
    /// failures, `HTTP <status>`.
    #[error("model request failed: {0}")]
    Http(String),
    /// Dispatch received a tool call with no registered tool.
    #[error("unknown tool: {0}")]
    UnknownTool(String),
    /// Container for anything else.
    #[error("{0}")]
    Other(String),
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Parse(e.to_string())
    }
}

impl From<FromUtf8Error> for Error {
    fn from(e: FromUtf8Error) -> Self {
        Error::Other(e.to_string())
    }
}
