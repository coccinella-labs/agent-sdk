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

use crate::{Error, Result, ToolCall};

/// A tool the agent can invoke. `name` must match `ToolCall::name`.
pub trait Tool: Send + Sync {
    /// The tool name used by the model.
    fn name(&self) -> &'static str;
    /// Execute the tool for a parsed tool call.
    fn run(&self, call: &ToolCall) -> Result<String>;
}

/// Dispatch one tool call to the first tool whose `name` matches.
pub fn dispatch(call: &ToolCall, registry: &[Box<dyn Tool>]) -> Result<String> {
    for tool in registry {
        if tool.name() == call.name {
            return tool.run(call);
        }
    }
    Err(Error::UnknownTool(call.name.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_call::ToolCallSource;
    use serde_json::json;

    struct EchoTool;
    impl Tool for EchoTool {
        fn name(&self) -> &'static str {
            "echo"
        }
        fn run(&self, call: &ToolCall) -> Result<String> {
            Ok(format!("echo:{}", call.arguments))
        }
    }

    #[test]
    fn dispatch_round_trips_matching_tool() {
        let call = ToolCall {
            id: None,
            name: "echo".into(),
            arguments: json!({"message": "hi"}),
            source: ToolCallSource::OpenAi,
        };
        let out = dispatch(&call, &[Box::new(EchoTool)]).unwrap();
        assert_eq!(out, "echo:{\"message\":\"hi\"}");
    }

    #[test]
    fn dispatch_errors_on_unregistered_tool() {
        let call = ToolCall {
            id: None,
            name: "nope".into(),
            arguments: json!({}),
            source: ToolCallSource::OpenAi,
        };
        assert!(matches!(dispatch(&call, &[]), Err(Error::UnknownTool(_))));
    }
}
