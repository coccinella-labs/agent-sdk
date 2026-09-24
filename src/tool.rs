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
use serde_json::Value;

/// A tool the agent can invoke.
///
/// Contract for v0.1:
///
/// * `name` is a stable identifier. `dispatch` matches it against
///   `ToolCall::name` (exact string equality). Duplicate names in one
///   registry are undefined for callers: the first match wins.
/// * `run` receives the full parsed call. Arguments are a `serde_json::Value`
///   (object, array, string, number, bool, or null). Use [`arg_str`] and
///   [`arg_object`] for shape-checked access, or read `call.arguments`
///   directly.
/// * On success, return the tool output as a UTF-8 `String`. The agent loop
///   feeds that string back to the model as a tool result; it is not parsed
///   further by the SDK.
/// * On failure, return `Err`. Prefer [`Error::InvalidArguments`] for
///   malformed input and [`Error::Other`] for execution failures. Unknown
///   names never reach `run`; [`dispatch`] returns [`Error::UnknownTool`]
///   first.
/// * The surface is only `ToolCall` plus `serde_json`. No Harper types are
///   part of this trait.
pub trait Tool: Send + Sync {
    /// The tool name used by the model.
    fn name(&self) -> &'static str;
    /// Execute the tool for a parsed tool call.
    fn run(&self, call: &ToolCall) -> Result<String>;
}

/// Dispatch one tool call to the first tool whose `name` matches.
///
/// Failure paths, in order:
///
/// 1. No matching name -> [`Error::UnknownTool`] with the call's name.
/// 2. Matching tool returns `Err` -> that error is propagated unchanged.
/// 3. Matching tool returns `Ok` -> the output string is returned as-is.
///
/// Duplicate names: the earliest entry in `registry` wins. Arguments are not
/// validated by `dispatch`; shape checks belong to the tool.
pub fn dispatch(call: &ToolCall, registry: &[Box<dyn Tool>]) -> Result<String> {
    for tool in registry {
        if tool.name() == call.name {
            return tool.run(call);
        }
    }
    Err(Error::UnknownTool(call.name.clone()))
}

/// Read a required string argument from `call.arguments`.
///
/// Fails with [`Error::InvalidArguments`] when the key is missing or is not
/// a JSON string. Non-object argument payloads (null, array, scalar) also
/// fail as missing keys.
pub fn arg_str<'call>(call: &'call ToolCall, key: &str) -> Result<&'call str> {
    let value = call
        .arguments
        .get(key)
        .ok_or_else(|| invalid_args(call, format!("missing string argument `{key}`")))?;
    value
        .as_str()
        .ok_or_else(|| invalid_args(call, format!("argument `{key}` is not a string")))
}

/// Read `call.arguments` as a JSON object.
///
/// Fails with [`Error::InvalidArguments`] when the payload is not an object
/// (including `null` and arrays).
pub fn arg_object(call: &ToolCall) -> Result<&serde_json::Map<String, Value>> {
    call.arguments
        .as_object()
        .ok_or_else(|| invalid_args(call, "arguments must be a JSON object".into()))
}

fn invalid_args(call: &ToolCall, message: String) -> Error {
    Error::InvalidArguments {
        tool: call.name.clone(),
        message,
    }
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

    struct FailTool;
    impl Tool for FailTool {
        fn name(&self) -> &'static str {
            "fail"
        }
        fn run(&self, _call: &ToolCall) -> Result<String> {
            Err(Error::Other("boom".into()))
        }
    }

    struct GreetTool;
    impl Tool for GreetTool {
        fn name(&self) -> &'static str {
            "greet"
        }
        fn run(&self, call: &ToolCall) -> Result<String> {
            let name = arg_str(call, "name")?;
            Ok(format!("hello {name}"))
        }
    }

    struct FirstEcho;
    impl Tool for FirstEcho {
        fn name(&self) -> &'static str {
            "echo"
        }
        fn run(&self, _call: &ToolCall) -> Result<String> {
            Ok("first".into())
        }
    }

    struct SecondEcho;
    impl Tool for SecondEcho {
        fn name(&self) -> &'static str {
            "echo"
        }
        fn run(&self, _call: &ToolCall) -> Result<String> {
            Ok("second".into())
        }
    }

    fn call(name: &str, arguments: Value) -> ToolCall {
        ToolCall {
            id: None,
            name: name.into(),
            arguments,
            source: ToolCallSource::OpenAi,
        }
    }

    #[test]
    fn dispatch_round_trips_matching_tool() {
        let out = dispatch(
            &call("echo", json!({"message": "hi"})),
            &[Box::new(EchoTool)],
        )
        .unwrap();
        assert_eq!(out, "echo:{\"message\":\"hi\"}");
    }

    #[test]
    fn dispatch_errors_on_unregistered_tool() {
        assert!(matches!(
            dispatch(&call("nope", json!({})), &[]),
            Err(Error::UnknownTool(name)) if name == "nope"
        ));
    }

    #[test]
    fn dispatch_selects_by_name_in_multi_tool_registry() {
        let registry: Vec<Box<dyn Tool>> = vec![Box::new(EchoTool), Box::new(GreetTool)];
        let echo = dispatch(&call("echo", json!({"message": "x"})), &registry).unwrap();
        assert_eq!(echo, "echo:{\"message\":\"x\"}");
        let greet = dispatch(&call("greet", json!({"name": "Ada"})), &registry).unwrap();
        assert_eq!(greet, "hello Ada");
    }

    #[test]
    fn dispatch_unknown_among_registered_tools() {
        let registry: Vec<Box<dyn Tool>> = vec![Box::new(EchoTool), Box::new(GreetTool)];
        assert!(matches!(
            dispatch(&call("missing", json!({})), &registry),
            Err(Error::UnknownTool(name)) if name == "missing"
        ));
    }

    #[test]
    fn dispatch_first_duplicate_name_wins() {
        let registry: Vec<Box<dyn Tool>> = vec![Box::new(FirstEcho), Box::new(SecondEcho)];
        let out = dispatch(&call("echo", json!({})), &registry).unwrap();
        assert_eq!(out, "first");
    }

    #[test]
    fn dispatch_propagates_tool_error() {
        let err = dispatch(&call("fail", json!({})), &[Box::new(FailTool)]).unwrap_err();
        assert!(matches!(err, Error::Other(msg) if msg == "boom"));
    }

    #[test]
    fn dispatch_does_not_validate_arguments_shape() {
        let registry: Vec<Box<dyn Tool>> = vec![Box::new(EchoTool)];
        let out = dispatch(&call("echo", json!("not-an-object")), &registry).unwrap();
        assert_eq!(out, "echo:\"not-an-object\"");
    }

    #[test]
    fn arg_str_reads_string_and_fails_on_shape() {
        let ok = call("greet", json!({"name": "Ada"}));
        assert_eq!(arg_str(&ok, "name").unwrap(), "Ada");

        let missing = call("greet", json!({}));
        assert!(matches!(
            arg_str(&missing, "name"),
            Err(Error::InvalidArguments { tool, message })
                if tool == "greet" && message.contains("missing")
        ));

        let wrong_type = call("greet", json!({"name": 42}));
        assert!(matches!(
            arg_str(&wrong_type, "name"),
            Err(Error::InvalidArguments { message, .. })
                if message.contains("not a string")
        ));

        let null_args = call("greet", Value::Null);
        assert!(matches!(
            arg_str(&null_args, "name"),
            Err(Error::InvalidArguments { .. })
        ));
    }

    #[test]
    fn arg_object_requires_json_object() {
        let ok = call("greet", json!({"name": "Ada"}));
        assert_eq!(arg_object(&ok).unwrap().len(), 1);

        let arr = call("greet", json!([1, 2]));
        assert!(matches!(
            arg_object(&arr),
            Err(Error::InvalidArguments { message, .. })
                if message.contains("JSON object")
        ));
    }

    #[test]
    fn tool_run_invalid_arguments_surface_from_dispatch() {
        let registry: Vec<Box<dyn Tool>> = vec![Box::new(GreetTool)];
        let err = dispatch(&call("greet", json!({"name": 7})), &registry).unwrap_err();
        assert!(matches!(err, Error::InvalidArguments { tool, .. } if tool == "greet"));
    }
}
