//! Backend-neutral view of a chat completion, shared by the OpenAI guards and
//! the Anthropic/Responses translators.
//!
//! Text and reasoning deltas pass through as they arrive. Tool calls are
//! buffered until the stream ends so identical consecutive calls can be
//! collapsed (M0: Qwen3.6 repeated one call 7×, `docs/spikes/m0-gpu.md`).

use std::collections::BTreeMap;

use serde_json::{Value, json};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// JSON-encoded arguments, as produced by the model.
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Delta {
    Text(String),
    Reasoning(String),
}

/// How a completion ended.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Finish {
    /// OpenAI `finish_reason`: `stop`, `length`, `tool_calls`, …
    pub reason: Option<String>,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub tool_calls: Vec<ToolCall>,
    /// Identical consecutive tool calls that were dropped.
    pub collapsed: usize,
}

/// A whole (non-streamed) completion.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Completion {
    pub text: String,
    pub reasoning: String,
    pub finish: Finish,
}

impl Completion {
    /// From an OpenAI `chat.completion` object (first choice).
    pub fn from_openai(resp: &Value) -> Self {
        let choice = &resp["choices"][0];
        let msg = &choice["message"];
        let calls = msg["tool_calls"]
            .as_array()
            .map(|a| a.iter().enumerate().map(|(i, c)| tool_call(c, i)).collect())
            .unwrap_or_default();
        let (tool_calls, collapsed) = collapse_repeats(calls);
        Self {
            text: msg["content"].as_str().unwrap_or_default().to_owned(),
            reasoning: reasoning_of(msg).unwrap_or_default().to_owned(),
            finish: Finish {
                reason: choice["finish_reason"].as_str().map(str::to_owned),
                prompt_tokens: resp["usage"]["prompt_tokens"].as_u64(),
                completion_tokens: resp["usage"]["completion_tokens"].as_u64(),
                tool_calls,
                collapsed,
            },
        }
    }
}

/// Incremental parser for OpenAI `chat.completion.chunk` streams.
#[derive(Debug, Default)]
pub struct StreamNormalizer {
    calls: BTreeMap<u64, ToolCall>,
    finish: Finish,
    /// `id`, `model` and `created` of the first chunk, for re-emitting chunks.
    pub meta: Option<(Value, Value, Value)>,
}

impl StreamNormalizer {
    pub fn push(&mut self, chunk: &Value) -> Vec<Delta> {
        if self.meta.is_none() {
            self.meta = Some((chunk["id"].clone(), chunk["model"].clone(), chunk["created"].clone()));
        }
        if let Some(u) = chunk.get("usage").filter(|u| u.is_object()) {
            self.finish.prompt_tokens = u["prompt_tokens"].as_u64().or(self.finish.prompt_tokens);
            self.finish.completion_tokens = u["completion_tokens"].as_u64().or(self.finish.completion_tokens);
        }
        let mut out = Vec::new();
        let Some(choice) = chunk["choices"].get(0) else { return out };
        if let Some(r) = choice["finish_reason"].as_str() {
            self.finish.reason = Some(r.to_owned());
        }
        let delta = &choice["delta"];
        if let Some(r) = reasoning_of(delta).filter(|s| !s.is_empty()) {
            out.push(Delta::Reasoning(r.to_owned()));
        }
        if let Some(t) = delta["content"].as_str().filter(|s| !s.is_empty()) {
            out.push(Delta::Text(t.to_owned()));
        }
        for (pos, tc) in delta["tool_calls"].as_array().into_iter().flatten().enumerate() {
            let index = tc["index"].as_u64().unwrap_or(pos as u64);
            let entry = self.calls.entry(index).or_insert_with(|| ToolCall {
                id: String::new(),
                name: String::new(),
                arguments: String::new(),
            });
            if let Some(id) = tc["id"].as_str().filter(|s| !s.is_empty()) {
                entry.id = id.to_owned();
            }
            if let Some(n) = tc["function"]["name"].as_str() {
                entry.name.push_str(n);
            }
            if let Some(a) = tc["function"]["arguments"].as_str() {
                entry.arguments.push_str(a);
            }
        }
        out
    }

    /// Call at end of stream: buffered tool calls (deduplicated) + finish info.
    pub fn finish(&mut self) -> Finish {
        let calls = std::mem::take(&mut self.calls)
            .into_values()
            .enumerate()
            .map(|(i, mut c)| {
                if c.id.is_empty() {
                    c.id = format!("call_{i}");
                }
                c
            })
            .collect();
        let (tool_calls, collapsed) = collapse_repeats(calls);
        let mut f = std::mem::take(&mut self.finish);
        f.tool_calls = tool_calls;
        f.collapsed = collapsed;
        f
    }
}

/// llama.cpp/FLM put reasoning in `reasoning_content`; some servers use `reasoning`.
fn reasoning_of(v: &Value) -> Option<&str> {
    v["reasoning_content"].as_str().or_else(|| v["reasoning"].as_str())
}

fn tool_call(c: &Value, i: usize) -> ToolCall {
    ToolCall {
        id: c["id"].as_str().map_or_else(|| format!("call_{i}"), str::to_owned),
        name: c["function"]["name"].as_str().unwrap_or_default().to_owned(),
        arguments: c["function"]["arguments"].as_str().unwrap_or("{}").to_owned(),
    }
}

/// Drop tool calls identical (name + semantically equal arguments) to the one
/// before them. Returns the kept calls and how many were dropped.
pub fn collapse_repeats(calls: Vec<ToolCall>) -> (Vec<ToolCall>, usize) {
    let mut kept: Vec<ToolCall> = Vec::with_capacity(calls.len());
    let mut dropped = 0;
    for c in calls {
        if kept.last().is_some_and(|prev| same_call(prev, &c)) {
            dropped += 1;
        } else {
            kept.push(c);
        }
    }
    (kept, dropped)
}

pub(crate) fn same_call(a: &ToolCall, b: &ToolCall) -> bool {
    if a.name != b.name {
        return false;
    }
    match (serde_json::from_str::<Value>(&a.arguments), serde_json::from_str::<Value>(&b.arguments)) {
        (Ok(x), Ok(y)) => x == y,
        _ => a.arguments == b.arguments,
    }
}

/// A Chat Completions function tool. Fields the client left out are omitted:
/// llama.cpp rejects `"description": null` ("type must be string").
pub fn function_tool(name: &Value, description: &Value, parameters: &Value) -> Value {
    let mut function = json!({ "name": name });
    if description.is_string() {
        function["description"] = description.clone();
    }
    if parameters.is_object() {
        function["parameters"] = parameters.clone();
    }
    json!({ "type": "function", "function": function })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn call(id: &str, args: &str) -> ToolCall {
        ToolCall { id: id.into(), name: "get_weather".into(), arguments: args.into() }
    }

    #[test]
    fn collapses_only_consecutive_identical_calls() {
        let calls = vec![
            call("1", r#"{"city":"Karlsruhe"}"#),
            call("2", r#"{ "city": "Karlsruhe" }"#), // same after parsing
            call("3", r#"{"city":"Berlin"}"#),
            call("4", r#"{"city":"Karlsruhe"}"#), // not consecutive → kept
        ];
        let (kept, dropped) = collapse_repeats(calls);
        assert_eq!(dropped, 1);
        assert_eq!(kept.iter().map(|c| c.id.as_str()).collect::<Vec<_>>(), ["1", "3", "4"]);
    }

    #[test]
    fn completion_from_openai() {
        let resp = json!({
            "choices": [{"finish_reason": "tool_calls", "message": {
                "content": null, "reasoning_content": "think",
                "tool_calls": [
                    {"id": "a", "function": {"name": "f", "arguments": "{\"x\":1}"}},
                    {"id": "b", "function": {"name": "f", "arguments": "{\"x\":1}"}}
                ]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        });
        let c = Completion::from_openai(&resp);
        assert_eq!(c.reasoning, "think");
        assert_eq!(c.text, "");
        assert_eq!(c.finish.tool_calls.len(), 1);
        assert_eq!(c.finish.collapsed, 1);
        assert_eq!((c.finish.prompt_tokens, c.finish.completion_tokens), (Some(10), Some(5)));
    }

    #[test]
    fn stream_buffers_and_assembles_tool_calls() {
        let mut n = StreamNormalizer::default();
        let chunks = [
            json!({"id": "x", "model": "m", "created": 1, "choices": [{"delta": {"reasoning_content": "hmm"}}]}),
            json!({"choices": [{"delta": {"content": "Let me check."}}]}),
            json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "get_", "arguments": ""}}]}}]}),
            json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "weather", "arguments": "{\"city\":"}}]}}]}),
            json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\"Karlsruhe\"}"}}]}}]}),
            json!({"choices": [{"delta": {"tool_calls": [{"index": 1, "id": "c2", "function": {"name": "get_weather", "arguments": "{\"city\":\"Karlsruhe\"}"}}]}}]}),
            json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
            json!({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}),
        ];
        let deltas: Vec<Delta> = chunks.iter().flat_map(|c| n.push(c)).collect();
        assert_eq!(deltas, vec![Delta::Reasoning("hmm".into()), Delta::Text("Let me check.".into())]);
        let f = n.finish();
        assert_eq!(f.tool_calls, vec![call("c1", r#"{"city":"Karlsruhe"}"#)]);
        assert_eq!(f.collapsed, 1);
        assert_eq!(f.reason.as_deref(), Some("tool_calls"));
        assert_eq!((f.prompt_tokens, f.completion_tokens), (Some(7), Some(3)));
        assert_eq!(n.meta.unwrap().1, "m");
    }

    #[test]
    fn function_tool_omits_missing_fields() {
        let t = function_tool(&json!("f"), &Value::Null, &Value::Null);
        assert_eq!(t, json!({"type": "function", "function": {"name": "f"}}));
        let t = function_tool(&json!("f"), &json!("d"), &json!({"type": "object"}));
        assert_eq!(t["function"]["description"], "d");
        assert_eq!(t["function"]["parameters"]["type"], "object");
    }
}
