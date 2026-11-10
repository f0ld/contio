//! Gateway guards for `/v1/chat/completions` (spec §7.2):
//! - collapse identical consecutive tool calls (repetition loops)
//! - validate `response_format` output and report violations
//!
//! Findings are reported in a `contio` extension object on the response and
//! logged; content is never rewritten beyond dropping repeated tool calls.

use axum::body::Bytes;
use futures_util::{Stream, StreamExt, stream};
use serde_json::{Value, json};

use super::normalize::{StreamNormalizer, ToolCall, same_call};
use super::sse;

/// Apply all guards to a non-streamed `chat.completion` in place.
/// `relaxed`: the backend didn't enforce `response_format` (`relax_response_format`).
pub fn apply(request: &Value, resp: &mut Value, relaxed: bool) {
    let mut report = serde_json::Map::new();

    let mut collapsed = 0;
    for choice in resp["choices"].as_array_mut().into_iter().flatten() {
        let Some(calls) = choice["message"]["tool_calls"].as_array_mut() else { continue };
        let mut kept: Vec<Value> = Vec::with_capacity(calls.len());
        for c in calls.drain(..) {
            if kept.last().is_some_and(|prev| same_call(&as_tool_call(prev), &as_tool_call(&c))) {
                collapsed += 1;
            } else {
                kept.push(c);
            }
        }
        *calls = kept;
    }
    if collapsed > 0 {
        tracing::warn!(collapsed, "collapsed repeated tool calls");
        report.insert("tool_calls_collapsed".into(), json!(collapsed));
    }

    if let Some((valid, errors)) = check_response_format(request, resp) {
        if !valid {
            tracing::warn!(?errors, "response violates response_format");
        }
        report.insert("schema_valid".into(), json!(valid));
        if relaxed {
            report.insert("schema_enforced".into(), json!(false));
        }
        if !errors.is_empty() {
            report.insert("schema_errors".into(), json!(errors));
        }
    }

    if !report.is_empty() {
        resp["contio"] = Value::Object(report);
    }
}

fn as_tool_call(c: &Value) -> ToolCall {
    ToolCall {
        id: c["id"].as_str().unwrap_or_default().to_owned(),
        name: c["function"]["name"].as_str().unwrap_or_default().to_owned(),
        arguments: c["function"]["arguments"].as_str().unwrap_or_default().to_owned(),
    }
}

/// llama.cpp can't combine `tools` with a `response_format` grammar ("failed
/// to parse grammar"). Drop the format and ask for the JSON in the system
/// prompt instead; `apply` still validates the answer. Returns whether the
/// request was changed.
pub fn relax_response_format(request: &mut Value) -> bool {
    let has_tools = request["tools"].as_array().is_some_and(|t| !t.is_empty());
    let kind = request["response_format"]["type"].as_str().unwrap_or_default();
    if !has_tools || (kind != "json_schema" && kind != "json_object") {
        return false;
    }
    let mut hint = String::from("If you answer without calling a tool, reply with only a JSON object");
    if kind == "json_schema" {
        let schema = &request["response_format"]["json_schema"]["schema"];
        hint.push_str(&format!(" that matches this JSON schema:\n{schema}"));
    } else {
        hint.push('.');
    }
    if let Some(obj) = request.as_object_mut() {
        obj.remove("response_format");
    }
    let Some(messages) = request["messages"].as_array_mut() else { return true };
    // chat templates (Qwen) reject system messages after the first turn, so
    // extend the leading one
    match messages.first_mut() {
        Some(first) if first["role"] == "system" => match &mut first["content"] {
            Value::String(s) => {
                s.push_str("\n\n");
                s.push_str(&hint);
            }
            Value::Array(parts) => parts.push(json!({ "type": "text", "text": hint })),
            other => *other = Value::String(hint),
        },
        _ => messages.insert(0, json!({ "role": "system", "content": hint })),
    }
    true
}

/// `Some((valid, errors))` if the request asked for JSON output.
fn check_response_format(request: &Value, resp: &Value) -> Option<(bool, Vec<String>)> {
    let format = &request["response_format"];
    let kind = format["type"].as_str()?;
    if kind != "json_schema" && kind != "json_object" {
        return None;
    }
    // the format applies to the answer, not to tool calls
    if resp["choices"][0]["message"]["tool_calls"].as_array().is_some_and(|c| !c.is_empty()) {
        return None;
    }
    let content = resp["choices"][0]["message"]["content"].as_str().unwrap_or_default();
    let instance: Value = match serde_json::from_str(content) {
        Ok(v) => v,
        Err(e) => return Some((false, vec![format!("content is not JSON: {e}")])),
    };
    if kind == "json_object" {
        return Some((instance.is_object(), vec![]));
    }
    let schema = &format["json_schema"]["schema"];
    let validator = match jsonschema::validator_for(schema) {
        Ok(v) => v,
        Err(e) => return Some((false, vec![format!("invalid schema in request: {e}")])),
    };
    let errors: Vec<String> = validator.iter_errors(&instance).map(|e| e.to_string()).take(10).collect();
    Some((errors.is_empty(), errors))
}

/// Re-emit a streamed chat completion with tool calls buffered and collapsed.
/// Text and reasoning deltas are forwarded immediately.
pub fn rewrite_tool_stream<S>(events: S) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Send
where
    S: Stream<Item = Result<String, String>> + Send + Unpin + 'static,
{
    struct St<S> {
        events: S,
        norm: StreamNormalizer,
        ended: bool,
    }
    stream::unfold(St { events, norm: StreamNormalizer::default(), ended: false }, |mut st| async move {
        if st.ended {
            return None;
        }
        loop {
            match st.events.next().await {
                Some(Ok(data)) => {
                    let Ok(chunk) = serde_json::from_str::<Value>(&data) else { continue };
                    let deltas = st.norm.push(&chunk);
                    if deltas.is_empty() {
                        continue;
                    }
                    let mut out = Vec::new();
                    for d in deltas {
                        let delta = match d {
                            super::normalize::Delta::Text(t) => json!({ "content": t }),
                            super::normalize::Delta::Reasoning(r) => json!({ "reasoning_content": r }),
                        };
                        out.extend_from_slice(&sse::encode(None, &chunk_with(&st.norm, delta, None)));
                    }
                    return Some((Ok(Bytes::from(out)), st));
                }
                Some(Err(e)) => {
                    st.ended = true;
                    return Some((Err(std::io::Error::other(e)), st));
                }
                None => {
                    st.ended = true;
                    let f = st.norm.finish();
                    if f.collapsed > 0 {
                        tracing::warn!(collapsed = f.collapsed, "collapsed repeated tool calls (stream)");
                    }
                    let mut out = Vec::new();
                    if !f.tool_calls.is_empty() {
                        let calls: Vec<Value> = f
                            .tool_calls
                            .iter()
                            .enumerate()
                            .map(|(i, c)| {
                                json!({"index": i, "id": c.id, "type": "function",
                                       "function": {"name": c.name, "arguments": c.arguments}})
                            })
                            .collect();
                        out.extend_from_slice(&sse::encode(None, &chunk_with(&st.norm, json!({ "tool_calls": calls }), None)));
                    }
                    let reason = f.reason.clone().unwrap_or_else(|| "stop".into());
                    out.extend_from_slice(&sse::encode(None, &chunk_with(&st.norm, json!({}), Some(&reason))));
                    if let (Some(p), Some(c)) = (f.prompt_tokens, f.completion_tokens) {
                        let mut usage = chunk_with(&st.norm, json!({}), None);
                        usage["choices"] = json!([]);
                        usage["usage"] = json!({"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c});
                        out.extend_from_slice(&sse::encode(None, &usage));
                    }
                    out.extend_from_slice(&sse::done());
                    return Some((Ok(Bytes::from(out)), st));
                }
            }
        }
    })
}

fn chunk_with(norm: &StreamNormalizer, delta: Value, finish_reason: Option<&str>) -> Value {
    let (id, model, created) = norm.meta.clone().unwrap_or((json!("chatcmpl-contio"), json!(""), json!(0)));
    json!({
        "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{ "index": 0, "delta": delta, "finish_reason": finish_reason }]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn completion(content: &str, calls: Value) -> Value {
        json!({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content, "tool_calls": calls}}]})
    }

    #[test]
    fn collapses_repeated_tool_calls_and_reports() {
        let call = |id: &str| json!({"id": id, "type": "function", "function": {"name": "f", "arguments": "{\"a\":1}"}});
        let mut resp = completion("", json!([call("1"), call("2"), call("3")]));
        apply(&json!({}), &mut resp, false);
        assert_eq!(resp["choices"][0]["message"]["tool_calls"].as_array().unwrap().len(), 1);
        assert_eq!(resp["contio"]["tool_calls_collapsed"], 2);
    }

    #[test]
    fn leaves_clean_responses_untouched() {
        let mut resp = completion("hi", Value::Null);
        let before = resp.clone();
        apply(&json!({"messages": []}), &mut resp, false);
        assert_eq!(resp, before);
    }

    #[test]
    fn validates_json_schema() {
        let req = json!({"response_format": {"type": "json_schema", "json_schema": {"name": "p", "schema": {
            "type": "object", "properties": {"age": {"type": "integer"}}, "required": ["age"]}}}});
        let mut ok = completion(r#"{"age": 36}"#, Value::Null);
        apply(&req, &mut ok, false);
        assert_eq!(ok["contio"]["schema_valid"], true);

        let mut wrong_type = completion(r#"{"age": "36"}"#, Value::Null);
        apply(&req, &mut wrong_type, false);
        assert_eq!(wrong_type["contio"]["schema_valid"], false);
        assert!(wrong_type["contio"]["schema_errors"][0].as_str().unwrap().contains("integer"));

        let mut fenced = completion("```json\n{\"age\":36}\n```", Value::Null);
        apply(&req, &mut fenced, false);
        assert_eq!(fenced["contio"]["schema_valid"], false);
    }

    #[test]
    fn json_object_mode_requires_an_object() {
        let req = json!({"response_format": {"type": "json_object"}});
        let mut arr = completion("[1]", Value::Null);
        apply(&req, &mut arr, false);
        assert_eq!(arr["contio"]["schema_valid"], false);
    }

    fn tool_request(messages: Value, format: Value) -> Value {
        json!({"messages": messages, "response_format": format,
               "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]})
    }

    #[test]
    fn relaxes_schema_into_the_system_prompt() {
        let schema = json!({"type": "object", "required": ["age"]});
        let format = json!({"type": "json_schema", "json_schema": {"name": "p", "schema": schema}});

        let mut req = tool_request(json!([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}]), format.clone());
        assert!(relax_response_format(&mut req));
        assert!(req.get("response_format").is_none());
        let system = req["messages"][0]["content"].as_str().unwrap();
        assert!(system.starts_with("Be brief.\n\n") && system.contains(r#""required":["age"]"#), "{system}");
        assert_eq!(req["messages"].as_array().unwrap().len(), 2);

        let mut no_system = tool_request(json!([{"role": "user", "content": "hi"}]), json!({"type": "json_object"}));
        assert!(relax_response_format(&mut no_system));
        assert_eq!(no_system["messages"][0]["role"], "system");
        assert_eq!(no_system["messages"][1]["role"], "user");

        let mut parts = tool_request(json!([{"role": "system", "content": [{"type": "text", "text": "x"}]}]), format);
        assert!(relax_response_format(&mut parts));
        assert_eq!(parts["messages"][0]["content"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn keeps_format_without_tools_or_json() {
        let mut no_tools = json!({"messages": [], "response_format": {"type": "json_object"}});
        assert!(!relax_response_format(&mut no_tools));
        assert!(no_tools.get("response_format").is_some());

        let mut text = tool_request(json!([]), json!({"type": "text"}));
        let before = text.clone();
        assert!(!relax_response_format(&mut text));
        assert_eq!(text, before);
    }

    #[test]
    fn tool_call_answers_skip_schema_check_and_report_relaxed() {
        let req = json!({"response_format": {"type": "json_object"}});
        let call = json!([{"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]);
        let mut calls = completion("", call);
        apply(&req, &mut calls, true);
        assert!(calls.get("contio").is_none());

        let mut answer = completion(r#"{"a": 1}"#, Value::Null);
        apply(&req, &mut answer, true);
        assert_eq!(answer["contio"]["schema_valid"], true);
        assert_eq!(answer["contio"]["schema_enforced"], false);
    }

    #[tokio::test]
    async fn stream_rewrite_buffers_and_dedupes_tool_calls() {
        let tc = |i: u64, id: &str| {
            json!({"choices": [{"delta": {"tool_calls": [{"index": i, "id": id, "function": {"name": "f", "arguments": "{}"}}]}}]})
                .to_string()
        };
        let events = vec![
            json!({"id": "x", "model": "m", "created": 1, "choices": [{"delta": {"content": "hi"}}]}).to_string(),
            tc(0, "a"),
            tc(1, "b"),
            json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}).to_string(),
        ];
        let out: Vec<Bytes> = rewrite_tool_stream(stream::iter(events.into_iter().map(Ok)))
            .map(|r| r.unwrap())
            .collect()
            .await;
        let text: String = out.iter().map(|b| String::from_utf8_lossy(b).into_owned()).collect();
        let datas: Vec<Value> = text
            .split("\n\n")
            .filter_map(|e| e.strip_prefix("data: "))
            .filter(|d| *d != "[DONE]")
            .map(|d| serde_json::from_str(d).unwrap())
            .collect();
        assert_eq!(datas[0]["choices"][0]["delta"]["content"], "hi");
        assert_eq!(datas[1]["choices"][0]["delta"]["tool_calls"].as_array().unwrap().len(), 1);
        assert_eq!(datas[2]["choices"][0]["finish_reason"], "tool_calls");
        assert!(text.ends_with("data: [DONE]\n\n"));
    }
}
