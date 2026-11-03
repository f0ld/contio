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
pub fn apply(request: &Value, resp: &mut Value) {
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

/// `Some((valid, errors))` if the request asked for JSON output.
fn check_response_format(request: &Value, resp: &Value) -> Option<(bool, Vec<String>)> {
    let format = &request["response_format"];
    let kind = format["type"].as_str()?;
    if kind != "json_schema" && kind != "json_object" {
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
        apply(&json!({}), &mut resp);
        assert_eq!(resp["choices"][0]["message"]["tool_calls"].as_array().unwrap().len(), 1);
        assert_eq!(resp["contio"]["tool_calls_collapsed"], 2);
    }

    #[test]
    fn leaves_clean_responses_untouched() {
        let mut resp = completion("hi", Value::Null);
        let before = resp.clone();
        apply(&json!({"messages": []}), &mut resp);
        assert_eq!(resp, before);
    }

    #[test]
    fn validates_json_schema() {
        let req = json!({"response_format": {"type": "json_schema", "json_schema": {"name": "p", "schema": {
            "type": "object", "properties": {"age": {"type": "integer"}}, "required": ["age"]}}}});
        let mut ok = completion(r#"{"age": 36}"#, Value::Null);
        apply(&req, &mut ok);
        assert_eq!(ok["contio"]["schema_valid"], true);

        let mut wrong_type = completion(r#"{"age": "36"}"#, Value::Null);
        apply(&req, &mut wrong_type);
        assert_eq!(wrong_type["contio"]["schema_valid"], false);
        assert!(wrong_type["contio"]["schema_errors"][0].as_str().unwrap().contains("integer"));

        let mut fenced = completion("```json\n{\"age\":36}\n```", Value::Null);
        apply(&req, &mut fenced);
        assert_eq!(fenced["contio"]["schema_valid"], false);
    }

    #[test]
    fn json_object_mode_requires_an_object() {
        let req = json!({"response_format": {"type": "json_object"}});
        let mut arr = completion("[1]", Value::Null);
        apply(&req, &mut arr);
        assert_eq!(arr["contio"]["schema_valid"], false);
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
