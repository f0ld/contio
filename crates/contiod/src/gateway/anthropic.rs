//! Anthropic Messages API (`POST /v1/messages`) translated onto chat completions.
//!
//! Supports text, images, tool use/results, system prompts, streaming, and
//! reasoning (as `thinking` blocks). Incoming `thinking` blocks are dropped;
//! server tools (web search etc.) are ignored. `count_tokens` is an estimate.

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use futures_util::{Stream, StreamExt, stream};
use serde_json::{Value, json};

use super::normalize::{Completion, Delta, Finish, StreamNormalizer};
use super::{new_id, send, sse, sse_response};
use crate::api::AppState;

/// `POST /v1/messages`
pub async fn messages(State(state): State<AppState>, body: Bytes) -> Response {
    let request: Value = match serde_json::from_slice(&body) {
        Ok(v @ Value::Object(_)) => v,
        _ => return error(StatusCode::BAD_REQUEST, "invalid_request_error", "body must be a JSON object"),
    };
    let chat = match to_chat(&request) {
        Ok(c) => c,
        Err(msg) => return error(StatusCode::BAD_REQUEST, "invalid_request_error", &msg),
    };
    let model = request["model"].as_str().unwrap_or_default().to_owned();
    let streaming = request["stream"] == true;

    let up = match send(&state, chat, "/v1/chat/completions").await {
        Ok(up) => up,
        Err(resp) => return convert_error(resp).await,
    };
    if !up.response.status().is_success() {
        let status = up.response.status().as_u16();
        let text = up.response.text().await.unwrap_or_default();
        return error(StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY), "api_error", &text);
    }

    if !streaming {
        let resp: Value = match up.response.json().await {
            Ok(v) => v,
            Err(e) => return error(StatusCode::BAD_GATEWAY, "api_error", &e.to_string()),
        };
        return Json(from_completion(&Completion::from_openai(&resp), &model)).into_response();
    }

    let lease = up.lease;
    let events = sse::data_events(up.response.bytes_stream());
    let body = stream_events(events, model).map(move |b| {
        let _keep_alive = &lease;
        b
    });
    sse_response(body)
}

/// `POST /v1/messages/count_tokens`: ~4 characters per token estimate.
pub async fn count_tokens(body: Bytes) -> Response {
    let Ok(request) = serde_json::from_slice::<Value>(&body) else {
        return error(StatusCode::BAD_REQUEST, "invalid_request_error", "body must be JSON");
    };
    let chars = ["system", "messages", "tools"]
        .iter()
        .map(|k| request.get(*k).map_or(0, |v| v.to_string().len()))
        .sum::<usize>();
    Json(json!({ "input_tokens": chars.div_ceil(4) })).into_response()
}

// --- request ------------------------------------------------------------------

pub(crate) fn to_chat(req: &Value) -> Result<Value, String> {
    let mut messages = Vec::new();
    match &req["system"] {
        Value::String(s) if !s.is_empty() => messages.push(json!({"role": "system", "content": s})),
        Value::Array(blocks) => {
            let text = text_of(blocks);
            if !text.is_empty() {
                messages.push(json!({"role": "system", "content": text}));
            }
        }
        _ => {}
    }

    for m in req["messages"].as_array().ok_or("`messages` must be an array")? {
        let role = m["role"].as_str().ok_or("message without role")?;
        let blocks = match &m["content"] {
            Value::String(s) => vec![json!({"type": "text", "text": s})],
            Value::Array(b) => b.clone(),
            _ => return Err("message content must be a string or array".into()),
        };
        match role {
            "user" => {
                // tool results become `tool` messages, placed first (they answer
                // the previous assistant turn), then the remaining user content
                let mut parts = Vec::new();
                for b in &blocks {
                    match b["type"].as_str() {
                        Some("tool_result") => {
                            let content = match &b["content"] {
                                Value::String(s) => s.clone(),
                                Value::Array(inner) => text_of(inner),
                                _ => String::new(),
                            };
                            let content = if b["is_error"] == true { format!("Error: {content}") } else { content };
                            messages.push(json!({
                                "role": "tool",
                                "tool_call_id": b["tool_use_id"],
                                "content": content,
                            }));
                        }
                        Some("text") => parts.push(json!({"type": "text", "text": b["text"]})),
                        Some("image") => {
                            if let Some(url) = image_url(&b["source"]) {
                                parts.push(json!({"type": "image_url", "image_url": {"url": url}}));
                            }
                        }
                        _ => {}
                    }
                }
                if !parts.is_empty() {
                    let only_text = parts.iter().all(|p| p["type"] == "text");
                    let content = if only_text {
                        Value::String(parts.iter().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join("\n"))
                    } else {
                        Value::Array(parts)
                    };
                    messages.push(json!({"role": "user", "content": content}));
                }
            }
            "assistant" => {
                let text = text_of(&blocks);
                let calls: Vec<Value> = blocks
                    .iter()
                    .filter(|b| b["type"] == "tool_use")
                    .map(|b| {
                        json!({"id": b["id"], "type": "function",
                               "function": {"name": b["name"], "arguments": b["input"].to_string()}})
                    })
                    .collect();
                let mut msg = json!({"role": "assistant", "content": if text.is_empty() { Value::Null } else { text.into() }});
                if !calls.is_empty() {
                    msg["tool_calls"] = Value::Array(calls);
                }
                messages.push(msg);
            }
            other => return Err(format!("unsupported role `{other}`")),
        }
    }

    let mut chat = json!({ "model": req["model"], "messages": messages });
    for (from, to) in [("max_tokens", "max_tokens"), ("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k"), ("stop_sequences", "stop")] {
        if !req[from].is_null() {
            chat[to] = req[from].clone();
        }
    }
    if req["stream"] == true {
        chat["stream"] = true.into();
        chat["stream_options"] = json!({"include_usage": true});
    }

    let tools: Vec<Value> = req["tools"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|t| t.get("input_schema").is_some()) // skip server tools
        .map(|t| {
            json!({"type": "function", "function": {
                "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}})
        })
        .collect();
    if !tools.is_empty() {
        chat["tools"] = Value::Array(tools);
        match req["tool_choice"]["type"].as_str() {
            Some("any") => chat["tool_choice"] = "required".into(),
            Some("none") => chat["tool_choice"] = "none".into(),
            Some("tool") => {
                chat["tool_choice"] = json!({"type": "function", "function": {"name": req["tool_choice"]["name"]}})
            }
            _ => {}
        }
        if req["tool_choice"]["disable_parallel_tool_use"] == true {
            chat["parallel_tool_calls"] = false.into();
        }
    }
    Ok(chat)
}

fn text_of(blocks: &[Value]) -> String {
    blocks
        .iter()
        .filter(|b| b["type"] == "text")
        .filter_map(|b| b["text"].as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

fn image_url(source: &Value) -> Option<String> {
    match source["type"].as_str()? {
        "base64" => Some(format!("data:{};base64,{}", source["media_type"].as_str()?, source["data"].as_str()?)),
        "url" => source["url"].as_str().map(str::to_owned),
        _ => None,
    }
}

// --- response -----------------------------------------------------------------

fn stop_reason(f: &Finish) -> &'static str {
    if !f.tool_calls.is_empty() {
        return "tool_use";
    }
    match f.reason.as_deref() {
        Some("length") => "max_tokens",
        _ => "end_turn",
    }
}

fn tool_input(arguments: &str) -> Value {
    serde_json::from_str(arguments).unwrap_or_else(|_| json!({}))
}

pub(crate) fn from_completion(c: &Completion, model: &str) -> Value {
    let mut content = Vec::new();
    if !c.reasoning.is_empty() {
        content.push(json!({"type": "thinking", "thinking": c.reasoning, "signature": ""}));
    }
    if !c.text.is_empty() {
        content.push(json!({"type": "text", "text": c.text}));
    }
    for t in &c.finish.tool_calls {
        content.push(json!({"type": "tool_use", "id": t.id, "name": t.name, "input": tool_input(&t.arguments)}));
    }
    json!({
        "id": new_id("msg"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason(&c.finish),
        "stop_sequence": null,
        "usage": {
            "input_tokens": c.finish.prompt_tokens.unwrap_or(0),
            "output_tokens": c.finish.completion_tokens.unwrap_or(0),
        }
    })
}

#[derive(Clone, Copy, PartialEq)]
enum Block {
    Thinking,
    Text,
}

/// Translate an OpenAI chunk stream into Anthropic stream events.
pub(crate) fn stream_events<S>(events: S, model: String) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Send
where
    S: Stream<Item = Result<String, String>> + Send + Unpin + 'static,
{
    struct St<S> {
        events: S,
        norm: StreamNormalizer,
        model: String,
        started: bool,
        open: Option<Block>,
        index: usize,
        ended: bool,
    }

    fn ev(name: &str, data: Value) -> Bytes {
        sse::encode(Some(name), &data)
    }

    impl<S> St<S> {
        fn switch_to(&mut self, block: Block, out: &mut Vec<u8>) {
            if self.open == Some(block) {
                return;
            }
            self.close(out);
            let content_block = match block {
                Block::Thinking => json!({"type": "thinking", "thinking": ""}),
                Block::Text => json!({"type": "text", "text": ""}),
            };
            out.extend_from_slice(&ev(
                "content_block_start",
                json!({"type": "content_block_start", "index": self.index, "content_block": content_block}),
            ));
            self.open = Some(block);
        }

        fn close(&mut self, out: &mut Vec<u8>) {
            if let Some(block) = self.open.take() {
                if block == Block::Thinking {
                    out.extend_from_slice(&ev(
                        "content_block_delta",
                        json!({"type": "content_block_delta", "index": self.index,
                               "delta": {"type": "signature_delta", "signature": ""}}),
                    ));
                }
                out.extend_from_slice(&ev("content_block_stop", json!({"type": "content_block_stop", "index": self.index})));
                self.index += 1;
            }
        }

        fn start(&mut self, out: &mut Vec<u8>) {
            if !self.started {
                self.started = true;
                out.extend_from_slice(&ev(
                    "message_start",
                    json!({"type": "message_start", "message": {
                        "id": new_id("msg"), "type": "message", "role": "assistant", "model": self.model,
                        "content": [], "stop_reason": null, "stop_sequence": null,
                        "usage": {"input_tokens": 0, "output_tokens": 0}}}),
                ));
            }
        }
    }

    let st = St { events, norm: StreamNormalizer::default(), model, started: false, open: None, index: 0, ended: false };
    stream::unfold(st, |mut st| async move {
        if st.ended {
            return None;
        }
        if !st.started {
            let mut out = Vec::new();
            st.start(&mut out);
            return Some((Ok(Bytes::from(out)), st));
        }
        loop {
            let mut out = Vec::new();
            match st.events.next().await {
                Some(Ok(data)) => {
                    let Ok(chunk) = serde_json::from_str::<Value>(&data) else { continue };
                    for d in st.norm.push(&chunk) {
                        let (block, delta) = match d {
                            Delta::Reasoning(t) => (Block::Thinking, json!({"type": "thinking_delta", "thinking": t})),
                            Delta::Text(t) => (Block::Text, json!({"type": "text_delta", "text": t})),
                        };
                        st.switch_to(block, &mut out);
                        out.extend_from_slice(&ev(
                            "content_block_delta",
                            json!({"type": "content_block_delta", "index": st.index, "delta": delta}),
                        ));
                    }
                    if out.is_empty() {
                        continue;
                    }
                    return Some((Ok(Bytes::from(out)), st));
                }
                Some(Err(e)) => {
                    st.ended = true;
                    out.extend_from_slice(&ev(
                        "error",
                        json!({"type": "error", "error": {"type": "api_error", "message": e}}),
                    ));
                    return Some((Ok(Bytes::from(out)), st));
                }
                None => {
                    st.ended = true;
                    st.close(&mut out);
                    let f = st.norm.finish();
                    for t in &f.tool_calls {
                        out.extend_from_slice(&ev(
                            "content_block_start",
                            json!({"type": "content_block_start", "index": st.index,
                                   "content_block": {"type": "tool_use", "id": t.id, "name": t.name, "input": {}}}),
                        ));
                        out.extend_from_slice(&ev(
                            "content_block_delta",
                            json!({"type": "content_block_delta", "index": st.index,
                                   "delta": {"type": "input_json_delta", "partial_json": t.arguments}}),
                        ));
                        out.extend_from_slice(&ev("content_block_stop", json!({"type": "content_block_stop", "index": st.index})));
                        st.index += 1;
                    }
                    out.extend_from_slice(&ev(
                        "message_delta",
                        json!({"type": "message_delta",
                               "delta": {"stop_reason": stop_reason(&f), "stop_sequence": null},
                               "usage": {"input_tokens": f.prompt_tokens.unwrap_or(0),
                                         "output_tokens": f.completion_tokens.unwrap_or(0)}}),
                    ));
                    out.extend_from_slice(&ev("message_stop", json!({"type": "message_stop"})));
                    return Some((Ok(Bytes::from(out)), st));
                }
            }
        }
    })
}

// --- errors -------------------------------------------------------------------

fn error(status: StatusCode, kind: &str, message: &str) -> Response {
    (status, Json(json!({"type": "error", "error": {"type": kind, "message": message}}))).into_response()
}

/// Re-shape an OpenAI-style error response from the gateway.
async fn convert_error(resp: Response) -> Response {
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20).await.unwrap_or_default();
    let message = serde_json::from_slice::<Value>(&bytes)
        .ok()
        .and_then(|v| v["error"]["message"].as_str().map(str::to_owned))
        .unwrap_or_else(|| String::from_utf8_lossy(&bytes).into_owned());
    let kind = match status {
        StatusCode::BAD_REQUEST => "invalid_request_error",
        StatusCode::NOT_FOUND => "not_found_error",
        StatusCode::SERVICE_UNAVAILABLE => "overloaded_error",
        _ => "api_error",
    };
    error(status, kind, &message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translates_a_tool_use_conversation() {
        let req = json!({
            "model": "coder", "max_tokens": 1024, "stream": true,
            "system": [{"type": "text", "text": "You are helpful."}],
            "tools": [
                {"name": "get_weather", "description": "Weather", "input_schema": {"type": "object"}},
                {"type": "web_search_20250305", "name": "web_search"}
            ],
            "tool_choice": {"type": "any"},
            "messages": [
                {"role": "user", "content": "Weather in Karlsruhe?"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "need tool", "signature": "x"},
                    {"type": "text", "text": "Checking."},
                    {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "Karlsruhe"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu_1", "content": [{"type": "text", "text": "18°C"}]},
                    {"type": "text", "text": "Thanks"}
                ]}
            ]
        });
        let chat = to_chat(&req).unwrap();
        let m = chat["messages"].as_array().unwrap();
        assert_eq!(m[0], json!({"role": "system", "content": "You are helpful."}));
        assert_eq!(m[1], json!({"role": "user", "content": "Weather in Karlsruhe?"}));
        assert_eq!(m[2]["content"], "Checking.");
        assert_eq!(m[2]["tool_calls"][0]["function"]["arguments"], "{\"city\":\"Karlsruhe\"}");
        assert_eq!(m[3], json!({"role": "tool", "tool_call_id": "tu_1", "content": "18°C"}));
        assert_eq!(m[4], json!({"role": "user", "content": "Thanks"}));
        assert_eq!(chat["tools"].as_array().unwrap().len(), 1, "server tools are dropped");
        assert_eq!(chat["tool_choice"], "required");
        assert_eq!(chat["stream_options"]["include_usage"], true);
        assert_eq!(chat["max_tokens"], 1024);
    }

    #[test]
    fn images_become_data_urls() {
        let req = json!({"model": "m", "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA"}},
            {"type": "text", "text": "What is this?"}
        ]}]});
        let chat = to_chat(&req).unwrap();
        let parts = chat["messages"][0]["content"].as_array().unwrap();
        assert_eq!(parts[0]["image_url"]["url"], "data:image/png;base64,AAA");
    }

    #[test]
    fn completion_to_message() {
        let c = Completion::from_openai(&json!({
            "choices": [{"finish_reason": "tool_calls", "message": {"content": "Checking.", "reasoning_content": "hmm",
                "tool_calls": [{"id": "c1", "function": {"name": "get_weather", "arguments": "{\"city\":\"K\"}"}}]}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7}
        }));
        let m = from_completion(&c, "coder");
        assert_eq!(m["stop_reason"], "tool_use");
        assert_eq!(m["content"][0]["type"], "thinking");
        assert_eq!(m["content"][1], json!({"type": "text", "text": "Checking."}));
        assert_eq!(m["content"][2]["input"], json!({"city": "K"}));
        assert_eq!(m["usage"], json!({"input_tokens": 12, "output_tokens": 7}));
        assert_eq!(m["model"], "coder");
    }

    #[tokio::test]
    async fn stream_produces_well_formed_event_sequence() {
        let chunks = vec![
            json!({"choices": [{"delta": {"reasoning_content": "hm"}}]}),
            json!({"choices": [{"delta": {"content": "Hi"}}]}),
            json!({"choices": [{"delta": {"content": "!"}}]}),
            json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "f", "arguments": "{}"}}]}}]}),
            json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
            json!({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 4}}),
        ];
        let input = stream::iter(chunks.into_iter().map(|c| Ok(c.to_string())));
        let bytes: Vec<Bytes> = stream_events(input, "m".into()).map(|b| b.unwrap()).collect().await;
        let text: String = bytes.iter().map(|b| String::from_utf8_lossy(b).into_owned()).collect();
        let names: Vec<&str> = text.lines().filter_map(|l| l.strip_prefix("event: ")).collect();
        assert_eq!(
            names,
            [
                "message_start",
                "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop", // thinking (+signature)
                "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop", // text
                "content_block_start", "content_block_delta", "content_block_stop", // tool_use
                "message_delta", "message_stop",
            ]
        );
        assert!(text.contains(r#""stop_reason":"tool_use""#));
        assert!(text.contains(r#""output_tokens":4"#));
        assert!(text.contains(r#""index":2"#), "tool_use is the third block");
    }
}
