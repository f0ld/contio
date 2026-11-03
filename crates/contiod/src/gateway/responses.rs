//! OpenAI Responses API (`POST /v1/responses`) translated onto chat completions.
//!
//! Supports string and item inputs (messages, function calls and outputs),
//! `instructions`, function tools, `text.format` (JSON schema), streaming, and
//! `previous_response_id` via an in-memory store (last 256 responses; lost on
//! restart; SQLite later). Built-in tools (web search, file search, …) are ignored.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use futures_util::{Stream, StreamExt, stream};
use serde_json::{Value, json};

use super::normalize::{Completion, Delta, Finish, StreamNormalizer};
use super::{new_id, openai_error, send, sse, sse_response};
use crate::api::AppState;

const STORE_CAPACITY: usize = 256;

/// Conversation history per response id, for `previous_response_id`.
#[derive(Clone, Default)]
pub struct Store(Arc<Mutex<StoreInner>>);

#[derive(Default)]
struct StoreInner {
    conversations: HashMap<String, Vec<Value>>,
    order: VecDeque<String>,
}

impl Store {
    fn get(&self, id: &str) -> Option<Vec<Value>> {
        self.0.lock().expect("store poisoned").conversations.get(id).cloned()
    }

    fn put(&self, id: String, messages: Vec<Value>) {
        let mut s = self.0.lock().expect("store poisoned");
        s.order.push_back(id.clone());
        s.conversations.insert(id, messages);
        while s.order.len() > STORE_CAPACITY {
            if let Some(old) = s.order.pop_front() {
                s.conversations.remove(&old);
            }
        }
    }
}

/// `POST /v1/responses`
pub async fn create(State(state): State<AppState>, body: Bytes) -> Response {
    let request: Value = match serde_json::from_slice(&body) {
        Ok(v @ Value::Object(_)) => v,
        _ => return openai_error(StatusCode::BAD_REQUEST, "invalid_request_error", "body must be a JSON object", None),
    };
    let previous = match request["previous_response_id"].as_str() {
        None => Vec::new(),
        Some(id) => match state.responses.get(id) {
            Some(h) => h,
            None => {
                return openai_error(
                    StatusCode::NOT_FOUND,
                    "invalid_request_error",
                    &format!("previous response `{id}` not found (contio keeps the last {STORE_CAPACITY} in memory)"),
                    Some("previous_response_id"),
                );
            }
        },
    };
    let (chat, conversation) = match to_chat(&request, previous) {
        Ok(x) => x,
        Err(msg) => return openai_error(StatusCode::BAD_REQUEST, "invalid_request_error", &msg, None),
    };
    let ctx = Ctx {
        id: new_id("resp"),
        model: request["model"].as_str().unwrap_or_default().to_owned(),
        created_at: now_secs(),
        request: request.clone(),
        store: (request["store"] != false).then(|| (state.responses.clone(), conversation)),
    };

    let up = match send(&state, chat, "/v1/chat/completions").await {
        Ok(up) => up,
        Err(resp) => return resp,
    };
    if !up.response.status().is_success() {
        return super::relay(up.response, up.lease);
    }

    if request["stream"] != true {
        let resp: Value = match up.response.json().await {
            Ok(v) => v,
            Err(e) => return openai_error(StatusCode::BAD_GATEWAY, "upstream_error", &e.to_string(), None),
        };
        let c = Completion::from_openai(&resp);
        let output = output_items(&c.reasoning, &c.text, &c.finish);
        ctx.remember(&c.text, &c.finish);
        return Json(ctx.object(status_of(&c.finish), output, &c.finish)).into_response();
    }

    let lease = up.lease;
    let events = sse::data_events(up.response.bytes_stream());
    let body = stream_events(events, ctx).map(move |b| {
        let _keep_alive = &lease;
        b
    });
    sse_response(body)
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}

// --- request ------------------------------------------------------------------

/// Returns the chat request and the conversation to remember (without
/// `instructions`, which don't carry over per the Responses API).
pub(crate) fn to_chat(req: &Value, previous: Vec<Value>) -> Result<(Value, Vec<Value>), String> {
    let mut conversation = previous;
    match &req["input"] {
        Value::String(s) => conversation.push(json!({"role": "user", "content": s})),
        Value::Array(items) => {
            for item in items {
                push_item(&mut conversation, item)?;
            }
        }
        Value::Null => {}
        _ => return Err("`input` must be a string or an array".into()),
    }

    let mut messages = Vec::new();
    if let Some(instr) = req["instructions"].as_str().filter(|s| !s.is_empty()) {
        messages.push(json!({"role": "system", "content": instr}));
    }
    messages.extend(conversation.iter().cloned());

    let mut chat = json!({ "model": req["model"], "messages": messages });
    for (from, to) in [("max_output_tokens", "max_tokens"), ("temperature", "temperature"), ("top_p", "top_p")] {
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
        .filter(|t| t["type"] == "function")
        .map(|t| {
            json!({"type": "function", "function": {
                "name": t["name"], "description": t["description"], "parameters": t["parameters"]}})
        })
        .collect();
    if !tools.is_empty() {
        chat["tools"] = Value::Array(tools);
        match &req["tool_choice"] {
            Value::String(s) => chat["tool_choice"] = s.as_str().into(),
            Value::Object(o) if o.get("type").and_then(Value::as_str) == Some("function") => {
                chat["tool_choice"] = json!({"type": "function", "function": {"name": o["name"]}})
            }
            _ => {}
        }
        if req["parallel_tool_calls"] == false {
            chat["parallel_tool_calls"] = false.into();
        }
    }

    let format = &req["text"]["format"];
    match format["type"].as_str() {
        Some("json_schema") => {
            chat["response_format"] = json!({"type": "json_schema", "json_schema": {
                "name": format["name"], "schema": format["schema"], "strict": format["strict"]}})
        }
        Some("json_object") => chat["response_format"] = json!({"type": "json_object"}),
        _ => {}
    }
    Ok((chat, conversation))
}

fn push_item(conv: &mut Vec<Value>, item: &Value) -> Result<(), String> {
    match item["type"].as_str() {
        Some("function_call") => {
            let call = json!({"id": item["call_id"], "type": "function",
                              "function": {"name": item["name"], "arguments": item["arguments"]}});
            // consecutive calls belong to one assistant turn
            if let Some(last) = conv.last_mut()
                && last["role"] == "assistant"
                && last["tool_calls"].is_array()
            {
                last["tool_calls"].as_array_mut().expect("checked").push(call);
            } else {
                conv.push(json!({"role": "assistant", "content": null, "tool_calls": [call]}));
            }
        }
        Some("function_call_output") => {
            let output = match &item["output"] {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            conv.push(json!({"role": "tool", "tool_call_id": item["call_id"], "content": output}));
        }
        Some("reasoning") => {}
        Some("message") | None => {
            let role = match item["role"].as_str() {
                Some("developer") => "system",
                Some(r @ ("user" | "assistant" | "system")) => r,
                _ => return Err("input message needs role user/assistant/system/developer".into()),
            };
            conv.push(json!({"role": role, "content": message_content(&item["content"])}));
        }
        Some(other) => tracing::debug!("ignoring unsupported input item `{other}`"),
    }
    Ok(())
}

fn message_content(content: &Value) -> Value {
    let Value::Array(parts) = content else { return content.clone() };
    let converted: Vec<Value> = parts
        .iter()
        .filter_map(|p| match p["type"].as_str() {
            Some("input_text" | "output_text" | "text") => Some(json!({"type": "text", "text": p["text"]})),
            Some("input_image") => {
                let url = p["image_url"].as_str()?;
                Some(json!({"type": "image_url", "image_url": {"url": url}}))
            }
            _ => None,
        })
        .collect();
    if converted.iter().all(|p| p["type"] == "text") {
        Value::String(converted.iter().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join("\n"))
    } else {
        Value::Array(converted)
    }
}

// --- response -----------------------------------------------------------------

struct Ctx {
    id: String,
    model: String,
    created_at: u64,
    request: Value,
    store: Option<(Store, Vec<Value>)>,
}

impl Ctx {
    fn object(&self, status: &str, output: Vec<Value>, f: &Finish) -> Value {
        let (inp, out) = (f.prompt_tokens.unwrap_or(0), f.completion_tokens.unwrap_or(0));
        json!({
            "id": self.id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "incomplete_details": if status == "incomplete" { json!({"reason": "max_output_tokens"}) } else { Value::Null },
            "model": self.model,
            "output": output,
            "instructions": self.request["instructions"],
            "previous_response_id": self.request["previous_response_id"],
            "tools": self.request["tools"].as_array().cloned().unwrap_or_default(),
            "text": if self.request["text"].is_null() { json!({"format": {"type": "text"}}) } else { self.request["text"].clone() },
            "usage": {
                "input_tokens": inp,
                "output_tokens": out,
                "total_tokens": inp + out,
            },
        })
    }

    /// Store the conversation plus this turn's assistant message.
    fn remember(&self, text: &str, f: &Finish) {
        let Some((store, conversation)) = &self.store else { return };
        let mut conv = conversation.clone();
        let mut msg = json!({"role": "assistant", "content": if text.is_empty() { Value::Null } else { text.into() }});
        if !f.tool_calls.is_empty() {
            msg["tool_calls"] = f
                .tool_calls
                .iter()
                .map(|t| json!({"id": t.id, "type": "function", "function": {"name": t.name, "arguments": t.arguments}}))
                .collect();
        }
        conv.push(msg);
        store.put(self.id.clone(), conv);
    }
}

fn status_of(f: &Finish) -> &'static str {
    if f.reason.as_deref() == Some("length") && f.tool_calls.is_empty() { "incomplete" } else { "completed" }
}

fn reasoning_item(id: &str, text: &str) -> Value {
    json!({"type": "reasoning", "id": id, "summary": [], "content": [{"type": "reasoning_text", "text": text}]})
}

fn message_item(id: &str, text: &str, status: &str) -> Value {
    json!({"type": "message", "id": id, "status": status, "role": "assistant",
           "content": [{"type": "output_text", "text": text, "annotations": []}]})
}

fn call_item(id: &str, t: &super::normalize::ToolCall, status: &str) -> Value {
    json!({"type": "function_call", "id": id, "call_id": t.id, "name": t.name, "arguments": t.arguments, "status": status})
}

fn output_items(reasoning: &str, text: &str, f: &Finish) -> Vec<Value> {
    let mut out = Vec::new();
    if !reasoning.is_empty() {
        out.push(reasoning_item(&new_id("rs"), reasoning));
    }
    if !text.is_empty() {
        out.push(message_item(&new_id("msg"), text, "completed"));
    }
    for t in &f.tool_calls {
        out.push(call_item(&new_id("fc"), t, "completed"));
    }
    out
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Reasoning,
    Message,
}

fn stream_events<S>(events: S, ctx: Ctx) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Send
where
    S: Stream<Item = Result<String, String>> + Send + Unpin + 'static,
{
    struct Open {
        kind: Kind,
        id: String,
        text: String,
    }
    struct St<S> {
        events: S,
        norm: StreamNormalizer,
        ctx: Ctx,
        seq: u64,
        output: Vec<Value>,
        open: Option<Open>,
        full_text: String,
        started: bool,
        ended: bool,
    }

    impl<S> St<S> {
        fn ev(&mut self, out: &mut Vec<u8>, mut data: Value) {
            data["sequence_number"] = self.seq.into();
            self.seq += 1;
            let name = data["type"].as_str().unwrap_or_default().to_owned();
            out.extend_from_slice(&sse::encode(Some(&name), &data));
        }

        fn open(&mut self, kind: Kind, out: &mut Vec<u8>) {
            if self.open.as_ref().is_some_and(|o| o.kind == kind) {
                return;
            }
            self.close(out);
            let index = self.output.len();
            let (id, item) = match kind {
                Kind::Reasoning => {
                    let id = new_id("rs");
                    let item = json!({"type": "reasoning", "id": id, "summary": []});
                    (id, item)
                }
                Kind::Message => {
                    let id = new_id("msg");
                    let item = json!({"type": "message", "id": id, "status": "in_progress", "role": "assistant", "content": []});
                    (id, item)
                }
            };
            self.ev(out, json!({"type": "response.output_item.added", "output_index": index, "item": item}));
            let part = match kind {
                Kind::Reasoning => json!({"type": "reasoning_text", "text": ""}),
                Kind::Message => json!({"type": "output_text", "text": "", "annotations": []}),
            };
            self.ev(out, json!({"type": "response.content_part.added", "item_id": id, "output_index": index,
                                "content_index": 0, "part": part}));
            self.open = Some(Open { kind, id, text: String::new() });
        }

        fn delta(&mut self, kind: Kind, text: String, out: &mut Vec<u8>) {
            self.open(kind, out);
            let index = self.output.len();
            let o = self.open.as_mut().expect("just opened");
            o.text.push_str(&text);
            let id = o.id.clone();
            let ty = match kind {
                Kind::Reasoning => "response.reasoning_text.delta",
                Kind::Message => "response.output_text.delta",
            };
            if kind == Kind::Message {
                self.full_text.push_str(&text);
            }
            self.ev(out, json!({"type": ty, "item_id": id, "output_index": index, "content_index": 0, "delta": text}));
        }

        fn close(&mut self, out: &mut Vec<u8>) {
            let Some(o) = self.open.take() else { return };
            let index = self.output.len();
            let (done_ty, part, item) = match o.kind {
                Kind::Reasoning => (
                    "response.reasoning_text.done",
                    json!({"type": "reasoning_text", "text": o.text}),
                    reasoning_item(&o.id, &o.text),
                ),
                Kind::Message => (
                    "response.output_text.done",
                    json!({"type": "output_text", "text": o.text, "annotations": []}),
                    message_item(&o.id, &o.text, "completed"),
                ),
            };
            self.ev(out, json!({"type": done_ty, "item_id": o.id, "output_index": index, "content_index": 0, "text": o.text}));
            self.ev(out, json!({"type": "response.content_part.done", "item_id": o.id, "output_index": index,
                                "content_index": 0, "part": part}));
            self.ev(out, json!({"type": "response.output_item.done", "output_index": index, "item": item}));
            self.output.push(item);
        }
    }

    let st = St {
        events,
        norm: StreamNormalizer::default(),
        ctx,
        seq: 0,
        output: Vec::new(),
        open: None,
        full_text: String::new(),
        started: false,
        ended: false,
    };
    stream::unfold(st, |mut st| async move {
        if st.ended {
            return None;
        }
        let mut out = Vec::new();
        if !st.started {
            st.started = true;
            let mut obj = st.ctx.object("in_progress", vec![], &Finish::default());
            obj["usage"] = Value::Null;
            st.ev(&mut out, json!({"type": "response.created", "response": obj.clone()}));
            st.ev(&mut out, json!({"type": "response.in_progress", "response": obj}));
            return Some((Ok(Bytes::from(out)), st));
        }
        loop {
            match st.events.next().await {
                Some(Ok(data)) => {
                    let Ok(chunk) = serde_json::from_str::<Value>(&data) else { continue };
                    for d in st.norm.push(&chunk) {
                        match d {
                            Delta::Reasoning(t) => st.delta(Kind::Reasoning, t, &mut out),
                            Delta::Text(t) => st.delta(Kind::Message, t, &mut out),
                        }
                    }
                    if out.is_empty() {
                        continue;
                    }
                    return Some((Ok(Bytes::from(out)), st));
                }
                Some(Err(e)) => {
                    st.ended = true;
                    let mut obj = st.ctx.object("failed", std::mem::take(&mut st.output), &Finish::default());
                    obj["error"] = json!({"code": "server_error", "message": e});
                    st.ev(&mut out, json!({"type": "response.failed", "response": obj}));
                    return Some((Ok(Bytes::from(out)), st));
                }
                None => {
                    st.ended = true;
                    st.close(&mut out);
                    let f = st.norm.finish();
                    for t in &f.tool_calls {
                        let index = st.output.len();
                        let id = new_id("fc");
                        st.ev(&mut out, json!({"type": "response.output_item.added", "output_index": index,
                                               "item": {"type": "function_call", "id": id, "call_id": t.id, "name": t.name,
                                                        "arguments": "", "status": "in_progress"}}));
                        st.ev(&mut out, json!({"type": "response.function_call_arguments.delta", "item_id": id,
                                               "output_index": index, "delta": t.arguments}));
                        st.ev(&mut out, json!({"type": "response.function_call_arguments.done", "item_id": id,
                                               "output_index": index, "arguments": t.arguments}));
                        let item = call_item(&id, t, "completed");
                        st.ev(&mut out, json!({"type": "response.output_item.done", "output_index": index, "item": item}));
                        st.output.push(item);
                    }
                    let status = status_of(&f);
                    st.ctx.remember(&st.full_text, &f);
                    let obj = st.ctx.object(status, std::mem::take(&mut st.output), &f);
                    let ty = if status == "incomplete" { "response.incomplete" } else { "response.completed" };
                    st.ev(&mut out, json!({"type": ty, "response": obj}));
                    return Some((Ok(Bytes::from(out)), st));
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translates_items_tools_and_format() {
        let req = json!({
            "model": "coder", "instructions": "Be brief.", "max_output_tokens": 500, "stream": true,
            "input": [
                {"role": "developer", "content": "Use tools."},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Weather?"}]},
                {"type": "reasoning", "id": "rs_1", "summary": []},
                {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": "{\"city\":\"K\"}"},
                {"type": "function_call", "call_id": "c2", "name": "get_weather", "arguments": "{\"city\":\"B\"}"},
                {"type": "function_call_output", "call_id": "c1", "output": "18°C"}
            ],
            "tools": [{"type": "function", "name": "get_weather", "parameters": {"type": "object"}}, {"type": "web_search"}],
            "tool_choice": "required",
            "text": {"format": {"type": "json_schema", "name": "w", "schema": {"type": "object"}, "strict": true}}
        });
        let (chat, conv) = to_chat(&req, vec![]).unwrap();
        let m = chat["messages"].as_array().unwrap();
        assert_eq!(m[0], json!({"role": "system", "content": "Be brief."}));
        assert_eq!(m[1], json!({"role": "system", "content": "Use tools."}));
        assert_eq!(m[2], json!({"role": "user", "content": "Weather?"}));
        assert_eq!(m[3]["tool_calls"].as_array().unwrap().len(), 2, "consecutive calls merge");
        assert_eq!(m[4], json!({"role": "tool", "tool_call_id": "c1", "content": "18°C"}));
        assert_eq!(conv.len(), 4, "instructions are not part of the stored conversation");
        assert_eq!(chat["tools"].as_array().unwrap().len(), 1);
        assert_eq!(chat["tool_choice"], "required");
        assert_eq!(chat["max_tokens"], 500);
        assert_eq!(chat["response_format"]["json_schema"]["name"], "w");
    }

    #[test]
    fn previous_conversation_is_prepended() {
        let prev = vec![json!({"role": "user", "content": "Hi"}), json!({"role": "assistant", "content": "Hello"})];
        let (chat, conv) = to_chat(&json!({"model": "m", "input": "And now?"}), prev).unwrap();
        assert_eq!(chat["messages"].as_array().unwrap().len(), 3);
        assert_eq!(conv[2]["content"], "And now?");
    }

    #[test]
    fn store_evicts_oldest() {
        let s = Store::default();
        for i in 0..STORE_CAPACITY + 1 {
            s.put(format!("r{i}"), vec![]);
        }
        assert!(s.get("r0").is_none());
        assert!(s.get(&format!("r{STORE_CAPACITY}")).is_some());
    }

    fn ctx(store: Option<(Store, Vec<Value>)>) -> Ctx {
        Ctx { id: "resp_1".into(), model: "coder".into(), created_at: 1, request: json!({"model": "coder"}), store }
    }

    #[test]
    fn output_items_and_status() {
        let c = Completion::from_openai(&json!({
            "choices": [{"finish_reason": "length", "message": {"content": "partial", "reasoning_content": "r"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 9}
        }));
        let items = output_items(&c.reasoning, &c.text, &c.finish);
        assert_eq!(items[0]["type"], "reasoning");
        assert_eq!(items[1]["content"][0]["text"], "partial");
        let obj = ctx(None).object(status_of(&c.finish), items, &c.finish);
        assert_eq!(obj["status"], "incomplete");
        assert_eq!(obj["usage"]["total_tokens"], 14);
    }

    #[tokio::test]
    async fn stream_emits_ordered_events_and_stores_conversation() {
        let store = Store::default();
        let chunks = vec![
            json!({"choices": [{"delta": {"content": "Hel"}}]}),
            json!({"choices": [{"delta": {"content": "lo"}}]}),
            json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "f", "arguments": "{}"}}]}}]}),
            json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
        ];
        let input = stream::iter(chunks.into_iter().map(|c| Ok(c.to_string())));
        let bytes: Vec<Bytes> = stream_events(input, ctx(Some((store.clone(), vec![json!({"role": "user", "content": "x"})]))))
            .map(|b| b.unwrap())
            .collect()
            .await;
        let text: String = bytes.iter().map(|b| String::from_utf8_lossy(b).into_owned()).collect();
        let names: Vec<&str> = text.lines().filter_map(|l| l.strip_prefix("event: ")).collect();
        assert_eq!(
            names,
            [
                "response.created", "response.in_progress",
                "response.output_item.added", "response.content_part.added",
                "response.output_text.delta", "response.output_text.delta",
                "response.output_text.done", "response.content_part.done", "response.output_item.done",
                "response.output_item.added", "response.function_call_arguments.delta",
                "response.function_call_arguments.done", "response.output_item.done",
                "response.completed",
            ]
        );
        let seqs: Vec<u64> = text
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .map(|d| serde_json::from_str::<Value>(d).unwrap()["sequence_number"].as_u64().unwrap())
            .collect();
        assert!(seqs.windows(2).all(|w| w[1] == w[0] + 1), "sequence numbers are consecutive");
        assert!(text.contains(r#""text":"Hello""#));

        let conv = store.get("resp_1").unwrap();
        assert_eq!(conv[1]["content"], "Hello");
        assert_eq!(conv[1]["tool_calls"][0]["id"], "c1");
    }
}
