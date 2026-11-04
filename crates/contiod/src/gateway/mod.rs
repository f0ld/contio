//! OpenAI-compatible gateway (spec §7.1–7.2): resolve the `model` (alias or
//! id), load on demand, forward to the backend, and stream the response back.
//!
//! `/v1/chat/completions` goes through the guards (`guards.rs`); the
//! Anthropic and Responses APIs are translated onto it (`anthropic.rs`,
//! `responses.rs`); other endpoints are passed through unchanged.

pub mod anthropic;
pub mod audio;
mod guards;
mod normalize;
pub mod responses;
mod sse;

use axum::Json;
use axum::body::{Body, Bytes};
use axum::extract::{OriginalUri, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use contio_backends::{BackendError, Lease};
use contio_core::model::ModelState;
use futures_util::{Stream, StreamExt};
use serde_json::{Value, json};

use crate::api::AppState;

/// An in-flight upstream response plus the lease that keeps its model loaded.
pub struct Upstream {
    pub lease: Lease,
    pub response: reqwest::Response,
}

/// Resolve `request.model`, make sure the model runs, rewrite the model name for
/// the backend and send the request to `path` on it.
pub async fn send(state: &AppState, mut request: Value, path: &str) -> Result<Upstream, Response> {
    let Some(name) = request.get("model").and_then(Value::as_str) else {
        return Err(openai_error(StatusCode::BAD_REQUEST, "invalid_request_error", "missing `model`", Some("model")));
    };
    let (id, upstream_override) = resolve(state, name);
    let lease = state.manager.acquire(&id).await.map_err(|e| backend_error(&e))?;
    request["model"] = Value::String(upstream_override.unwrap_or_else(|| lease.upstream_model.clone()));
    let url = format!("{}{path}", lease.base_url);
    match state.http.post(&url).json(&request).send().await {
        Ok(response) => Ok(Upstream { lease, response }),
        Err(e) => Err(openai_error(
            StatusCode::BAD_GATEWAY,
            "upstream_error",
            &format!("backend request failed: {e}"),
            None,
        )),
    }
}

/// A model name or alias → (model id, `model` to send upstream instead of the backend's own).
pub fn resolve(state: &AppState, name: &str) -> (String, Option<String>) {
    match state.aliases.get(name) {
        Some(a) => (a.model, a.upstream_model),
        None => (name.to_owned(), None),
    }
}

/// `POST /v1/chat/completions` with guards (spec §7.2).
pub async fn chat(State(state): State<AppState>, body: Bytes) -> Response {
    let request = match parse_object(&body) {
        Ok(v) => v,
        Err(r) => return r,
    };
    let streaming = request["stream"] == true;
    let has_tools = request["tools"].as_array().is_some_and(|t| !t.is_empty());
    let up = match send(&state, request.clone(), "/v1/chat/completions").await {
        Ok(up) => up,
        Err(r) => return r,
    };
    if !up.response.status().is_success() {
        return relay(up.response, up.lease);
    }

    if !streaming {
        let status = up.response.status().as_u16();
        let bytes = match up.response.bytes().await {
            Ok(b) => b,
            Err(e) => return openai_error(StatusCode::BAD_GATEWAY, "upstream_error", &e.to_string(), None),
        };
        drop(up.lease);
        let Ok(mut resp) = serde_json::from_slice::<Value>(&bytes) else {
            return (StatusCode::from_u16(status).unwrap_or(StatusCode::OK), bytes).into_response();
        };
        guards::apply(&request, &mut resp);
        return Json(resp).into_response();
    }

    if has_tools {
        let lease = up.lease;
        let events = sse::data_events(up.response.bytes_stream());
        let body = guards::rewrite_tool_stream(events).map(move |chunk| {
            let _keep_alive = &lease;
            chunk
        });
        return sse_response(body);
    }
    relay(up.response, up.lease)
}

/// `POST /v1/completions`, `/v1/embeddings`: unchanged passthrough.
pub async fn passthrough(State(state): State<AppState>, OriginalUri(uri): OriginalUri, body: Bytes) -> Response {
    let request = match parse_object(&body) {
        Ok(v) => v,
        Err(r) => return r,
    };
    match send(&state, request, uri.path()).await {
        Ok(up) => relay(up.response, up.lease),
        Err(r) => r,
    }
}

/// Stream an upstream response back unchanged. `guard` lives until the body
/// is fully sent (or the client disconnects).
pub fn relay<G: Send + 'static>(upstream: reqwest::Response, guard: G) -> Response {
    let status = StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let content_type = upstream.headers().get(reqwest::header::CONTENT_TYPE).cloned();
    let stream = upstream.bytes_stream().map(move |chunk| {
        let _keep_alive = &guard;
        chunk
    });
    let mut response = Response::new(Body::from_stream(stream));
    *response.status_mut() = status;
    if let Some(ct) = content_type.and_then(|v| HeaderValue::from_bytes(v.as_bytes()).ok()) {
        response.headers_mut().insert(header::CONTENT_TYPE, ct);
    }
    response
}

pub fn sse_response<S, E>(body: S) -> Response
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: Into<Box<dyn std::error::Error + Send + Sync>>,
{
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(body))
        .expect("static headers are valid")
}

/// OpenAI `GET /v1/models`: models and aliases, with contio details under `contio`.
pub async fn models(State(state): State<AppState>) -> Json<Value> {
    let statuses = state.manager.statuses();
    let mut data: Vec<Value> = statuses
        .iter()
        .map(|s| {
            json!({
                "id": s.id, "object": "model", "created": 0, "owned_by": "contio",
                "contio": {
                    "device": s.device,
                    "backend": s.backend,
                    "loaded": matches!(s.state, ModelState::Ready { .. }),
                }
            })
        })
        .collect();
    for a in state.aliases.list() {
        data.push(json!({
            "id": a.name, "object": "model", "created": 0, "owned_by": "contio",
            "contio": { "alias_of": a.model }
        }));
    }
    Json(json!({ "object": "list", "data": data }))
}

fn parse_object(body: &[u8]) -> Result<Value, Response> {
    match serde_json::from_slice(body) {
        Ok(v @ Value::Object(_)) => Ok(v),
        _ => Err(openai_error(StatusCode::BAD_REQUEST, "invalid_request_error", "body must be a JSON object", None)),
    }
}

pub fn backend_error(e: &BackendError) -> Response {
    let (status, kind, param) = match e {
        BackendError::UnknownModel(_) => (StatusCode::NOT_FOUND, "model_not_found", Some("model")),
        BackendError::DuplicateModel(_) => (StatusCode::CONFLICT, "model_exists", None),
        BackendError::NotManaged(_) => (StatusCode::CONFLICT, "model_in_config", None),
        BackendError::InsufficientMemory { .. } => (StatusCode::SERVICE_UNAVAILABLE, "insufficient_memory", None),
        BackendError::StartFailed { .. } => (StatusCode::SERVICE_UNAVAILABLE, "backend_start_failed", None),
        BackendError::Unsupported(_) | BackendError::MissingBinary(_) | BackendError::MissingModelFile(_) => {
            (StatusCode::SERVICE_UNAVAILABLE, "backend_unavailable", None)
        }
        BackendError::Io(_) => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error", None),
    };
    openai_error(status, kind, &e.to_string(), param)
}

/// Error body in the shape OpenAI clients parse.
pub fn openai_error(status: StatusCode, kind: &str, message: &str, param: Option<&str>) -> Response {
    let body = json!({ "error": { "message": message, "type": kind, "param": param, "code": kind } });
    (status, Json(body)).into_response()
}

/// Random-enough unique ids for response objects (`msg_…`, `resp_…`).
pub fn new_id(prefix: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or_default();
    format!("{prefix}_{nanos:x}{:04x}", COUNTER.fetch_add(1, Ordering::Relaxed) & 0xffff)
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use axum::Router;
    use axum::routing::post;
    use http_body_util::BodyExt;

    use super::*;

    /// Upstream that echoes the request, or streams SSE when asked.
    async fn upstream() -> String {
        async fn chat(Json(req): Json<Value>) -> Response {
            if req["stream"] == true {
                let chunks = ["data: {\"n\":1}\n\n", "data: {\"n\":2}\n\n", "data: [DONE]\n\n"];
                let body = Body::from_stream(futures_util::stream::iter(
                    chunks.map(|c| Ok::<_, std::io::Error>(Bytes::from(c))),
                ));
                return ([(header::CONTENT_TYPE, "text/event-stream")], body).into_response();
            }
            if req["model"] == "boom" {
                return (StatusCode::BAD_REQUEST, Json(json!({"error": {"message": "bad"}}))).into_response();
            }
            Json(json!({ "echo": req })).into_response()
        }
        let app = Router::new().route("/v1/chat/completions", post(chat));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}/v1/chat/completions")
    }

    async fn send_json(url: &str, body: Value) -> reqwest::Response {
        reqwest::Client::new().post(url).json(&body).send().await.unwrap()
    }

    pub(crate) async fn body_text(r: Response) -> String {
        String::from_utf8(r.into_body().collect().await.unwrap().to_bytes().to_vec()).unwrap()
    }

    /// Records when it is dropped.
    struct Guard(Arc<AtomicBool>);
    impl Drop for Guard {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn relays_json_unchanged() {
        let url = upstream().await;
        let req = json!({"model": "upstream-name", "messages": [{"role": "user", "content": "hi"}], "tools": []});
        let r = relay(send_json(&url, req.clone()).await, ());
        assert_eq!(r.status(), StatusCode::OK);
        let v: Value = serde_json::from_str(&body_text(r).await).unwrap();
        assert_eq!(v["echo"], req);
    }

    #[tokio::test]
    async fn relays_sse_and_holds_guard_until_body_is_consumed() {
        let url = upstream().await;
        let dropped = Arc::new(AtomicBool::new(false));
        let r = relay(send_json(&url, json!({"model": "m", "stream": true})).await, Guard(dropped.clone()));
        assert_eq!(r.headers()[header::CONTENT_TYPE], "text/event-stream");
        assert!(!dropped.load(Ordering::SeqCst), "lease released before the stream was read");
        let text = body_text(r).await;
        assert!(text.contains("{\"n\":2}") && text.ends_with("data: [DONE]\n\n"), "{text}");
        assert!(dropped.load(Ordering::SeqCst), "lease must be released after the stream ends");
    }

    #[tokio::test]
    async fn relays_upstream_errors() {
        let url = upstream().await;
        let r = relay(send_json(&url, json!({"model": "boom"})).await, ());
        assert_eq!(r.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn ids_are_unique_and_prefixed() {
        let (a, b) = (new_id("msg"), new_id("msg"));
        assert!(a.starts_with("msg_") && a != b);
    }
}
