//! OpenAI-compatible gateway (spec §7.1–7.2): route by `model`, load on
//! demand, forward to the backend, and stream the response through.

use axum::body::{Body, Bytes};
use axum::extract::{OriginalUri, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::Json;
use contio_backends::BackendError;
use contio_core::model::ModelState;
use futures_util::StreamExt;
use serde_json::{Value, json};

use crate::api::AppState;

/// `POST /v1/chat/completions`, `/v1/completions`, `/v1/embeddings`.
pub async fn proxy(State(state): State<AppState>, OriginalUri(uri): OriginalUri, body: Bytes) -> Response {
    let mut request: Value = match serde_json::from_slice(&body) {
        Ok(v @ Value::Object(_)) => v,
        _ => return openai_error(StatusCode::BAD_REQUEST, "invalid_request_error", "body must be a JSON object", None),
    };
    let Some(model) = request.get("model").and_then(Value::as_str).map(str::to_owned) else {
        return openai_error(StatusCode::BAD_REQUEST, "invalid_request_error", "missing `model`", Some("model"));
    };

    let lease = match state.manager.acquire(&model).await {
        Ok(lease) => lease,
        Err(e) => return backend_error(&e),
    };
    request["model"] = Value::String(lease.upstream_model.clone());
    let url = format!("{}{}", lease.base_url, uri.path());
    forward(&state.http, &url, request, lease).await
}

/// Send `request` to `url` and stream the upstream response back unchanged.
/// `guard` lives until the response body is fully sent (or the client
/// disconnects), which keeps the model's lease alive for streaming responses.
pub async fn forward<G: Send + 'static>(http: &reqwest::Client, url: &str, request: Value, guard: G) -> Response {
    let upstream = match http.post(url).json(&request).send().await {
        Ok(r) => r,
        Err(e) => {
            return openai_error(
                StatusCode::BAD_GATEWAY,
                "upstream_error",
                &format!("backend request failed: {e}"),
                None,
            );
        }
    };
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

/// OpenAI `GET /v1/models`, with contio details under a `contio` key.
pub async fn models(State(state): State<AppState>) -> Json<Value> {
    let data: Vec<Value> = state
        .manager
        .statuses()
        .into_iter()
        .map(|s| {
            json!({
                "id": s.id,
                "object": "model",
                "created": 0,
                "owned_by": "contio",
                "contio": {
                    "device": s.device,
                    "backend": s.backend,
                    "loaded": matches!(s.state, ModelState::Ready { .. }),
                }
            })
        })
        .collect();
    Json(json!({ "object": "list", "data": data }))
}

pub fn backend_error(e: &BackendError) -> Response {
    let (status, kind, param) = match e {
        BackendError::UnknownModel(_) => (StatusCode::NOT_FOUND, "model_not_found", Some("model")),
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

#[cfg(test)]
mod tests {
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

    async fn body_text(r: Response) -> String {
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
    async fn forwards_json_and_preserves_fields() {
        let url = upstream().await;
        let req = json!({"model": "upstream-name", "messages": [{"role": "user", "content": "hi"}], "tools": []});
        let r = forward(&reqwest::Client::new(), &url, req.clone(), ()).await;
        assert_eq!(r.status(), StatusCode::OK);
        let v: Value = serde_json::from_str(&body_text(r).await).unwrap();
        assert_eq!(v["echo"], req);
    }

    #[tokio::test]
    async fn streams_sse_and_holds_guard_until_body_is_consumed() {
        let url = upstream().await;
        let dropped = Arc::new(AtomicBool::new(false));
        let r = forward(&reqwest::Client::new(), &url, json!({"model": "m", "stream": true}), Guard(dropped.clone()))
            .await;
        assert_eq!(r.headers()[header::CONTENT_TYPE], "text/event-stream");
        assert!(!dropped.load(Ordering::SeqCst), "lease released before the stream was read");
        let text = body_text(r).await;
        assert!(text.contains("{\"n\":2}") && text.ends_with("data: [DONE]\n\n"), "{text}");
        assert!(dropped.load(Ordering::SeqCst), "lease must be released after the stream ends");
    }

    #[tokio::test]
    async fn passes_upstream_errors_through() {
        let url = upstream().await;
        let r = forward(&reqwest::Client::new(), &url, json!({"model": "boom"}), ()).await;
        assert_eq!(r.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn unreachable_upstream_is_bad_gateway() {
        let r = forward(&reqwest::Client::new(), "http://127.0.0.1:1/v1/chat/completions", json!({}), ()).await;
        assert_eq!(r.status(), StatusCode::BAD_GATEWAY);
        assert!(body_text(r).await.contains("upstream_error"));
    }
}
