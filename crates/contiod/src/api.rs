//! HTTP surface of the daemon: management API (`/api/*`) and the
//! OpenAI-compatible gateway (`/v1/*`, see `gateway.rs`).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Path, State};
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::http::{HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use contio_backends::ModelManager;
use contio_core::hw::{HardwareInfo, Metrics};
use contio_hw::HwProbe;
use serde_json::json;
use tokio::sync::watch;
use tower_http::cors::CorsLayer;

/// Origins of the contio GUI: Vite dev server and the Tauri webview (Linux/Windows).
const GUI_ORIGINS: [&str; 4] = [
    "http://localhost:1420",
    "http://127.0.0.1:1420",
    "tauri://localhost",
    "http://tauri.localhost",
];

use crate::config::Alias;
use crate::install_api::{self, Installer};
use crate::gateway;

#[derive(Clone)]
pub struct AppState {
    pub probe: HwProbe,
    pub metrics: watch::Receiver<Option<Metrics>>,
    pub manager: ModelManager,
    /// Client for backend requests (no overall timeout: generations can be long).
    pub http: reqwest::Client,
    pub aliases: Arc<HashMap<String, Alias>>,
    /// Conversations for the Responses API's `previous_response_id`.
    pub responses: gateway::responses::Store,
    pub installer: Installer,
}

impl AppState {
    /// Starts a background sampler that refreshes metrics every `interval`.
    /// All WebSocket clients share this one sampler.
    pub fn start(
        probe: HwProbe,
        manager: ModelManager,
        aliases: Vec<Alias>,
        installer: Installer,
        interval: Duration,
    ) -> Self {
        let (tx, rx) = watch::channel(None);
        let sampler = probe.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(interval);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tick.tick().await;
                let p = sampler.clone();
                match tokio::task::spawn_blocking(move || p.sample()).await {
                    Ok(Ok(m)) => {
                        if tx.send(Some(m)).is_err() {
                            break; // all receivers (incl. AppState) dropped
                        }
                    }
                    Ok(Err(e)) => tracing::warn!("metrics sample failed: {e}"),
                    Err(e) => tracing::error!("metrics sampler panicked: {e}"),
                }
            }
        });
        Self {
            probe,
            metrics: rx,
            manager,
            http: reqwest::Client::new(),
            aliases: Arc::new(aliases.into_iter().map(|a| (a.name.clone(), a)).collect()),
            responses: gateway::responses::Store::default(),
            installer,
        }
    }
}

pub fn router(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(GUI_ORIGINS.map(HeaderValue::from_static))
        .allow_methods([Method::GET, Method::POST]);

    Router::new()
        .route("/api/health", get(health))
        .route("/api/hw", get(hardware))
        .route("/api/events", get(events))
        .route("/api/models", get(model_statuses))
        .route("/api/models/{id}/load", post(load_model))
        .route("/api/models/{id}/unload", post(unload_model))
        .route("/api/models/download", post(install_api::download_model))
        .route("/api/jobs", get(install_api::list_jobs))
        .route("/api/jobs/{id}", get(install_api::get_job))
        .route("/api/jobs/{id}/cancel", post(install_api::cancel_job))
        .route("/api/backends", get(install_api::list_backends))
        .route("/api/backends/preflight", get(install_api::preflight))
        .route("/api/backends/install", post(install_api::install_backend))
        .route("/v1/models", get(gateway::models))
        .route("/v1/chat/completions", post(gateway::chat))
        .route("/v1/completions", post(gateway::passthrough))
        .route("/v1/embeddings", post(gateway::passthrough))
        .route("/v1/messages", post(gateway::anthropic::messages))
        .route("/v1/messages/count_tokens", post(gateway::anthropic::count_tokens))
        .route("/v1/responses", post(gateway::responses::create))
        .layer(cors)
        .with_state(state)
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({ "status": "ok", "version": env!("CARGO_PKG_VERSION") }))
}

async fn hardware(State(state): State<AppState>) -> Response {
    let probe = state.probe.clone();
    match tokio::task::spawn_blocking(move || probe.probe()).await {
        Ok(Ok(hw)) => Json::<HardwareInfo>(hw).into_response(),
        Ok(Err(e)) => error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

async fn model_statuses(State(state): State<AppState>) -> Response {
    Json(state.manager.statuses()).into_response()
}

/// Load (or keep) a model without sending a request; returns its status.
async fn load_model(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    match state.manager.acquire(&id).await {
        Ok(lease) => {
            drop(lease);
            let status = state.manager.statuses().into_iter().find(|s| s.id == id);
            Json(status).into_response()
        }
        Err(e) => gateway::backend_error(&e),
    }
}

async fn unload_model(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    match state.manager.unload(&id).await {
        Ok(unloaded) => Json(json!({ "unloaded": unloaded })).into_response(),
        Err(e) => gateway::backend_error(&e),
    }
}

/// Live metrics stream: one JSON `Metrics` message per sample.
async fn events(ws: WebSocketUpgrade, State(state): State<AppState>) -> Response {
    ws.on_upgrade(move |socket| stream_metrics(socket, state.metrics))
}

async fn stream_metrics(mut socket: WebSocket, mut rx: watch::Receiver<Option<Metrics>>) {
    loop {
        let msg = rx.borrow_and_update().as_ref().and_then(|m| serde_json::to_string(m).ok());
        if let Some(text) = msg
            && socket.send(Message::Text(text.into())).await.is_err()
        {
            return; // client went away
        }
        tokio::select! {
            changed = rx.changed() => if changed.is_err() { return },
            incoming = socket.recv() => match incoming {
                Some(Ok(Message::Close(_))) | None | Some(Err(_)) => return,
                _ => {}
            },
        }
    }
}

fn error(status: StatusCode, message: &str) -> Response {
    (status, Json(json!({ "error": { "message": message } }))).into_response()
}

#[cfg(test)]
mod tests {
    use std::fs;

    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use super::*;

    fn state(probe: HwProbe) -> AppState {
        let manager = ModelManager::new(
            contio_backends::ManagerConfig {
                log_dir: std::env::temp_dir(),
                startup_timeout: Duration::from_secs(1),
                default_idle_ttl: None,
                pool_budgets: Default::default(),
            },
            vec![],
            vec![],
        );
        AppState::start(probe, manager, vec![], Installer::new(std::env::temp_dir(), 4), Duration::from_secs(60))
    }

    fn fixture() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("proc")).unwrap();
        fs::write(dir.path().join("proc/cpuinfo"), "processor\t: 0\nmodel name\t: Test CPU\n").unwrap();
        fs::write(dir.path().join("proc/meminfo"), "MemTotal: 2048 kB\nMemAvailable: 1024 kB\n").unwrap();
        dir
    }

    async fn get_json(app: Router, uri: &str) -> (StatusCode, serde_json::Value) {
        let res = app.oneshot(Request::get(uri).body(Body::empty()).unwrap()).await.unwrap();
        let status = res.status();
        let body = res.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&body).unwrap())
    }

    #[tokio::test]
    async fn serves_health_hw_and_models() {
        let dir = fixture();
        let app = router(state(HwProbe::new(dir.path())));

        let (s, body) = get_json(app.clone(), "/api/health").await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(body["status"], "ok");

        let (s, body) = get_json(app.clone(), "/api/hw").await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(body["cpu"]["model"], "Test CPU");
        assert_eq!(body["pools"][0]["kind"], "system");

        let (s, body) = get_json(app.clone(), "/v1/models").await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(body["object"], "list");

        let (s, body) = get_json(app, "/api/models").await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(body, serde_json::json!([]));
    }

    #[tokio::test]
    async fn gateway_rejects_bad_requests_in_openai_shape() {
        let dir = fixture();
        let app = router(state(HwProbe::new(dir.path())));
        let post_json = |body: &'static str| {
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap()
        };
        for (body, status, kind) in [
            ("not json", StatusCode::BAD_REQUEST, "invalid_request_error"),
            (r#"{"messages":[]}"#, StatusCode::BAD_REQUEST, "invalid_request_error"),
            (r#"{"model":"nope","messages":[]}"#, StatusCode::NOT_FOUND, "model_not_found"),
        ] {
            let res = app.clone().oneshot(post_json(body)).await.unwrap();
            assert_eq!(res.status(), status, "{body}");
            let v: serde_json::Value =
                serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
            assert_eq!(v["error"]["type"], kind, "{body}");
        }
    }

    #[tokio::test]
    async fn hw_probe_failure_is_a_json_error() {
        let dir = tempfile::tempdir().unwrap(); // no procfs at all
        let app = router(state(HwProbe::new(dir.path())));
        let (s, body) = get_json(app, "/api/hw").await;
        assert_eq!(s, StatusCode::INTERNAL_SERVER_ERROR);
        assert!(body["error"]["message"].as_str().unwrap().contains("cpuinfo"));
    }

    #[tokio::test]
    async fn cors_allows_gui_origin_only() {
        let dir = fixture();
        let app = router(state(HwProbe::new(dir.path())));
        for (origin, allowed) in [("tauri://localhost", true), ("http://evil.example", false)] {
            let res = app
                .clone()
                .oneshot(Request::get("/api/health").header("origin", origin).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(res.headers().contains_key("access-control-allow-origin"), allowed, "{origin}");
        }
    }
}
