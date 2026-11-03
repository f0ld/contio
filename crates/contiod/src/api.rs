//! HTTP surface of the daemon: management API (`/api/*`) and the
//! OpenAI-compatible gateway (`/v1/*`, stubbed until backends land in M1).

use std::time::Duration;

use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::http::{HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
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

#[derive(Clone)]
pub struct AppState {
    pub probe: HwProbe,
    pub metrics: watch::Receiver<Option<Metrics>>,
}

impl AppState {
    /// Starts a background sampler that refreshes metrics every `interval`.
    /// All WebSocket clients share this one sampler.
    pub fn start(probe: HwProbe, interval: Duration) -> Self {
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
        Self { probe, metrics: rx }
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
        .route("/v1/models", get(models))
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

/// OpenAI `GET /v1/models`. Empty until the model registry exists.
async fn models() -> Json<serde_json::Value> {
    Json(json!({ "object": "list", "data": [] }))
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
        let app = router(AppState::start(HwProbe::new(dir.path()), Duration::from_secs(60)));

        let (s, body) = get_json(app.clone(), "/api/health").await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(body["status"], "ok");

        let (s, body) = get_json(app.clone(), "/api/hw").await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(body["cpu"]["model"], "Test CPU");
        assert_eq!(body["pools"][0]["kind"], "system");

        let (s, body) = get_json(app, "/v1/models").await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(body["object"], "list");
    }

    #[tokio::test]
    async fn hw_probe_failure_is_a_json_error() {
        let dir = tempfile::tempdir().unwrap(); // no procfs at all
        let app = router(AppState::start(HwProbe::new(dir.path()), Duration::from_secs(60)));
        let (s, body) = get_json(app, "/api/hw").await;
        assert_eq!(s, StatusCode::INTERNAL_SERVER_ERROR);
        assert!(body["error"]["message"].as_str().unwrap().contains("cpuinfo"));
    }

    #[tokio::test]
    async fn cors_allows_gui_origin_only() {
        let dir = fixture();
        let app = router(AppState::start(HwProbe::new(dir.path()), Duration::from_secs(60)));
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
