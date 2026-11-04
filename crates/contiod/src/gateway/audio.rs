//! `/v1/audio/transcriptions` and `/v1/audio/translations` (spec §7.1):
//! OpenAI multipart uploads, forwarded to whisper.cpp or FastFlowLM with the
//! `model` field rewritten.

use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::Response;
use contio_backends::WHISPER_INFERENCE_PATH;
use contio_core::model::ModelSource;
use reqwest::multipart::{Form, Part};

use super::{backend_error, openai_error, relay, resolve};
use crate::api::AppState;

/// Uploads larger than this are rejected (OpenAI allows 25 MB).
pub const MAX_UPLOAD: usize = 200 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Task {
    Transcribe,
    Translate,
}

pub async fn transcriptions(state: State<AppState>, form: Multipart) -> Response {
    handle(state, form, Task::Transcribe).await
}

pub async fn translations(state: State<AppState>, form: Multipart) -> Response {
    handle(state, form, Task::Translate).await
}

struct Field {
    name: String,
    file_name: Option<String>,
    content_type: Option<String>,
    data: axum::body::Bytes,
}

fn invalid(message: &str, param: Option<&str>) -> Response {
    openai_error(StatusCode::BAD_REQUEST, "invalid_request_error", message, param)
}

async fn handle(State(state): State<AppState>, mut form: Multipart, task: Task) -> Response {
    let mut fields = Vec::new();
    loop {
        match form.next_field().await {
            Ok(Some(f)) => {
                let name = f.name().unwrap_or_default().to_owned();
                let file_name = f.file_name().map(str::to_owned);
                let content_type = f.content_type().map(str::to_owned);
                match f.bytes().await {
                    Ok(data) => fields.push(Field { name, file_name, content_type, data }),
                    Err(e) => return invalid(&format!("reading `{name}`: {e}"), None),
                }
            }
            Ok(None) => break,
            Err(e) => return invalid(&format!("bad multipart body: {e}"), None),
        }
    }
    let Some(model) = fields.iter().find(|f| f.name == "model").map(|f| String::from_utf8_lossy(&f.data).into_owned())
    else {
        return invalid("missing `model`", Some("model"));
    };
    if !fields.iter().any(|f| f.name == "file") {
        return invalid("missing `file`", Some("file"));
    }

    let (id, upstream_override) = resolve(&state, &model);
    let profile = match state.manager.profile(&id) {
        Ok(p) => p,
        Err(e) => return backend_error(&e),
    };
    let (path, translate) = match (&profile.source, task) {
        (ModelSource::Whisper { .. }, Task::Transcribe) => (WHISPER_INFERENCE_PATH, false),
        // whisper.cpp translates to English on the same endpoint
        (ModelSource::Whisper { .. }, Task::Translate) => (WHISPER_INFERENCE_PATH, true),
        (ModelSource::Flm { .. }, Task::Transcribe) => ("/v1/audio/transcriptions", false),
        (ModelSource::Flm { .. }, Task::Translate) => {
            return invalid("FastFlowLM's Whisper only transcribes; use a whisper.cpp model to translate", Some("model"));
        }
        (ModelSource::Gguf { .. }, _) => {
            return invalid(&format!("`{model}` is not a speech model"), Some("model"));
        }
    };

    let lease = match state.manager.acquire(&id).await {
        Ok(l) => l,
        Err(e) => return backend_error(&e),
    };
    let upstream_model = upstream_override.unwrap_or_else(|| lease.upstream_model.clone());
    let mut out = Form::new();
    for f in fields {
        if f.name == "model" {
            out = out.text("model", upstream_model.clone());
            continue;
        }
        let make = || {
            let part = Part::bytes(f.data.to_vec());
            match &f.file_name {
                Some(n) => part.file_name(n.clone()),
                None => part,
            }
        };
        // an unparsable content type is dropped rather than failing the upload
        let part = match &f.content_type {
            Some(ct) => make().mime_str(ct).unwrap_or_else(|_| make()),
            None => make(),
        };
        out = out.part(f.name, part);
    }
    if translate {
        out = out.text("translate", "true");
    }
    let url = format!("{}{path}", lease.base_url);
    match state.http.post(&url).multipart(out).send().await {
        Ok(r) => relay(r, lease),
        Err(e) => openai_error(StatusCode::BAD_GATEWAY, "upstream_error", &format!("backend request failed: {e}"), None),
    }
}
