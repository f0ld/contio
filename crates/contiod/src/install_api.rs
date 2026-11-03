//! Management API for background jobs: model downloads and backend installs
//! (spec §6, §7.4).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::Json;
use axum::extract::{Path as UrlPath, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use contio_backends::{Backend, FlmBackend, FlmConfig, LlamaCppBackend, LlamaCppConfig, LlamaRuntime};
use contio_install::download::{self, hf_remote};
use contio_install::{
    BackendKind, FLM_DEFAULT_VERSION, InstalledBackend, LLAMACPP_DEFAULT_VERSION, Method, discover, flm_preflight,
    install_flm, install_llamacpp, llamacpp_preflight,
};
use serde::Deserialize;
use serde_json::json;

use crate::api::AppState;

const HF_BASE: &str = "https://huggingface.co";

/// Shared state for installs and downloads.
#[derive(Clone)]
pub struct Installer {
    pub jobs: contio_install::jobs::Jobs,
    pub data_dir: PathBuf,
    /// Follows redirects (downloads, GitHub).
    pub http: reqwest::Client,
    /// Doesn't follow redirects: HF metadata lives on the 302 response.
    pub http_no_redirect: reqwest::Client,
    pub cpu_threads: u32,
}

impl Installer {
    pub fn new(data_dir: PathBuf, cpu_threads: u32) -> Self {
        Self {
            jobs: Default::default(),
            data_dir,
            http: reqwest::Client::builder().user_agent("contio").build().expect("static TLS config"),
            http_no_redirect: reqwest::Client::builder()
                .user_agent("contio")
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .expect("static TLS config"),
            cpu_threads,
        }
    }

    pub fn models_dir(&self) -> PathBuf {
        self.data_dir.join("models")
    }
}

/// Build a runnable backend from an install marker.
pub fn backend_from_installed(b: &InstalledBackend, installer: &Installer) -> Option<Arc<dyn Backend>> {
    match b.backend {
        BackendKind::LlamaCpp => {
            let runtime = match (&b.bin_dir, &b.image) {
                (Some(bin_dir), _) => LlamaRuntime::Native { bin_dir: bin_dir.clone() },
                (None, Some(image)) => LlamaRuntime::Podman { image: image.clone() },
                _ => return None,
            };
            Some(Arc::new(LlamaCppBackend::new(LlamaCppConfig { runtime, cpu_threads: Some(installer.cpu_threads) })))
        }
        BackendKind::Flm => Some(Arc::new(FlmBackend::new(FlmConfig {
            prefix: b.prefix.clone()?,
            model_path: installer.models_dir().join("flm"),
            xrt_dir: "/opt/xilinx/xrt".into(),
            embed: true,
        }))),
    }
}

// --- jobs ---------------------------------------------------------------------

pub async fn list_jobs(State(state): State<AppState>) -> Response {
    Json(state.installer.jobs.list()).into_response()
}

pub async fn get_job(State(state): State<AppState>, UrlPath(id): UrlPath<u64>) -> Response {
    match state.installer.jobs.get(id) {
        Some(j) => Json(j).into_response(),
        None => not_found(&format!("no job {id}")),
    }
}

pub async fn cancel_job(State(state): State<AppState>, UrlPath(id): UrlPath<u64>) -> Response {
    Json(json!({ "cancelled": state.installer.jobs.cancel(id) })).into_response()
}

// --- downloads ----------------------------------------------------------------

/// Either a Hugging Face GGUF (`repo` + `file`) or a FastFlowLM tag (`flm`).
#[derive(Debug, Deserialize)]
pub struct DownloadRequest {
    #[serde(default)]
    pub repo: String,
    #[serde(default)]
    pub file: String,
    #[serde(default = "main")]
    pub revision: String,
    #[serde(default)]
    pub flm: Option<String>,
}

fn main() -> String {
    "main".into()
}

/// `owner/name` with conservative characters (it becomes a path).
fn valid_repo(repo: &str) -> bool {
    let parts: Vec<&str> = repo.split('/').collect();
    parts.len() == 2
        && parts.iter().all(|p| {
            !p.is_empty() && !p.starts_with('.') && p.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c))
        })
}

/// A relative path inside the repo without `..` or absolute components.
fn valid_file(file: &str) -> bool {
    file.split('/').all(|seg| !seg.is_empty() && seg != "." && seg != "..") && file.chars().all(|c| !c.is_control())
}

pub async fn download_model(State(state): State<AppState>, Json(req): Json<DownloadRequest>) -> Response {
    if let Some(tag) = req.flm {
        return pull_flm(state, tag);
    }
    if !valid_repo(&req.repo) || !valid_file(&req.file) || !valid_file(&req.revision) {
        return bad_request("repo must be `owner/name`; file a relative path without `..`");
    }
    let inst = state.installer.clone();
    let dest = inst.models_dir().join("hf").join(&req.repo).join(&req.file);
    let title = format!("{}/{}", req.repo, req.file);
    let key = Some(dest.display().to_string());
    let jobs = inst.jobs.clone();
    let id = jobs.spawn("download", &title, key, move |ctx| async move {
        ctx.step("looking up file");
        let token = std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty());
        let remote = hf_remote(&inst.http_no_redirect, HF_BASE, &req.repo, &req.file, &req.revision, token).await?;
        if remote.sha256.is_none() {
            ctx.log("no sha256 published for this file; only the size will be checked");
        }
        download::fetch(&inst.http, &remote, &dest, &ctx).await?;
        let stem = Path::new(&req.file).file_stem().and_then(|s| s.to_str()).unwrap_or("model").to_lowercase();
        Ok(json!({
            "path": dest,
            "sha256": remote.sha256,
            "size": remote.size,
            "config_snippet": format!(
                "[[models]]\nid = \"{stem}\"\nformat = \"gguf\"\npath = \"{}\"\ndevice = \"gpu\"\n",
                dest.display()
            ),
        }))
    });
    (StatusCode::ACCEPTED, Json(json!({ "job_id": id }))).into_response()
}

/// `flm pull <tag>` as a job, with the installed FLM and contio's env.
fn pull_flm(state: AppState, tag: String) -> Response {
    if tag.is_empty() || !tag.chars().all(|c| c.is_ascii_alphanumeric() || "-_.:".contains(c)) {
        return bad_request("invalid FastFlowLM tag");
    }
    let Some(installed) = discover(&state.installer.data_dir).into_iter().find(|b| b.backend == BackendKind::Flm) else {
        return bad_request("FastFlowLM is not installed (contio install flm --method source)");
    };
    let Some(prefix) = installed.prefix else { return bad_request("FastFlowLM install has no prefix") };
    let flm = FlmBackend::new(FlmConfig {
        prefix,
        model_path: state.installer.models_dir().join("flm"),
        xrt_dir: "/opt/xilinx/xrt".into(),
        embed: false,
    });
    let jobs = state.installer.jobs.clone();
    let key = Some(format!("flm-pull:{tag}"));
    let id = jobs.spawn("download", &format!("flm {tag}"), key, move |ctx| async move {
        let mut cmd = flm.flm_command().map_err(|e| e.to_string())?;
        cmd.args(["pull", &tag]);
        ctx.step("flm pull");
        contio_install::run_logged(&ctx, &mut cmd).await?;
        Ok(json!({
            "flm_tag": tag,
            "config_snippet": format!(
                "[[models]]\nid = \"{}-npu\"\nformat = \"flm\"\ntag = \"{tag}\"\ndevice = \"npu\"\n",
                tag.replace([':', '.'], "-")
            ),
        }))
    });
    (StatusCode::ACCEPTED, Json(json!({ "job_id": id }))).into_response()
}

// --- backends -----------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct BackendQuery {
    pub backend: BackendKind,
    pub method: Method,
}

pub async fn list_backends(State(state): State<AppState>) -> Response {
    let installed = discover(&state.installer.data_dir);
    Json(json!({
        "registered": state.manager.backend_names(),
        "installed": installed,
        "defaults": { "llama.cpp": LLAMACPP_DEFAULT_VERSION, "flm": FLM_DEFAULT_VERSION },
    }))
    .into_response()
}

pub async fn preflight(Query(q): Query<BackendQuery>) -> Response {
    let pf = tokio::task::spawn_blocking(move || match q.backend {
        BackendKind::LlamaCpp => llamacpp_preflight(q.method),
        BackendKind::Flm => flm_preflight(q.method),
    })
    .await
    .unwrap_or_default();
    Json(json!({ "ok": pf.ok() && pf.notes.is_empty(), "install_hint": pf.install_hint(), "preflight": pf }))
        .into_response()
}

#[derive(Debug, Deserialize)]
pub struct InstallRequest {
    pub backend: BackendKind,
    pub method: Method,
    #[serde(default)]
    pub version: Option<String>,
}

pub async fn install_backend(State(state): State<AppState>, Json(req): Json<InstallRequest>) -> Response {
    let version = req.version.clone().unwrap_or_else(|| match req.backend {
        BackendKind::LlamaCpp => LLAMACPP_DEFAULT_VERSION.into(),
        BackendKind::Flm => FLM_DEFAULT_VERSION.into(),
    });
    if !version.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c)) || version.starts_with('.') {
        return bad_request("invalid version");
    }
    let inst = state.installer.clone();
    let manager = state.manager.clone();
    let title = format!("{} {version} ({:?})", req.backend.dir_name(), req.method).to_lowercase();
    let key = Some(format!("install:{}", req.backend.dir_name()));
    let jobs = inst.jobs.clone();
    let id = jobs.spawn("install", &title, key, move |ctx| async move {
        let installed = match req.backend {
            BackendKind::LlamaCpp => install_llamacpp(&inst.http, &inst.data_dir, req.method, &version, &ctx).await?,
            BackendKind::Flm => install_flm(&inst.data_dir, req.method, &version, &ctx).await?,
        };
        if let Some(b) = backend_from_installed(&installed, &inst) {
            manager.add_backend(b);
        }
        serde_json::to_value(&installed).map_err(|e| e.to_string())
    });
    (StatusCode::ACCEPTED, Json(json!({ "job_id": id }))).into_response()
}

fn bad_request(msg: &str) -> Response {
    (StatusCode::BAD_REQUEST, Json(json!({"error": {"message": msg, "type": "invalid_request_error"}}))).into_response()
}

fn not_found(msg: &str) -> Response {
    (StatusCode::NOT_FOUND, Json(json!({"error": {"message": msg, "type": "not_found"}}))).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_path_traversal() {
        assert!(valid_repo("bartowski/Llama-3.2-1B-Instruct-GGUF"));
        for bad in ["a", "a/b/c", "../x", "a/..", "a/.hidden", "a/b c", ""] {
            assert!(!valid_repo(bad), "{bad}");
        }
        assert!(valid_file("Q4_K_M/model-00001-of-00002.gguf"));
        for bad in ["../etc/passwd", "/etc/passwd", "a/../../b", "", "a/./b", "x\u{0}y"] {
            assert!(!valid_file(bad), "{bad:?}");
        }
    }

    #[test]
    fn backends_from_markers() {
        let inst = Installer::new("/data".into(), 12);
        let mut b = InstalledBackend {
            backend: BackendKind::LlamaCpp,
            method: Method::Podman,
            version: "b1".into(),
            dir: "/d".into(),
            bin_dir: None,
            image: Some("img".into()),
            prefix: None,
            installed_at: 1,
        };
        assert_eq!(backend_from_installed(&b, &inst).unwrap().name(), "llama.cpp");
        b.image = None;
        assert!(backend_from_installed(&b, &inst).is_none(), "marker without runtime");
        b.backend = BackendKind::Flm;
        b.prefix = Some("/p".into());
        assert_eq!(backend_from_installed(&b, &inst).unwrap().name(), "FastFlowLM");
    }
}
