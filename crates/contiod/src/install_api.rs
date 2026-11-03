//! Management API for background jobs: backend installs, job control,
//! settings (spec §6, §7.4). Model installs live in `catalog_api.rs`.

use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use axum::Json;
use axum::extract::{Path as UrlPath, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use contio_backends::{
    Backend, FlmBackend, FlmConfig, LlamaCppBackend, LlamaCppConfig, LlamaRuntime, ModelManager,
};
use contio_install::jobs::{JobCtx, JobInfo, Jobs};
use contio_install::{
    BackendKind, FLM_DEFAULT_VERSION, InstalledBackend, LLAMACPP_DEFAULT_VERSION, Method, discover, flm_preflight,
    install_flm, install_llamacpp, llamacpp_preflight,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::api::AppState;
use crate::catalog_api::{self, InstallModelRequest};
use crate::registry::Settings;

pub const XRT_DIR: &str = "/opt/xilinx/xrt";

/// Shared state for installs and downloads.
#[derive(Clone)]
pub struct Installer {
    pub jobs: Jobs,
    pub data_dir: PathBuf,
    /// Follows redirects (downloads, GitHub).
    pub http: reqwest::Client,
    pub cpu_threads: u32,
    /// The FastFlowLM install models are pulled with and served from; set
    /// from `config.toml`, an installed backend, or a fresh install.
    flm: Arc<RwLock<Option<FlmConfig>>>,
}

impl Installer {
    /// Jobs are kept in `<data_dir>/jobs.json`; returns the jobs a previous
    /// daemon left unfinished.
    pub fn persistent(data_dir: PathBuf, cpu_threads: u32) -> (Self, Vec<JobInfo>) {
        let (jobs, interrupted) = Jobs::persistent(data_dir.join("jobs.json"));
        (Self::with_jobs(jobs, data_dir, cpu_threads), interrupted)
    }

    /// Jobs in memory only (tests).
    #[cfg(test)]
    pub fn in_memory(data_dir: PathBuf, cpu_threads: u32) -> Self {
        Self::with_jobs(Jobs::default(), data_dir, cpu_threads)
    }

    fn with_jobs(jobs: Jobs, data_dir: PathBuf, cpu_threads: u32) -> Self {
        Self {
            jobs,
            data_dir,
            http: reqwest::Client::builder().user_agent("contio").build().expect("static TLS config"),
            cpu_threads,
            flm: Default::default(),
        }
    }

    pub fn models_dir(&self) -> PathBuf {
        self.data_dir.join("models")
    }

    pub fn flm(&self) -> Option<FlmConfig> {
        self.flm.read().expect("flm lock poisoned").clone()
    }

    pub fn set_flm(&self, config: FlmConfig) {
        *self.flm.write().expect("flm lock poisoned") = Some(config);
    }

    /// FLM settings for a contio-installed FLM.
    fn flm_config(&self, prefix: PathBuf) -> FlmConfig {
        FlmConfig { prefix, model_path: self.models_dir().join("flm"), xrt_dir: XRT_DIR.into(), embed: true }
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
        BackendKind::Flm => {
            let config = installer.flm_config(b.prefix.clone()?);
            installer.set_flm(config.clone());
            Some(Arc::new(FlmBackend::new(config)))
        }
    }
}

/// Install a backend and register it with the manager right away.
pub async fn install_and_register(
    inst: &Installer,
    manager: &ModelManager,
    backend: BackendKind,
    method: Method,
    version: &str,
    ctx: &JobCtx,
) -> Result<InstalledBackend, String> {
    let installed = match backend {
        BackendKind::LlamaCpp => install_llamacpp(&inst.http, &inst.data_dir, method, version, ctx).await?,
        BackendKind::Flm => install_flm(&inst.data_dir, method, version, ctx).await?,
    };
    if let Some(b) = backend_from_installed(&installed, inst) {
        manager.add_backend(b);
    }
    Ok(installed)
}

// --- jobs ---------------------------------------------------------------------

/// What a job was started with; stored with the job for retry/resume.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum JobSpec {
    InstallBackend(InstallRequest),
    InstallModel(InstallModelRequest),
}

/// Start a job from its spec. `Err` is a client error.
pub fn start_job(state: &AppState, spec: JobSpec) -> Result<u64, String> {
    match spec {
        JobSpec::InstallBackend(req) => start_backend_install(state, req),
        JobSpec::InstallModel(req) => catalog_api::start_install(state, req),
    }
}

/// Start the jobs a previous daemon left unfinished (downloads resume).
pub fn resume_interrupted(state: &AppState, interrupted: Vec<JobInfo>) {
    for job in interrupted {
        let Some(spec) = job.spec.clone().and_then(|s| serde_json::from_value::<JobSpec>(s).ok()) else {
            continue;
        };
        match start_job(state, spec) {
            Ok(new) => {
                tracing::info!(old = job.id, new, "resuming interrupted job");
                state.installer.jobs.mark_retried(job.id, new);
            }
            Err(e) => tracing::warn!(job = job.id, "can't resume: {e}"),
        }
    }
}

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

/// Start a finished (failed/cancelled/done) job again with the same request.
pub async fn retry_job(State(state): State<AppState>, UrlPath(id): UrlPath<u64>) -> Response {
    let Some(job) = state.installer.jobs.get(id) else { return not_found(&format!("no job {id}")) };
    if job.state == contio_install::jobs::JobState::Running {
        return bad_request("job is still running");
    }
    let Some(spec) = job.spec.and_then(|s| serde_json::from_value::<JobSpec>(s).ok()) else {
        return bad_request("this job can't be retried");
    };
    match start_job(&state, spec) {
        Ok(new) => {
            state.installer.jobs.mark_retried(id, new);
            accepted(new)
        }
        Err(e) => bad_request(&e),
    }
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
        "settings": state.registry.settings(),
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallRequest {
    pub backend: BackendKind,
    pub method: Method,
    #[serde(default)]
    pub version: Option<String>,
}

pub async fn install_backend(State(state): State<AppState>, Json(req): Json<InstallRequest>) -> Response {
    match start_backend_install(&state, req) {
        Ok(id) => accepted(id),
        Err(e) => bad_request(&e),
    }
}

fn start_backend_install(state: &AppState, req: InstallRequest) -> Result<u64, String> {
    let version = req.version.clone().unwrap_or_else(|| match req.backend {
        BackendKind::LlamaCpp => LLAMACPP_DEFAULT_VERSION.into(),
        BackendKind::Flm => FLM_DEFAULT_VERSION.into(),
    });
    if !version.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c)) || version.starts_with('.') {
        return Err("invalid version".into());
    }
    let inst = state.installer.clone();
    let manager = state.manager.clone();
    let title = format!("{} {version} ({:?})", req.backend.dir_name(), req.method).to_lowercase();
    let key = Some(format!("install:{}", req.backend.dir_name()));
    let spec = serde_json::to_value(JobSpec::InstallBackend(req.clone())).ok();
    Ok(inst.jobs.clone().spawn_with_spec("install", &title, key, spec, move |ctx| async move {
        let installed = install_and_register(&inst, &manager, req.backend, req.method, &version, &ctx).await?;
        serde_json::to_value(&installed).map_err(|e| e.to_string())
    }))
}

// --- settings -----------------------------------------------------------------

pub async fn get_settings(State(state): State<AppState>) -> Response {
    Json(state.registry.settings()).into_response()
}

pub async fn put_settings(State(state): State<AppState>, Json(settings): Json<Settings>) -> Response {
    if settings.flm_method != Method::Source {
        return bad_request("FastFlowLM can only be installed from source");
    }
    match state.registry.set_settings(settings.clone()) {
        Ok(()) => Json(settings).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": {"message": e.to_string()}})))
            .into_response(),
    }
}

// --- helpers ------------------------------------------------------------------

pub fn accepted(job_id: u64) -> Response {
    (StatusCode::ACCEPTED, Json(json!({ "job_id": job_id }))).into_response()
}

pub fn bad_request(msg: &str) -> Response {
    (StatusCode::BAD_REQUEST, Json(json!({"error": {"message": msg, "type": "invalid_request_error"}}))).into_response()
}

pub fn not_found(msg: &str) -> Response {
    (StatusCode::NOT_FOUND, Json(json!({"error": {"message": msg, "type": "not_found"}}))).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backends_from_markers() {
        let inst = Installer::in_memory("/data".into(), 12);
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
        assert!(inst.flm().is_none());
        b.backend = BackendKind::Flm;
        b.prefix = Some("/p".into());
        assert_eq!(backend_from_installed(&b, &inst).unwrap().name(), "FastFlowLM");
        let flm = inst.flm().expect("FLM config remembered for pulls");
        assert_eq!(flm.model_path, PathBuf::from("/data/models/flm"));
    }

    #[test]
    fn job_specs_round_trip() {
        let spec = JobSpec::InstallBackend(InstallRequest {
            backend: BackendKind::LlamaCpp,
            method: Method::Prebuilt,
            version: None,
        });
        let v = serde_json::to_value(&spec).unwrap();
        assert_eq!(v["type"], "install_backend");
        assert_eq!(v["backend"], "llama.cpp");
        assert!(matches!(serde_json::from_value::<JobSpec>(v).unwrap(), JobSpec::InstallBackend(_)));
    }
}
