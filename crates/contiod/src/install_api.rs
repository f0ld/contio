//! Management API for background jobs: backend installs, job control,
//! settings (spec §6, §7.4). Model installs live in `catalog_api.rs`.

use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use axum::Json;
use axum::extract::{Path as UrlPath, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use contio_backends::{
    Backend, FlmBackend, FlmConfig, LlamaCppBackend, LlamaCppConfig, LlamaRuntime, ModelManager, WhisperBackend,
    WhisperConfig,
};
use contio_core::hw::GpuApi;
use contio_install::jobs::{JobCtx, JobInfo, Jobs};
use contio_install::{
    BackendKind, FLM_DEFAULT_VERSION, InstalledBackend, LLAMACPP_DEFAULT_VERSION, LLAMACPP_VULKAN_FA_PREVIEW, Method,
    WHISPER_DEFAULT_VERSION, discover, flm_preflight, install_dir, install_flm, install_llamacpp, install_whisper,
    llamacpp_preflight, whisper_preflight,
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
        FlmConfig { prefix, model_path: self.models_dir().join("flm"), xrt_dir: XRT_DIR.into(), embed: true, asr: false }
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
            Some(Arc::new(LlamaCppBackend::new(LlamaCppConfig {
                runtime,
                gpu_api: b.gpu_api,
                cpu_threads: Some(installer.cpu_threads),
            })))
        }
        BackendKind::Flm => {
            let config = installer.flm_config(b.prefix.clone()?);
            installer.set_flm(config.clone());
            Some(Arc::new(FlmBackend::new(config)))
        }
        BackendKind::Whisper => Some(Arc::new(WhisperBackend::new(WhisperConfig {
            bin_dir: b.bin_dir.clone()?,
            cpu_threads: Some(installer.cpu_threads),
        }))),
    }
}

/// Install a backend and register it with the manager right away.
pub async fn install_and_register(
    inst: &Installer,
    manager: &ModelManager,
    backend: BackendKind,
    method: Method,
    version: &str,
    gpu_api: GpuApi,
    ctx: &JobCtx,
) -> Result<InstalledBackend, String> {
    let installed = match backend {
        BackendKind::LlamaCpp => install_llamacpp(&inst.http, &inst.data_dir, method, version, gpu_api, ctx).await?,
        BackendKind::Flm => install_flm(&inst.data_dir, method, version, ctx).await?,
        BackendKind::Whisper => install_whisper(&inst.data_dir, method, version, ctx).await?,
    };
    // the experimental build only runs while switched on (`use_vulkan_preview`)
    if !installed.is_preview()
        && let Some(b) = backend_from_installed(&installed, inst)
    {
        manager.add_backend(b);
    }
    Ok(installed)
}

fn installed_preview(inst: &Installer) -> Option<InstalledBackend> {
    discover(&inst.data_dir).into_iter().find(InstalledBackend::is_preview)
}

/// Switch the experimental Vulkan build (Settings → Experimental) on or off:
/// it goes in front of the regular Vulkan build, or out again. Loaded Vulkan
/// models restart on their next request.
pub async fn use_vulkan_preview(inst: &Installer, manager: &ModelManager, on: bool) -> Result<(), String> {
    let dir = install_dir(
        &inst.data_dir,
        BackendKind::LlamaCpp,
        LLAMACPP_VULKAN_FA_PREVIEW.version,
        Method::Source,
        GpuApi::Vulkan,
    )
    .display()
    .to_string();
    manager.remove_backends(|b| b.describe().contains(&dir));
    if on {
        let b = installed_preview(inst)
            .and_then(|i| backend_from_installed(&i, inst))
            .ok_or("the experimental Vulkan build isn't installed")?;
        manager.add_backend(b);
    }
    manager.restart_gpu_models(GpuApi::Vulkan).await;
    Ok(())
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
    #[serde(default)]
    pub gpu_api: GpuApi,
}

pub async fn list_backends(State(state): State<AppState>) -> Response {
    let installed = discover(&state.installer.data_dir);
    Json(json!({
        "registered": state.manager.backend_names(),
        "installed": installed,
        "defaults": {
            "llama.cpp": LLAMACPP_DEFAULT_VERSION, "flm": FLM_DEFAULT_VERSION, "whisper.cpp": WHISPER_DEFAULT_VERSION,
        },
        "settings": state.registry.settings(),
        "preview": {
            "version": LLAMACPP_VULKAN_FA_PREVIEW.version,
            "url": LLAMACPP_VULKAN_FA_PREVIEW.url,
            "installed": installed.iter().any(InstalledBackend::is_preview),
        },
    }))
    .into_response()
}

pub async fn preflight(Query(q): Query<BackendQuery>) -> Response {
    let pf = tokio::task::spawn_blocking(move || match q.backend {
        BackendKind::LlamaCpp => llamacpp_preflight(q.method, q.gpu_api),
        BackendKind::Flm => flm_preflight(q.method),
        BackendKind::Whisper => whisper_preflight(q.method),
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
    /// llama.cpp: build for Vulkan (default) or ROCm.
    #[serde(default, skip_serializing_if = "is_vulkan")]
    pub gpu_api: GpuApi,
}

fn is_vulkan(api: &GpuApi) -> bool {
    *api == GpuApi::Vulkan
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
        BackendKind::Whisper => WHISPER_DEFAULT_VERSION.into(),
    });
    if !version.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c)) || version.starts_with('.') {
        return Err("invalid version".into());
    }
    let inst = state.installer.clone();
    let manager = state.manager.clone();
    let rocm = req.backend == BackendKind::LlamaCpp && req.gpu_api == GpuApi::Rocm;
    let flavor = if rocm {
        " rocm"
    } else if version == LLAMACPP_VULKAN_FA_PREVIEW.version {
        " experimental"
    } else {
        ""
    };
    let title = format!("{}{flavor} {version} ({:?})", req.backend.dir_name(), req.method).to_lowercase();
    let key = Some(format!("install:{}{}", req.backend.dir_name(), flavor.replace(' ', ":")));
    let spec = serde_json::to_value(JobSpec::InstallBackend(req.clone())).ok();
    Ok(inst.jobs.clone().spawn_with_spec("install", &title, key, spec, move |ctx| async move {
        let installed =
            install_and_register(&inst, &manager, req.backend, req.method, &version, req.gpu_api, &ctx).await?;
        serde_json::to_value(&installed).map_err(|e| e.to_string())
    }))
}

// --- settings -----------------------------------------------------------------

pub async fn get_settings(State(state): State<AppState>) -> Response {
    Json(state.registry.settings()).into_response()
}

pub async fn put_settings(State(state): State<AppState>, Json(settings): Json<Settings>) -> Response {
    if settings.flm_method != Method::Source || settings.whisper_method != Method::Source {
        return bad_request("FastFlowLM and whisper.cpp can only be installed from source");
    }
    if settings.gpu_api != state.manager.default_gpu_api() && !state.manager.has_gpu_build(settings.gpu_api) {
        return bad_request(&format!("install the llama.cpp {} build first", settings.gpu_api));
    }
    let preview_changed = settings.vulkan_fa_preview != state.registry.settings().vulkan_fa_preview;
    if preview_changed && settings.vulkan_fa_preview && installed_preview(&state.installer).is_none() {
        return bad_request("build the experimental Vulkan build first");
    }
    match state.registry.set_settings(settings.clone()) {
        Ok(()) => {
            state.manager.set_default_gpu_api(settings.gpu_api).await;
            if preview_changed
                && let Err(e) = use_vulkan_preview(&state.installer, &state.manager, settings.vulkan_fa_preview).await
            {
                tracing::warn!("{e}");
            }
            Json(settings).into_response()
        }
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
            gpu_api: GpuApi::Rocm,
            installed_at: 1,
        };
        let llama = backend_from_installed(&b, &inst).unwrap();
        assert_eq!((llama.name(), llama.gpu_api()), ("llama.cpp", Some(GpuApi::Rocm)));
        b.image = None;
        assert!(backend_from_installed(&b, &inst).is_none(), "marker without runtime");
        assert!(inst.flm().is_none());
        b.backend = BackendKind::Flm;
        b.prefix = Some("/p".into());
        assert_eq!(backend_from_installed(&b, &inst).unwrap().name(), "FastFlowLM");
        let flm = inst.flm().expect("FLM config remembered for pulls");
        assert_eq!(flm.model_path, PathBuf::from("/data/models/flm"));
    }

    #[tokio::test]
    async fn vulkan_preview_switches_in_front_and_out() {
        let data = tempfile::tempdir().unwrap();
        let inst = Installer::in_memory(data.path().into(), 4);
        let manager = ModelManager::new(
            contio_backends::ManagerConfig {
                log_dir: std::env::temp_dir(),
                startup_timeout: std::time::Duration::from_secs(1),
                default_idle_ttl: None,
                pool_budgets: Default::default(),
            },
            vec![],
            vec![],
        );
        let regular = |bin: &str| {
            Arc::new(LlamaCppBackend::new(LlamaCppConfig {
                runtime: LlamaRuntime::Native { bin_dir: bin.into() },
                gpu_api: GpuApi::Vulkan,
                cpu_threads: None,
            }))
        };
        manager.add_backend(regular("/regular/bin"));
        assert!(use_vulkan_preview(&inst, &manager, true).await.is_err(), "not installed");

        let v = LLAMACPP_VULKAN_FA_PREVIEW.version;
        let dir = install_dir(data.path(), BackendKind::LlamaCpp, v, Method::Source, GpuApi::Vulkan);
        std::fs::create_dir_all(&dir).unwrap();
        let marker = json!({
            "backend": "llama.cpp", "method": "source", "version": v, "dir": dir,
            "bin_dir": dir.join("build/bin"), "gpu_api": "vulkan", "installed_at": 1,
        });
        std::fs::write(dir.join("contio-backend.json"), marker.to_string()).unwrap();
        assert!(installed_preview(&inst).is_some());

        use_vulkan_preview(&inst, &manager, true).await.unwrap();
        let names = manager.backend_names();
        assert_eq!(names.len(), 2);
        assert!(names[0].contains(v), "experimental build first: {names:?}");
        use_vulkan_preview(&inst, &manager, true).await.unwrap();
        assert_eq!(manager.backend_names().len(), 2, "switching on twice registers it once");
        use_vulkan_preview(&inst, &manager, false).await.unwrap();
        assert_eq!(manager.backend_names(), vec!["llama.cpp Vulkan (native: /regular/bin)".to_owned()]);
    }

    #[test]
    fn job_specs_round_trip() {
        let spec = JobSpec::InstallBackend(InstallRequest {
            backend: BackendKind::LlamaCpp,
            method: Method::Prebuilt,
            version: None,
            gpu_api: GpuApi::Vulkan,
        });
        let v = serde_json::to_value(&spec).unwrap();
        assert_eq!(v["type"], "install_backend");
        assert_eq!(v["backend"], "llama.cpp");
        assert!(v.get("gpu_api").is_none(), "Vulkan stays implicit in stored job specs");
        assert!(matches!(serde_json::from_value::<JobSpec>(v).unwrap(), JobSpec::InstallBackend(_)));
    }
}
