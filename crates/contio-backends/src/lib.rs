//! Inference backends and their lifecycle (spec §5, §7.2).
//!
//! contio does not run inference itself: each loaded model is a child process
//! (llama-server, `flm serve`, …) with its own OpenAI-compatible HTTP server on
//! an ephemeral localhost port. This crate knows how to launch those processes
//! ([`Backend`]), supervise them ([`Instance`]) and decide which ones are
//! running ([`ModelManager`]).

mod flm;
mod instance;
mod llamacpp;
mod manager;
mod whisper;

use std::path::PathBuf;

use contio_core::hw::GpuApi;
use contio_core::model::ModelProfile;
use tokio::process::Command;

pub use flm::{FlmBackend, FlmConfig};
pub use instance::{Instance, stop_orphans};
pub use llamacpp::{LlamaCppBackend, LlamaCppConfig, LlamaRuntime};
pub use manager::{Lease, ManagerConfig, ModelManager};
pub use whisper::{WHISPER_INFERENCE_PATH, WhisperBackend, WhisperConfig};

/// Knows how to launch one kind of inference server.
pub trait Backend: Send + Sync {
    /// Human-readable name, e.g. `llama.cpp`.
    fn name(&self) -> &'static str;

    /// Name plus how it runs, e.g. `llama.cpp (podman: <image>)`.
    fn describe(&self) -> String {
        self.name().to_owned()
    }

    /// The GPU API this backend is built for, if it matters (llama.cpp).
    fn gpu_api(&self) -> Option<GpuApi> {
        None
    }

    /// Whether this backend can run the profile (format + device). The
    /// manager fills in the default GPU API before asking.
    fn supports(&self, profile: &ModelProfile) -> bool;

    /// The command that starts a server for `profile` on `127.0.0.1:port`.
    /// Must not inherit configuration from the daemon's environment that would
    /// change behaviour (see `docs/spikes/m0-npu.md` §6.6).
    fn command(&self, profile: &ModelProfile, port: u16) -> Result<Command, BackendError>;

    /// Path that returns 2xx once the server can take requests.
    fn health_path(&self) -> &'static str;

    /// The value to put in the upstream request's `model` field.
    fn upstream_model(&self, profile: &ModelProfile) -> String;

    /// Resident memory estimate used for pool budgeting, if known.
    fn estimate_bytes(&self, _profile: &ModelProfile) -> Option<u64> {
        None
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("unknown model `{0}`")]
    UnknownModel(String),
    #[error("a model named `{0}` already exists")]
    DuplicateModel(String),
    #[error("`{0}` is defined in config.toml; edit it there")]
    NotManaged(String),
    #[error("no backend can run model `{0}` on this device")]
    Unsupported(String),
    #[error("`{model}` runs on {api}, but no llama.cpp {api} build is installed (install one under Backends, or pick another GPU backend)")]
    MissingGpuBuild { model: String, api: GpuApi },
    #[error("backend binary not found: {0}")]
    MissingBinary(PathBuf),
    #[error("model file not found: {0}")]
    MissingModelFile(PathBuf),
    #[error(
        "not enough memory in the {pool} pool for `{model}`: needs ~{needed_mib} MiB, \
         {available_mib} MiB available after evicting all unpinned models"
    )]
    InsufficientMemory { model: String, pool: String, needed_mib: u64, available_mib: u64 },
    #[error("`{model}` failed to start: {reason}")]
    StartFailed { model: String, reason: String },
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
