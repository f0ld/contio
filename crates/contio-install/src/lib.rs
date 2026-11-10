//! Background jobs: model downloads and backend installers (spec §6).
//!
//! Installed backends live in `<data_dir>/backends/<backend>/<version>-<method>/`
//! with a `contio-backend.json` marker, so the daemon can find and register them
//! without config edits.

pub mod download;
mod flm;
pub mod jobs;
mod llamacpp;
pub mod preflight;
mod whisper;

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use contio_core::hw::GpuApi;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

pub use flm::{FLM_DEFAULT_VERSION, flm_preflight, install_flm};
pub use llamacpp::{
    LLAMACPP_DEFAULT_VERSION, LLAMACPP_VULKAN_FA_PREVIEW, LlamaCppPreview, install_llamacpp, llamacpp_preflight,
};
pub use whisper::{WHISPER_DEFAULT_VERSION, install_whisper, whisper_preflight};

use crate::jobs::JobCtx;

const MARKER: &str = "contio-backend.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendKind {
    #[serde(rename = "llama.cpp")]
    LlamaCpp,
    #[serde(rename = "flm")]
    Flm,
    #[serde(rename = "whisper.cpp")]
    Whisper,
}

impl BackendKind {
    pub fn dir_name(self) -> &'static str {
        match self {
            Self::LlamaCpp => "llama.cpp",
            Self::Flm => "flm",
            Self::Whisper => "whisper.cpp",
        }
    }
}

/// How a backend gets onto the machine; chosen by the user (spec §5.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Method {
    Source,
    Prebuilt,
    Podman,
}

/// Contents of the marker file; also the result of an install job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledBackend {
    pub backend: BackendKind,
    pub method: Method,
    pub version: String,
    pub dir: PathBuf,
    /// Native llama.cpp / whisper.cpp: directory with the server binary.
    #[serde(default)]
    pub bin_dir: Option<PathBuf>,
    /// Podman llama.cpp: image reference.
    #[serde(default)]
    pub image: Option<String>,
    /// FLM: install prefix (bin/flm, lib/flm, share/flm).
    #[serde(default)]
    pub prefix: Option<PathBuf>,
    /// llama.cpp: what the build uses for GPU models (installs before ROCm support: Vulkan).
    #[serde(default)]
    pub gpu_api: GpuApi,
    pub installed_at: u64,
}

impl InstalledBackend {
    /// The experimental llama.cpp build ([`LLAMACPP_VULKAN_FA_PREVIEW`]): never used
    /// unless switched on in the settings.
    pub fn is_preview(&self) -> bool {
        self.backend == BackendKind::LlamaCpp && self.version == LLAMACPP_VULKAN_FA_PREVIEW.version
    }

    fn write_marker(&self) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(self.dir.join(MARKER), json).map_err(|e| format!("writing marker: {e}"))
    }
}

/// Where a given install goes; non-default GPU builds get a suffix (`b12068-source-rocm`).
pub fn install_dir(data_dir: &Path, backend: BackendKind, version: &str, method: Method, gpu_api: GpuApi) -> PathBuf {
    let method = serde_json::to_value(method).ok().and_then(|v| v.as_str().map(str::to_owned)).unwrap_or_default();
    let suffix = match gpu_api {
        GpuApi::Vulkan => "",
        GpuApi::Rocm => "-rocm",
    };
    data_dir.join("backends").join(backend.dir_name()).join(format!("{version}-{method}{suffix}"))
}

/// All installed backends, newest first.
pub fn discover(data_dir: &Path) -> Vec<InstalledBackend> {
    let mut found = Vec::new();
    for kind in [BackendKind::LlamaCpp, BackendKind::Flm, BackendKind::Whisper] {
        let Ok(entries) = std::fs::read_dir(data_dir.join("backends").join(kind.dir_name())) else { continue };
        for e in entries.flatten() {
            let Ok(text) = std::fs::read_to_string(e.path().join(MARKER)) else { continue };
            match serde_json::from_str::<InstalledBackend>(&text) {
                Ok(b) => found.push(b),
                Err(err) => tracing::warn!(path = %e.path().display(), "ignoring bad backend marker: {err}"),
            }
        }
    }
    found.sort_by(|a, b| b.installed_at.cmp(&a.installed_at));
    found
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or_default()
}

/// Run a command, streaming its output into the job log; killed on cancel.
pub async fn run_logged(ctx: &JobCtx, cmd: &mut Command) -> Result<(), String> {
    let program = cmd.as_std().get_program().to_string_lossy().into_owned();
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("starting {program}: {e}"))?;
    let mut out = BufReader::new(child.stdout.take().expect("piped")).lines();
    let mut err = BufReader::new(child.stderr.take().expect("piped")).lines();
    let (mut out_open, mut err_open) = (true, true);
    let mut tick = tokio::time::interval(Duration::from_millis(250));
    while out_open || err_open {
        tokio::select! {
            line = out.next_line(), if out_open => match line {
                Ok(Some(l)) => ctx.log(&l),
                _ => out_open = false,
            },
            line = err.next_line(), if err_open => match line {
                Ok(Some(l)) => ctx.log(&l),
                _ => err_open = false,
            },
            _ = tick.tick() => if ctx.is_cancelled() {
                let _ = child.kill().await;
                return Err("cancelled".into());
            },
        }
    }
    let status = child.wait().await.map_err(|e| e.to_string())?;
    if status.success() { Ok(()) } else { Err(format!("{program} failed ({status}); see the job log")) }
}

/// Shallow-fetch `rev` (tag or commit) of `repo` into `dir` and check it out.
pub(crate) async fn git_checkout(ctx: &JobCtx, repo: &str, rev: &str, dir: &Path) -> Result<(), String> {
    tokio::fs::create_dir_all(dir).await.map_err(|e| e.to_string())?;
    let git = |args: &[&str]| {
        let mut c = Command::new("git");
        c.arg("-C").arg(dir).args(args);
        c
    };
    if !dir.join(".git").exists() {
        run_logged(ctx, &mut git(&["init", "-q"])).await?;
        run_logged(ctx, &mut git(&["remote", "add", "origin", repo])).await?;
    }
    ctx.step(&format!("fetching {rev}"));
    run_logged(ctx, &mut git(&["fetch", "--depth", "1", "origin", rev])).await?;
    run_logged(ctx, &mut git(&["checkout", "-q", "--force", "FETCH_HEAD"])).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_dirs_and_discovery() {
        let data = tempfile::tempdir().unwrap();
        let dir = install_dir(data.path(), BackendKind::LlamaCpp, "b12068", Method::Prebuilt, GpuApi::Vulkan);
        assert!(dir.ends_with("backends/llama.cpp/b12068-prebuilt"));
        let rocm = install_dir(data.path(), BackendKind::LlamaCpp, "b12068", Method::Source, GpuApi::Rocm);
        assert!(rocm.ends_with("backends/llama.cpp/b12068-source-rocm"));

        let mut a = InstalledBackend {
            backend: BackendKind::LlamaCpp,
            method: Method::Prebuilt,
            version: "b12068".into(),
            dir: dir.clone(),
            bin_dir: Some(dir.join("bin")),
            image: None,
            prefix: None,
            gpu_api: GpuApi::Vulkan,
            installed_at: 10,
        };
        std::fs::create_dir_all(&dir).unwrap();
        a.write_marker().unwrap();

        let flm_dir = install_dir(data.path(), BackendKind::Flm, "d4a4553", Method::Source, GpuApi::Vulkan);
        std::fs::create_dir_all(&flm_dir).unwrap();
        let b = InstalledBackend {
            backend: BackendKind::Flm,
            method: Method::Source,
            version: "d4a4553".into(),
            dir: flm_dir.clone(),
            bin_dir: None,
            image: None,
            prefix: Some(flm_dir.join("prefix")),
            gpu_api: GpuApi::Vulkan,
            installed_at: 20,
        };
        b.write_marker().unwrap();
        std::fs::write(flm_dir.parent().unwrap().join("junk.json"), "{}").unwrap(); // ignored

        let found = discover(data.path());
        assert_eq!(found, vec![b, a.clone()], "newest first");
        a.installed_at = 0;
        assert_eq!(serde_json::to_value(&a).unwrap()["backend"], "llama.cpp");
        let mut old = serde_json::to_value(&a).unwrap();
        old.as_object_mut().unwrap().remove("gpu_api");
        assert_eq!(serde_json::from_value::<InstalledBackend>(old).unwrap().gpu_api, GpuApi::Vulkan, "older markers");
    }

    #[tokio::test]
    async fn run_logged_captures_output_and_status() {
        let jobs = jobs::Jobs::default();
        let id = jobs.spawn("t", "t", None, |ctx| async move {
            run_logged(&ctx, Command::new("sh").args(["-c", "echo out; echo err >&2"])).await?;
            run_logged(&ctx, Command::new("sh").args(["-c", "exit 3"])).await.map(|_| serde_json::Value::Null)
        });
        let job = loop {
            let j = jobs.get(id).unwrap();
            if j.state != jobs::JobState::Running {
                break j;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        };
        assert!(job.log.contains(&"out".to_owned()) && job.log.contains(&"err".to_owned()));
        assert!(job.error.unwrap().contains("failed"));
    }
}
