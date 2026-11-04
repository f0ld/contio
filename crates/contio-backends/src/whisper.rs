//! whisper.cpp `whisper-server` (speech to text, GPU via Vulkan or CPU).
//!
//! The server takes OpenAI-style multipart uploads on its inference path, so
//! it is mounted at `/v1/audio/transcriptions`. It loads one model and
//! ignores the request's `model` field.

use std::path::PathBuf;

use contio_core::hw::DeviceKind;
use contio_core::model::{ModelProfile, ModelSource, estimate_whisper_memory};
use tokio::process::Command;

use crate::{Backend, BackendError};

/// The upstream path transcriptions (and translations, with `translate=true`) go to.
pub const WHISPER_INFERENCE_PATH: &str = "/v1/audio/transcriptions";

#[derive(Debug, Clone)]
pub struct WhisperConfig {
    /// Directory containing `whisper-server`.
    pub bin_dir: PathBuf,
    pub cpu_threads: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct WhisperBackend {
    config: WhisperConfig,
}

impl WhisperBackend {
    pub fn new(config: WhisperConfig) -> Self {
        Self { config }
    }

    fn model_path(profile: &ModelProfile) -> Option<&PathBuf> {
        match &profile.source {
            ModelSource::Whisper { path } => Some(path),
            ModelSource::Gguf { .. } | ModelSource::Flm { .. } => None,
        }
    }

    fn args(&self, profile: &ModelProfile, port: u16, model: &str) -> Vec<String> {
        let mut a: Vec<String> = vec![
            "--model".into(),
            model.to_owned(),
            "--host".into(),
            "127.0.0.1".into(),
            "--port".into(),
            port.to_string(),
            "--inference-path".into(),
            WHISPER_INFERENCE_PATH.into(),
            // OpenAI detects the language when the request doesn't name one
            "--language".into(),
            "auto".into(),
        ];
        if profile.device == DeviceKind::Cpu {
            a.push("--no-gpu".into());
        }
        if let Some(t) = self.config.cpu_threads {
            a.extend(["--threads".into(), t.to_string()]);
        }
        a.extend(profile.extra_args.iter().cloned());
        a
    }
}

impl Backend for WhisperBackend {
    fn name(&self) -> &'static str {
        "whisper.cpp"
    }

    fn describe(&self) -> String {
        format!("whisper.cpp (native: {})", self.config.bin_dir.display())
    }

    fn supports(&self, profile: &ModelProfile) -> bool {
        Self::model_path(profile).is_some() && matches!(profile.device, DeviceKind::Gpu | DeviceKind::Cpu)
    }

    fn command(&self, profile: &ModelProfile, port: u16) -> Result<Command, BackendError> {
        let path = Self::model_path(profile).ok_or_else(|| BackendError::Unsupported(profile.id.clone()))?;
        if !path.is_file() {
            return Err(BackendError::MissingModelFile(path.clone()));
        }
        let bin = self.config.bin_dir.join("whisper-server");
        if !bin.is_file() {
            return Err(BackendError::MissingBinary(bin));
        }
        let mut cmd = Command::new(bin);
        cmd.args(self.args(profile, port, &path.display().to_string()));
        Ok(cmd)
    }

    fn health_path(&self) -> &'static str {
        "/health"
    }

    fn upstream_model(&self, profile: &ModelProfile) -> String {
        profile.id.clone()
    }

    fn estimate_bytes(&self, profile: &ModelProfile) -> Option<u64> {
        Some(estimate_whisper_memory(std::fs::metadata(Self::model_path(profile)?).ok()?.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend() -> WhisperBackend {
        WhisperBackend::new(WhisperConfig { bin_dir: "/nonexistent".into(), cpu_threads: Some(8) })
    }

    #[test]
    fn args_and_support() {
        let mut p = ModelProfile::new("whisper", ModelSource::Whisper { path: "/m/ggml-base.bin".into() }, DeviceKind::Gpu);
        let b = backend();
        assert!(b.supports(&p));
        assert_eq!(
            b.args(&p, 4000, "/m/ggml-base.bin").join(" "),
            "--model /m/ggml-base.bin --host 127.0.0.1 --port 4000 --inference-path /v1/audio/transcriptions \
             --language auto --threads 8"
        );
        p.device = DeviceKind::Cpu;
        assert!(b.args(&p, 4000, "m").contains(&"--no-gpu".to_owned()));
        p.device = DeviceKind::Npu;
        assert!(!b.supports(&p));
        let gguf = ModelProfile::new("x", ModelSource::Gguf { path: "/m/x.gguf".into(), mmproj: None }, DeviceKind::Gpu);
        assert!(!b.supports(&gguf));
    }

    #[test]
    fn missing_files_are_reported() {
        let p = ModelProfile::new("w", ModelSource::Whisper { path: "/nonexistent/ggml.bin".into() }, DeviceKind::Gpu);
        assert!(matches!(backend().command(&p, 1), Err(BackendError::MissingModelFile(_))));
    }
}
