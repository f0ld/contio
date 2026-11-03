//! llama.cpp `llama-server` (GGUF on GPU via Vulkan, or CPU).

use std::path::PathBuf;

use contio_core::hw::DeviceKind;
use contio_core::model::{ModelProfile, ModelSource};
use tokio::process::Command;

use crate::{Backend, BackendError, estimate_from_weights};

#[derive(Debug, Clone)]
pub struct LlamaCppConfig {
    /// Directory containing `llama-server` (a Vulkan build also serves CPU).
    pub bin_dir: PathBuf,
    /// Threads for CPU inference (physical cores work best, see m0-gpu.md).
    pub cpu_threads: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct LlamaCppBackend {
    config: LlamaCppConfig,
}

impl LlamaCppBackend {
    pub fn new(config: LlamaCppConfig) -> Self {
        Self { config }
    }

    fn gguf_path(profile: &ModelProfile) -> Option<&PathBuf> {
        match &profile.source {
            ModelSource::Gguf { path } => Some(path),
            ModelSource::Flm { .. } => None,
        }
    }

    /// Arguments after the binary; separated out for testing.
    fn args(&self, profile: &ModelProfile, port: u16) -> Vec<String> {
        let path = Self::gguf_path(profile).expect("supports() checked the format");
        let mut a: Vec<String> = vec![
            "--model".into(),
            path.display().to_string(),
            "--host".into(),
            "127.0.0.1".into(),
            "--port".into(),
            port.to_string(),
            "--alias".into(),
            profile.id.clone(),
            // chat templates from the GGUF: needed for tool calling (m0-gpu.md)
            "--jinja".into(),
            "--flash-attn".into(),
            "on".into(),
        ];
        if let Some(ctx) = profile.context {
            a.extend(["--ctx-size".into(), ctx.to_string()]);
        }
        match profile.device {
            DeviceKind::Gpu => a.extend(["--n-gpu-layers".into(), "999".into()]),
            _ => {
                a.extend(["--n-gpu-layers".into(), "0".into(), "--device".into(), "none".into()]);
                if let Some(t) = self.config.cpu_threads {
                    a.extend(["--threads".into(), t.to_string()]);
                }
            }
        }
        a.extend(profile.extra_args.iter().cloned());
        a
    }
}

impl Backend for LlamaCppBackend {
    fn name(&self) -> &'static str {
        "llama.cpp"
    }

    fn supports(&self, profile: &ModelProfile) -> bool {
        Self::gguf_path(profile).is_some() && matches!(profile.device, DeviceKind::Gpu | DeviceKind::Cpu)
    }

    fn command(&self, profile: &ModelProfile, port: u16) -> Result<Command, BackendError> {
        let bin = self.config.bin_dir.join("llama-server");
        if !bin.is_file() {
            return Err(BackendError::MissingBinary(bin));
        }
        if let Some(path) = Self::gguf_path(profile)
            && !path.is_file()
        {
            return Err(BackendError::MissingModelFile(path.clone()));
        }
        let mut cmd = Command::new(bin);
        cmd.args(self.args(profile, port));
        Ok(cmd)
    }

    fn health_path(&self) -> &'static str {
        "/health"
    }

    fn upstream_model(&self, profile: &ModelProfile) -> String {
        profile.id.clone() // served under --alias
    }

    fn estimate_bytes(&self, profile: &ModelProfile) -> Option<u64> {
        let weights = std::fs::metadata(Self::gguf_path(profile)?).ok()?.len();
        Some(estimate_from_weights(weights, profile.context))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(device: DeviceKind) -> ModelProfile {
        ModelProfile {
            id: "qwen3-8b".into(),
            source: ModelSource::Gguf { path: "/m/q.gguf".into() },
            device,
            context: Some(16384),
            pinned: false,
            idle_ttl_secs: None,
            extra_args: vec!["--cache-type-k".into(), "q8_0".into()],
        }
    }

    fn backend() -> LlamaCppBackend {
        LlamaCppBackend::new(LlamaCppConfig { bin_dir: "/nonexistent".into(), cpu_threads: Some(12) })
    }

    #[test]
    fn gpu_args() {
        let a = backend().args(&profile(DeviceKind::Gpu), 40001).join(" ");
        assert!(a.starts_with("--model /m/q.gguf --host 127.0.0.1 --port 40001 --alias qwen3-8b --jinja"));
        assert!(a.contains("--ctx-size 16384"));
        assert!(a.contains("--n-gpu-layers 999"));
        assert!(!a.contains("--threads"));
        assert!(a.ends_with("--cache-type-k q8_0"));
    }

    #[test]
    fn cpu_args_disable_gpu_and_set_threads() {
        let a = backend().args(&profile(DeviceKind::Cpu), 1).join(" ");
        assert!(a.contains("--n-gpu-layers 0 --device none --threads 12"));
    }

    #[test]
    fn supports_only_gguf_on_cpu_gpu() {
        let b = backend();
        assert!(b.supports(&profile(DeviceKind::Gpu)));
        assert!(!b.supports(&profile(DeviceKind::Npu)));
        let mut flm = profile(DeviceKind::Gpu);
        flm.source = ModelSource::Flm { tag: "x".into() };
        assert!(!b.supports(&flm));
    }

    #[test]
    fn missing_binary_is_reported() {
        let err = backend().command(&profile(DeviceKind::Gpu), 1).unwrap_err();
        assert!(matches!(err, BackendError::MissingBinary(_)));
    }
}
