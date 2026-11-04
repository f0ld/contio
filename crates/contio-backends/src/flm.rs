//! FastFlowLM `flm serve` (NPU, XDNA2).
//!
//! The environment is built here instead of sourcing FLM's `flm_env.sh`, which
//! at the pinned commit hardcodes a developer's `FLM_MODEL_PATH`
//! (`docs/spikes/m0-npu.md` §6.6).

use std::path::PathBuf;

use contio_core::hw::DeviceKind;
use contio_core::model::{ModelProfile, ModelSource};
use tokio::process::Command;

use crate::{Backend, BackendError};

#[derive(Debug, Clone)]
pub struct FlmConfig {
    /// Install prefix (contains `bin/flm`, `lib/flm`, `share/flm`).
    pub prefix: PathBuf,
    /// Where FLM keeps downloaded models (`FLM_MODEL_PATH`).
    pub model_path: PathBuf,
    /// XRT install (`/opt/xilinx/xrt` on Fedora RPMs).
    pub xrt_dir: PathBuf,
    /// Co-load the embedding model next to every LLM (`--embed 1`, m0-gpu.md §4).
    pub embed: bool,
    /// Co-load Whisper (`--asr 1`): FLM serves speech only next to an LLM.
    pub asr: bool,
}

#[derive(Debug, Clone)]
pub struct FlmBackend {
    config: FlmConfig,
}

impl FlmBackend {
    pub fn new(config: FlmConfig) -> Self {
        Self { config }
    }

    fn tag(profile: &ModelProfile) -> Option<&str> {
        match &profile.source {
            ModelSource::Flm { tag } => Some(tag),
            ModelSource::Gguf { .. } | ModelSource::Whisper { .. } => None,
        }
    }

    fn args(&self, profile: &ModelProfile, port: u16) -> Vec<String> {
        let mut a = vec![
            "serve".into(),
            Self::tag(profile).expect("supports() checked the format").to_owned(),
            "--port".into(),
            port.to_string(),
        ];
        if let Some(ctx) = profile.context {
            a.extend(["--ctx-len".into(), ctx.to_string()]);
        }
        if self.config.embed {
            a.extend(["--embed".into(), "1".into()]);
        }
        if self.config.asr {
            a.extend(["--asr".into(), "1".into()]);
        }
        a.extend(profile.extra_args.iter().cloned());
        a
    }

    /// `flm` with contio's environment, for other subcommands (e.g. `pull`).
    pub fn flm_command(&self) -> Result<Command, BackendError> {
        let bin = self.config.prefix.join("bin/flm");
        if !bin.is_file() {
            return Err(BackendError::MissingBinary(bin));
        }
        let mut cmd = Command::new(bin);
        cmd.envs(self.env());
        Ok(cmd)
    }

    fn env(&self) -> Vec<(&'static str, String)> {
        let c = &self.config;
        let share = c.prefix.join("share/flm");
        let ld = [c.prefix.join("lib/flm"), c.xrt_dir.join("lib")]
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(":");
        vec![
            ("FLM_CONFIG_PATH", share.join("model_list.json").display().to_string()),
            ("FLM_XCLBIN_PATH", share.display().to_string()),
            ("FLM_MODEL_PATH", c.model_path.display().to_string()),
            ("XILINX_XRT", c.xrt_dir.display().to_string()),
            ("LD_LIBRARY_PATH", ld),
        ]
    }
}

impl Backend for FlmBackend {
    fn name(&self) -> &'static str {
        "FastFlowLM"
    }

    fn describe(&self) -> String {
        format!("FastFlowLM ({})", self.config.prefix.display())
    }

    fn supports(&self, profile: &ModelProfile) -> bool {
        Self::tag(profile).is_some() && profile.device == DeviceKind::Npu
    }

    fn command(&self, profile: &ModelProfile, port: u16) -> Result<Command, BackendError> {
        let mut cmd = self.flm_command()?;
        cmd.args(self.args(profile, port));
        Ok(cmd)
    }

    fn health_path(&self) -> &'static str {
        "/v1/models"
    }

    fn upstream_model(&self, profile: &ModelProfile) -> String {
        Self::tag(profile).unwrap_or(&profile.id).to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend(embed: bool) -> FlmBackend {
        FlmBackend::new(FlmConfig {
            prefix: "/opt/flm".into(),
            model_path: "/data/flm".into(),
            xrt_dir: "/opt/xilinx/xrt".into(),
            embed,
            asr: false,
        })
    }

    fn profile() -> ModelProfile {
        let mut p = ModelProfile::new("qwen3-4b-npu", ModelSource::Flm { tag: "qwen3-it:4b".into() }, DeviceKind::Npu);
        p.context = Some(8192);
        p
    }

    #[test]
    fn args_and_upstream_name() {
        let b = backend(true);
        assert_eq!(
            b.args(&profile(), 5000).join(" "),
            "serve qwen3-it:4b --port 5000 --ctx-len 8192 --embed 1"
        );
        assert_eq!(b.upstream_model(&profile()), "qwen3-it:4b");
        assert!(!backend(false).args(&profile(), 1).contains(&"--embed".to_owned()));
    }

    #[test]
    fn env_is_explicit() {
        let env = backend(false).env();
        let get = |k| env.iter().find(|(n, _)| *n == k).map(|(_, v)| v.as_str()).unwrap();
        assert_eq!(get("FLM_MODEL_PATH"), "/data/flm");
        assert_eq!(get("FLM_CONFIG_PATH"), "/opt/flm/share/flm/model_list.json");
        assert_eq!(get("LD_LIBRARY_PATH"), "/opt/flm/lib/flm:/opt/xilinx/xrt/lib");
    }

    #[test]
    fn supports_only_flm_on_npu() {
        let b = backend(false);
        assert!(b.supports(&profile()));
        let mut gpu = profile();
        gpu.device = DeviceKind::Gpu;
        assert!(!b.supports(&gpu));
    }
}
