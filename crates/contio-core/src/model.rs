//! Installed model profiles and their runtime state (spec §4, §7.2).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::hw::{DeviceKind, MemoryPoolKind};

/// Where a model's weights come from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "format", rename_all = "lowercase")]
pub enum ModelSource {
    /// A local GGUF file (llama.cpp).
    Gguf { path: PathBuf },
    /// A FastFlowLM model tag, e.g. `qwen3:8b` (NPU).
    Flm { tag: String },
}

/// One runnable configuration of a model: weights + device + launch options.
/// The `id` is the name clients put in the OpenAI `model` field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelProfile {
    pub id: String,
    #[serde(flatten)]
    pub source: ModelSource,
    pub device: DeviceKind,
    /// Context length in tokens; backend default if unset.
    #[serde(default)]
    pub context: Option<u32>,
    /// Pinned models are never evicted or unloaded for being idle.
    #[serde(default)]
    pub pinned: bool,
    /// Unload after this many idle seconds; the manager default if unset.
    #[serde(default)]
    pub idle_ttl_secs: Option<u64>,
    /// Extra command-line arguments passed to the backend verbatim.
    #[serde(default)]
    pub extra_args: Vec<String>,
}

impl ModelProfile {
    /// The memory pool this model's weights live in.
    pub fn pool(&self) -> MemoryPoolKind {
        match self.device {
            DeviceKind::Gpu | DeviceKind::Hybrid => MemoryPoolKind::Gpu,
            DeviceKind::Cpu | DeviceKind::Npu => MemoryPoolKind::System,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "lowercase")]
pub enum ModelState {
    Stopped,
    Loading,
    Ready { port: u16 },
    Failed { error: String },
}

/// What the management API reports per model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelStatus {
    pub id: String,
    pub device: DeviceKind,
    pub pool: MemoryPoolKind,
    pub backend: Option<String>,
    pub pinned: bool,
    /// Estimated resident size used for pool budgeting.
    pub estimated_bytes: Option<u64>,
    #[serde(flatten)]
    pub state: ModelState,
    /// Requests currently being served.
    pub in_flight: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_toml_shape() {
        let p: ModelProfile = serde_json::from_value(serde_json::json!({
            "id": "qwen3-8b",
            "format": "gguf",
            "path": "/m/Qwen3-8B-Q4_K_M.gguf",
            "device": "gpu",
            "context": 16384
        }))
        .unwrap();
        assert_eq!(p.source, ModelSource::Gguf { path: "/m/Qwen3-8B-Q4_K_M.gguf".into() });
        assert_eq!(p.pool(), MemoryPoolKind::Gpu);
        assert!(!p.pinned && p.extra_args.is_empty());

        let npu: ModelProfile = serde_json::from_value(serde_json::json!({
            "id": "qwen3-4b-npu", "format": "flm", "tag": "qwen3-it:4b", "device": "npu"
        }))
        .unwrap();
        assert_eq!(npu.pool(), MemoryPoolKind::System);
    }

    #[test]
    fn status_serializes_flat_state() {
        let s = ModelStatus {
            id: "m".into(),
            device: DeviceKind::Gpu,
            pool: MemoryPoolKind::Gpu,
            backend: Some("llama.cpp".into()),
            pinned: false,
            estimated_bytes: Some(1),
            state: ModelState::Ready { port: 4242 },
            in_flight: 0,
        };
        let v = serde_json::to_value(&s).unwrap();
        assert_eq!(v["state"], "ready");
        assert_eq!(v["port"], 4242);
    }
}
