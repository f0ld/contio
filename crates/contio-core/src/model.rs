//! Installed model profiles and their runtime state (spec §4, §7.2).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::hw::{DeviceKind, MemoryPoolKind};

/// Where a model's weights come from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "format", rename_all = "lowercase")]
pub enum ModelSource {
    /// A local GGUF file (llama.cpp). For split models, the first shard.
    Gguf {
        path: PathBuf,
        /// Vision projector (`--mmproj`) for multimodal models.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mmproj: Option<PathBuf>,
    },
    /// A FastFlowLM model tag, e.g. `qwen3:8b` (NPU).
    Flm { tag: String },
    /// A whisper.cpp GGML model, e.g. `ggml-large-v3-turbo-q8_0.bin`.
    Whisper { path: PathBuf },
}

impl ModelSource {
    /// The local weights file, if the model has one.
    pub fn path(&self) -> Option<&PathBuf> {
        match self {
            Self::Gguf { path, .. } | Self::Whisper { path } => Some(path),
            Self::Flm { .. } => None,
        }
    }
}

/// What a model is served for; decides the backend's mode and API routes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelKind {
    #[default]
    Chat,
    /// `/v1/embeddings`
    Embedding,
    /// `/v1/rerank`
    Reranker,
    /// `/v1/audio/transcriptions` (speech to text)
    Transcription,
}

/// KV cache size per token of context, from the GGUF header
/// (`scripts/catalog/geometry.py`). Sliding-window layers only cache their
/// window, so they stop growing past it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvGeometry {
    pub full_bytes_per_token: u64,
    #[serde(default)]
    pub swa_bytes_per_token: u64,
    #[serde(default)]
    pub swa_window: u32,
}

/// llama.cpp keeps one micro-batch beyond the window in sliding-window caches.
const SWA_EXTRA_TOKENS: u32 = 512;

impl KvGeometry {
    pub fn bytes(&self, context: u32) -> u64 {
        let swa_tokens = context.min(self.swa_window.saturating_add(SWA_EXTRA_TOKENS));
        self.full_bytes_per_token * u64::from(context) + self.swa_bytes_per_token * u64::from(swa_tokens)
    }
}

/// Context assumed when a profile doesn't set one.
pub const DEFAULT_CONTEXT: u32 = 8192;

const MIB: u64 = 1024 * 1024;

/// Resident memory of a llama.cpp model: weights, KV cache and compute buffers.
/// Without a known geometry, assumes an 8B-class model (~64 KiB per token).
pub fn estimate_memory(weight_bytes: u64, kv: Option<&KvGeometry>, context: Option<u32>) -> u64 {
    let ctx = context.unwrap_or(DEFAULT_CONTEXT);
    let kv_bytes = kv.map_or(u64::from(ctx) * 64 * 1024, |g| g.bytes(ctx));
    // compute buffers grow with model width; ~3% of the weights, at least 384 MiB
    let buffers = (weight_bytes * 3 / 100).max(384 * MIB);
    weight_bytes + kv_bytes + buffers
}

/// Resident memory of a whisper.cpp model: weights plus encoder/decoder
/// buffers (~300 MiB for large-v3-turbo with 30 s windows).
pub fn estimate_whisper_memory(weight_bytes: u64) -> u64 {
    weight_bytes + (weight_bytes / 5).max(256 * MIB)
}

/// Where an installed model came from (shown in the GUI, used for updates).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelOrigin {
    /// Catalog model id, e.g. `qwen3.6-35b-a3b`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub catalog: Option<String>,
    /// Hugging Face repo, e.g. `unsloth/Qwen3.6-35B-A3B-GGUF`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo: Option<String>,
    /// Quant label, e.g. `UD-Q4_K_XL`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant: Option<String>,
}

/// One runnable configuration of a model: weights + device + launch options.
/// The `id` is the name clients put in the OpenAI `model` field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelProfile {
    pub id: String,
    #[serde(flatten)]
    pub source: ModelSource,
    pub device: DeviceKind,
    #[serde(default)]
    pub kind: ModelKind,
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
    /// For memory estimates; filled in from the catalog when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv: Option<KvGeometry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<ModelOrigin>,
}

impl ModelProfile {
    /// A profile with default launch options.
    pub fn new(id: impl Into<String>, source: ModelSource, device: DeviceKind) -> Self {
        Self {
            id: id.into(),
            source,
            device,
            kind: ModelKind::Chat,
            context: None,
            pinned: false,
            idle_ttl_secs: None,
            extra_args: Vec::new(),
            kv: None,
            origin: None,
        }
    }

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
    pub kind: ModelKind,
    pub context: Option<u32>,
    /// Registered through the API (removable), as opposed to `config.toml`.
    pub managed: bool,
    /// A `config.toml` model whose launch settings were changed through the API.
    #[serde(default)]
    pub overridden: bool,
    pub origin: Option<ModelOrigin>,
    pub source: ModelSource,
    pub pool: MemoryPoolKind,
    pub backend: Option<String>,
    pub pinned: bool,
    pub idle_ttl_secs: Option<u64>,
    pub extra_args: Vec<String>,
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
        assert_eq!(p.source, ModelSource::Gguf { path: "/m/Qwen3-8B-Q4_K_M.gguf".into(), mmproj: None });
        assert_eq!(p.kind, ModelKind::Chat);
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
            kind: ModelKind::Chat,
            context: None,
            managed: false,
            overridden: false,
            origin: None,
            source: ModelSource::Flm { tag: "t".into() },
            pool: MemoryPoolKind::Gpu,
            backend: Some("llama.cpp".into()),
            pinned: false,
            idle_ttl_secs: None,
            extra_args: vec![],
            estimated_bytes: Some(1),
            state: ModelState::Ready { port: 4242 },
            in_flight: 0,
        };
        let v = serde_json::to_value(&s).unwrap();
        assert_eq!(v["state"], "ready");
        assert_eq!(v["port"], 4242);
    }

    #[test]
    fn kv_geometry_caps_sliding_window_layers() {
        // gemma-4-12b: 16 KiB/token global layers, 320 KiB/token in a 1024 window
        let g = KvGeometry { full_bytes_per_token: 16 << 10, swa_bytes_per_token: 320 << 10, swa_window: 1024 };
        assert_eq!(g.bytes(1000), 1000 * (336 << 10));
        assert_eq!(g.bytes(32768), 32768 * (16 << 10) + 1536 * (320 << 10));
    }

    #[test]
    fn memory_estimate_uses_geometry() {
        let gib = 1u64 << 30;
        let g = KvGeometry { full_bytes_per_token: 20 << 10, ..Default::default() };
        // Qwen3.6-35B-A3B UD-Q4_K_XL at 32k: 20.8 GiB + 640 MiB KV + 3% buffers
        let e = estimate_memory(22_360_456_160, Some(&g), Some(32768));
        assert!(e > 21 * gib + gib / 2 && e < 23 * gib, "{e}");
        assert!(estimate_memory(gib, None, None) > estimate_memory(gib, Some(&g), None), "unknown = pessimistic");
    }

    #[test]
    fn mmproj_round_trips() {
        let p = ModelProfile::new(
            "vl",
            ModelSource::Gguf { path: "/m/a.gguf".into(), mmproj: Some("/m/mmproj.gguf".into()) },
            DeviceKind::Gpu,
        );
        let v = serde_json::to_value(&p).unwrap();
        assert_eq!(v["mmproj"], "/m/mmproj.gguf");
        assert_eq!(serde_json::from_value::<ModelProfile>(v).unwrap(), p);
    }
}
