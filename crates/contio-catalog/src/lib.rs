//! The model catalog (spec §4): family → model → quantization.
//!
//! The curated part is compiled in from `registry/catalog.toml` and
//! `registry/geometry.json`. Quant files and sizes come from Hugging Face at
//! runtime ([`hf`]), NPU models from FastFlowLM's model list ([`flm`]).

pub mod fit;
pub mod flm;
pub mod hf;
pub mod quant;

use std::collections::HashMap;

use contio_core::model::{KvGeometry, ModelKind};
use serde::{Deserialize, Serialize};

const CATALOG_TOML: &str = include_str!("../../../registry/catalog.toml");
const GEOMETRY_JSON: &str = include_str!("../../../registry/geometry.json");

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Family {
    pub id: String,
    pub name: String,
    pub vendor: String,
    pub license: String,
    pub summary: String,
}

/// Facts read from the GGUF header by `scripts/catalog/geometry.py`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Geometry {
    pub arch: String,
    pub context_length: Option<u32>,
    /// Total parameters.
    pub params: Option<u64>,
    pub kv: KvGeometry,
    pub tools: bool,
    pub reasoning: bool,
    /// The file the header was read from.
    pub probed: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogModel {
    pub id: String,
    pub family: String,
    pub name: String,
    /// Hugging Face repo with the GGUF files.
    pub repo: String,
    #[serde(default)]
    pub kind: ModelKind,
    /// Active parameters in billions (MoE).
    #[serde(default)]
    pub active_b: Option<f32>,
    #[serde(default)]
    pub recommended: Option<String>,
    /// Default context for new installs.
    #[serde(default)]
    pub context: Option<u32>,
    #[serde(default)]
    pub tags: Vec<String>,
    /// FastFlowLM tags of the same model (NPU).
    #[serde(default)]
    pub flm: Vec<String>,
    #[serde(default)]
    pub notes: Option<String>,
    #[serde(skip_deserializing)]
    pub geometry: Option<Geometry>,
}

impl CatalogModel {
    /// Capability labels: from the template (tools, reasoning), the repo
    /// (vision = has a projector) and the curated tags.
    pub fn capabilities(&self, has_mmproj: bool) -> Vec<String> {
        let g = self.geometry.as_ref();
        let mut caps = Vec::new();
        if self.kind == ModelKind::Chat {
            if g.is_some_and(|g| g.tools) {
                caps.push("tools".to_owned());
            }
            if g.is_some_and(|g| g.reasoning) {
                caps.push("reasoning".to_owned());
            }
        }
        if has_mmproj {
            caps.push("vision".to_owned());
        }
        for t in &self.tags {
            if !caps.contains(t) {
                caps.push(t.clone());
            }
        }
        caps
    }

    /// Context to install with: the curated default, capped at the maximum.
    pub fn default_context(&self) -> u32 {
        let max = self.geometry.as_ref().and_then(|g| g.context_length).unwrap_or(u32::MAX);
        self.context.unwrap_or(contio_core::model::DEFAULT_CONTEXT).min(max)
    }
}

#[derive(Debug, Clone, Default)]
pub struct Catalog {
    pub families: Vec<Family>,
    pub models: Vec<CatalogModel>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CatalogFile {
    families: Vec<Family>,
    models: Vec<CatalogModel>,
}

impl Catalog {
    /// The catalog shipped with this build.
    pub fn builtin() -> Self {
        Self::parse(CATALOG_TOML, GEOMETRY_JSON).expect("registry/ is validated by tests")
    }

    pub fn parse(catalog_toml: &str, geometry_json: &str) -> Result<Self, String> {
        let file: CatalogFile = toml::from_str(catalog_toml).map_err(|e| format!("catalog.toml: {e}"))?;
        let mut geometry: HashMap<String, Geometry> =
            serde_json::from_str(geometry_json).map_err(|e| format!("geometry.json: {e}"))?;
        let mut models = file.models;
        let mut ids = std::collections::HashSet::new();
        for m in &mut models {
            if !ids.insert(m.id.clone()) {
                return Err(format!("duplicate catalog id `{}`", m.id));
            }
            if !file.families.iter().any(|f| f.id == m.family) {
                return Err(format!("`{}` has unknown family `{}`", m.id, m.family));
            }
            m.geometry = geometry.remove(&m.repo);
        }
        Ok(Self { families: file.families, models })
    }

    pub fn model(&self, id: &str) -> Option<&CatalogModel> {
        self.models.iter().find(|m| m.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_catalog_is_valid_and_complete() {
        let c = Catalog::builtin();
        assert!(c.models.len() >= 40, "{}", c.models.len());
        for m in &c.models {
            let g = m.geometry.as_ref().unwrap_or_else(|| panic!("{}: run scripts/catalog/geometry.py", m.id));
            assert!(g.kv.full_bytes_per_token + g.kv.swa_bytes_per_token > 0, "{}: no KV geometry", m.id);
            assert!(m.id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || "-.".contains(c)), "{}", m.id);
            let rec = m.recommended.as_deref().unwrap_or_default();
            assert_eq!(quant::quant_label(&g.probed).as_deref(), Some(rec), "{}: probed file is not the recommended quant", m.id);
        }
        let q = c.model("qwen3.6-35b-a3b").unwrap();
        assert_eq!(q.default_context(), 32768);
        assert_eq!(q.capabilities(true), ["tools", "reasoning", "vision", "code"]);
        let e = c.model("embeddinggemma-300m").unwrap();
        assert_eq!(e.kind, ModelKind::Embedding);
        assert!(e.capabilities(false).is_empty(), "embedding models don't advertise chat features");
    }

    #[test]
    fn rejects_inconsistent_catalogs() {
        let fam = "[[families]]\nid='f'\nname='F'\nvendor='v'\nlicense='l'\nsummary='s'\n";
        let model = "[[models]]\nid='a'\nfamily='f'\nname='A'\nrepo='o/r'\n";
        assert!(Catalog::parse(&format!("{fam}{model}{model}"), "{}").unwrap_err().contains("duplicate"));
        let orphan = model.replace("family='f'", "family='nope'");
        assert!(Catalog::parse(&format!("{fam}{orphan}"), "{}").unwrap_err().contains("unknown family"));
        let c = Catalog::parse(&format!("{fam}{model}"), "{}").unwrap();
        assert_eq!(c.models[0].default_context(), 8192);
        assert!(c.models[0].geometry.is_none());
    }
}
