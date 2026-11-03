//! NPU models from FastFlowLM's `share/flm/model_list.json`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// XDNA2 (Strix / Krackan Point); the list also has entries for other NPUs.
pub const PLATFORM_XDNA2: &str = "aie2p";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlmModel {
    /// `family:variant`, the name `flm pull`/`flm serve` take.
    pub tag: String,
    pub name: String,
    pub family: String,
    pub size_bytes: Option<u64>,
    /// Resident memory in bytes (FLM's `footprint`, GB).
    pub footprint_bytes: Option<u64>,
    pub default_context: Option<u32>,
    pub quant: Option<String>,
    /// FLM's labels, e.g. `tool-calling`, `vision`, `single-turn`.
    pub labels: Vec<String>,
    pub repo_url: Option<String>,
}

impl FlmModel {
    /// `*-flash` variants: 1k context, one turn only (m0-npu.md).
    pub fn single_turn(&self) -> bool {
        self.labels.iter().any(|l| l == "single-turn")
    }

    pub fn is_embedding(&self) -> bool {
        self.labels.iter().any(|l| l == "embeddings")
    }

    pub fn is_transcription(&self) -> bool {
        self.labels.iter().any(|l| l == "transcription")
    }

    /// Same vocabulary as [`crate::CatalogModel::capabilities`].
    pub fn capabilities(&self) -> Vec<String> {
        fn map(l: &str) -> Option<&str> {
            match l {
                "tool-calling" => Some("tools"),
                "reasoning" | "vision" | "audio" => Some(l),
                _ => None,
            }
        }
        self.labels.iter().filter_map(|l| map(l)).map(str::to_owned).collect()
    }
}

/// Models in `json` that run on `platform`, in file order.
pub fn parse_model_list(json: &str, platform: &str) -> Result<Vec<FlmModel>, String> {
    let root: Value = serde_json::from_str(json).map_err(|e| format!("model_list.json: {e}"))?;
    let families = root["models"].as_object().ok_or("model_list.json: no `models` object")?;
    let mut out = Vec::new();
    for (family, variants) in families {
        for (variant, m) in variants.as_object().into_iter().flatten() {
            let on_platform = m["supported_platforms"].as_array().is_some_and(|p| p.iter().any(|x| x == platform));
            if !on_platform {
                continue;
            }
            let strings = |v: &Value| -> Vec<String> {
                v.as_array().into_iter().flatten().filter_map(|x| x.as_str().map(str::to_owned)).collect()
            };
            out.push(FlmModel {
                tag: format!("{family}:{variant}"),
                name: m["name"].as_str().unwrap_or(family).to_owned(),
                family: m["details"]["family"].as_str().unwrap_or(family).to_owned(),
                size_bytes: m["size"].as_u64(),
                footprint_bytes: m["footprint"].as_f64().map(|gb| (gb * 1e9) as u64),
                default_context: m["default_context_length"].as_u64().map(|c| c as u32),
                quant: m["details"]["quantization_level"].as_str().map(str::to_owned),
                labels: strings(&m["label"]),
                repo_url: m["url"].as_str().map(str::to_owned),
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIST: &str = r#"{"model_path": "models", "models": {
        "qwen3-it": {"4b": {"supported_platforms": ["aie2p"], "name": "Qwen3-4B-Instruct-2507-NPU2",
            "url": "https://huggingface.co/FastFlowLM/Qwen3-4B-Instruct-2507-NPU2", "size": 4000000000,
            "default_context_length": 32768, "details": {"family": "qwen3-it", "quantization_level": "Q4_1"},
            "label": ["tool-calling"], "footprint": 3.1}},
        "qwen3vl-flash": {"4b": {"supported_platforms": ["aie2p"], "name": "x", "label": ["vision", "single-turn"]}},
        "qwen3-core": {"4b": {"supported_platforms": ["dev04"], "name": "other NPU"}}
    }}"#;

    #[test]
    fn parses_platform_models() {
        let m = parse_model_list(LIST, PLATFORM_XDNA2).unwrap();
        assert_eq!(m.len(), 2, "dev04 entries skipped");
        assert_eq!(m[0].tag, "qwen3-it:4b");
        assert_eq!(m[0].footprint_bytes, Some(3_100_000_000));
        assert_eq!(m[0].capabilities(), ["tools"]);
        assert!(!m[0].single_turn() && m[1].single_turn());
        assert!(parse_model_list("{}", PLATFORM_XDNA2).is_err());
    }
}
