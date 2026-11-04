//! Benchmarks (spec §8): speed, energy and memory measured through the
//! OpenAI API the daemon serves ([`perf`]), quality suites, and a SQLite
//! result store ([`store`]).
//!
//! The daemon owns model lifecycle (load timing, memory baselines, unloading
//! other models) and runs these as background jobs.

pub mod monitor;
pub mod perf;
pub mod store;

use std::collections::BTreeMap;

use contio_core::hw::{MemoryPoolKind, PowerProfile};
use serde::{Deserialize, Serialize};

pub use monitor::Monitor;
pub use perf::{PerfOptions, PerfReport, Progress, Target};

/// How much to measure (spec §8.3).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Preset {
    /// ~1 minute: one short prompt.
    #[default]
    Quick,
    /// Short and 4k prompts, 3 runs each, 4 parallel requests.
    Standard,
    /// Up to 16k prompts, 5 runs each.
    Full,
}

impl Preset {
    pub fn perf_options(self) -> PerfOptions {
        match self {
            Self::Quick => PerfOptions { prompt_tokens: vec![512], gen_tokens: 64, reps: 1, concurrency: 0 },
            Self::Standard => PerfOptions { prompt_tokens: vec![512, 4096], gen_tokens: 128, reps: 3, concurrency: 4 },
            Self::Full => {
                PerfOptions { prompt_tokens: vec![512, 4096, 16384], gen_tokens: 256, reps: 5, concurrency: 4 }
            }
        }
    }
}

/// What was running and how, so results stay comparable (spec §8.1).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Conditions {
    pub power_profile: Option<PowerProfile>,
    pub on_ac: Option<bool>,
    pub battery_percent: Option<u8>,
    /// e.g. `llama.cpp (native: …/b12068-prebuilt/bin)`
    pub backend: Option<String>,
    pub contio_version: String,
    pub cpu: Option<String>,
    /// Other models loaded during the run (they share memory bandwidth).
    pub other_models: Vec<String>,
}

/// Load time and memory, measured by the daemon around loading the model.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LoadReport {
    /// `None` if the model was already loaded (and couldn't be reloaded).
    pub load_s: Option<f64>,
    /// Pool usage growth from before loading to the peak during the run.
    pub memory_delta: BTreeMap<MemoryPoolKind, u64>,
    /// Peak resident memory of the backend process (includes mapped weights).
    pub peak_rss_bytes: Option<u64>,
    /// What the daemon's estimator predicted (for checking it).
    pub estimated_bytes: Option<u64>,
}

/// Everything a performance run produced.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PerfRun {
    pub load: LoadReport,
    pub perf: PerfReport,
}

impl PerfRun {
    /// Headline numbers for tables and sorting, e.g. `pp512_tps`, `tg_tps`.
    pub fn metrics(&self) -> BTreeMap<String, f64> {
        let mut m = BTreeMap::new();
        let mut put = |k: String, v: Option<f64>| {
            if let Some(v) = v.filter(|v| v.is_finite()) {
                m.insert(k, v);
            }
        };
        put("load_s".into(), self.load.load_s);
        put("idle_watts".into(), self.perf.idle_watts);
        for (pool, bytes) in &self.load.memory_delta {
            let name = match pool {
                MemoryPoolKind::Gpu => "gpu",
                MemoryPoolKind::System => "system",
            };
            put(format!("mem_{name}_bytes"), Some(*bytes as f64));
        }
        put("estimated_bytes".into(), self.load.estimated_bytes.map(|b| b as f64));
        put("rss_bytes".into(), self.load.peak_rss_bytes.map(|b| b as f64));
        for p in &self.perf.prompts {
            let n = p.target_tokens;
            put(format!("pp{n}_tps"), p.summary.prefill_tps);
            put(format!("ttft{n}_s"), p.summary.ttft_s);
        }
        // decode speed and energy at the shortest prompt: the "chat" case
        if let Some(p) = self.perf.prompts.first() {
            put("tg_tps".into(), p.summary.decode_tps);
            put("watts".into(), p.summary.avg_watts);
            put("j_per_token".into(), p.summary.joules_per_token);
        }
        if let Some(c) = &self.perf.concurrency {
            put(format!("conc{}_tps", c.requests), Some(c.aggregate_tps));
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf::{PromptResult, Summary};

    #[test]
    fn headline_metrics() {
        let run = PerfRun {
            load: LoadReport {
                load_s: Some(2.5),
                memory_delta: [(MemoryPoolKind::Gpu, 5 << 30)].into(),
                estimated_bytes: Some(6 << 30),
                peak_rss_bytes: None,
            },
            perf: PerfReport {
                idle_watts: Some(5.0),
                prompts: vec![
                    PromptResult {
                        target_tokens: 512,
                        summary: Summary { decode_tps: Some(22.0), prefill_tps: Some(400.0), ttft_s: Some(1.3), joules_per_token: Some(0.9), ..Default::default() },
                        samples: vec![],
                    },
                    PromptResult {
                        target_tokens: 4096,
                        summary: Summary { decode_tps: Some(20.0), prefill_tps: Some(f64::NAN), ..Default::default() },
                        samples: vec![],
                    },
                ],
                ..Default::default()
            },
        };
        let m = run.metrics();
        assert_eq!(m["tg_tps"], 22.0, "decode at the shortest prompt");
        assert_eq!(m["pp512_tps"], 400.0);
        assert!(!m.contains_key("pp4096_tps"), "non-finite values are dropped");
        assert_eq!(m["mem_gpu_bytes"], (5u64 << 30) as f64);
        assert_eq!(m["j_per_token"], 0.9);
    }

    #[test]
    fn presets_grow() {
        let q = Preset::Quick.perf_options();
        let f = Preset::Full.perf_options();
        assert!(q.prompt_tokens.len() < f.prompt_tokens.len() && q.reps < f.reps);
        assert_eq!(serde_json::to_value(Preset::Standard).unwrap(), "standard");
    }
}
