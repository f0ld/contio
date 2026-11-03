//! Group a repo's files into quantizations (spec §4.1).
//!
//! GGUF repos name files `<model>-<QUANT>.gguf`, sometimes split into
//! `…-00001-of-00003.gguf` shards inside a `<QUANT>/` directory, next to
//! auxiliary files (vision projectors, imatrix data, draft models for
//! speculative decoding) that aren't quantizations of the model itself.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepoFile {
    pub path: String,
    pub size: u64,
    /// LFS object id (SHA-256), when the file is stored in LFS.
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Quant {
    /// e.g. `Q4_K_M`, `UD-Q4_K_XL`, `MXFP4`
    pub label: String,
    /// One file, or all shards in order.
    pub files: Vec<RepoFile>,
    pub size: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepoListing {
    /// Smallest first.
    pub quants: Vec<Quant>,
    /// Vision projectors.
    pub mmproj: Vec<RepoFile>,
}

impl RepoListing {
    pub fn quant(&self, label: &str) -> Option<&Quant> {
        self.quants.iter().find(|q| q.label.eq_ignore_ascii_case(label))
    }

    /// The projector to install with a vision model: F16 is supported by
    /// every llama.cpp backend; BF16/F32 are bigger or slower.
    pub fn preferred_mmproj(&self) -> Option<&RepoFile> {
        ["F16", "BF16", "Q8_0", "F32"]
            .iter()
            .find_map(|q| self.mmproj.iter().find(|f| quant_label(&f.path).as_deref() == Some(q)))
            .or_else(|| self.mmproj.first())
    }
}

const AUX_PREFIXES: [&str; 6] = ["mtp-", "eagle", "dflash", "mmproj", "imatrix", "draft"];

fn file_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// Not a quantization of the model: projector, imatrix, draft/MTP heads.
pub fn is_aux(path: &str) -> bool {
    let name = file_name(path).to_ascii_lowercase();
    AUX_PREFIXES.iter().any(|p| name.starts_with(p)) || format!("/{}", path.to_ascii_lowercase()).contains("/mtp/")
}

fn is_mmproj(path: &str) -> bool {
    file_name(path).to_ascii_lowercase().starts_with("mmproj") && path.ends_with(".gguf")
}

/// `(stem, Some((index, total)))` for shard files.
fn shard(path: &str) -> (&str, Option<(u32, u32)>) {
    let stem = file_name(path).strip_suffix(".gguf").unwrap_or(file_name(path));
    let parsed = stem.rsplit_once("-of-").and_then(|(rest, total)| {
        let (base, index) = rest.rsplit_once('-')?;
        (index.len() == 5 && total.len() == 5).then_some((base, (index.parse().ok()?, total.parse().ok()?)))
    });
    match parsed {
        Some((base, s)) => (base, Some(s)),
        None => (stem, None),
    }
}

fn looks_like_quant(token: &str) -> bool {
    let t = token.to_ascii_lowercase();
    let digit_after = |p: &str| t.strip_prefix(p).is_some_and(|r| r.starts_with(|c: char| c.is_ascii_digit()));
    digit_after("q") || digit_after("iq") || digit_after("tq")
        || ["mxfp4", "bf16", "f16", "f32", "fp16", "fp8"].iter().any(|p| t.starts_with(p))
}

/// The quant label in a GGUF file name: the last quant-like token of the stem
/// (`UD-` kept as part of it). Same rule as `scripts/catalog/geometry.py`.
pub fn quant_label(path: &str) -> Option<String> {
    let (stem, _) = shard(path);
    let tokens: Vec<&str> = stem.split(['-', '.']).collect();
    let i = tokens.iter().rposition(|t| looks_like_quant(t))?;
    let q = tokens[i].to_ascii_uppercase();
    Some(if i > 0 && tokens[i - 1].eq_ignore_ascii_case("ud") { format!("UD-{q}") } else { q })
}

/// Group `files` into quants (complete shard sets only) and projectors.
pub fn parse_tree(files: &[RepoFile]) -> RepoListing {
    let mut listing = RepoListing::default();
    let mut shards: Vec<(String, u32, u32, &RepoFile)> = Vec::new();
    for f in files.iter().filter(|f| f.path.ends_with(".gguf")) {
        if is_mmproj(&f.path) {
            listing.mmproj.push(f.clone());
            continue;
        }
        if is_aux(&f.path) {
            continue;
        }
        let Some(label) = quant_label(&f.path) else { continue };
        // ARM repacks (Q4_0_4_4 …) are converted on load by current llama.cpp
        if label.starts_with("Q4_0_") {
            continue;
        }
        match shard(&f.path).1 {
            Some((i, n)) => shards.push((label, i, n, f)),
            None => {
                if listing.quant(&label).is_none() {
                    listing.quants.push(Quant { label, size: f.size, files: vec![f.clone()] });
                }
            }
        }
    }
    shards.sort_by(|a, b| (&a.0, a.1).cmp(&(&b.0, b.1)));
    for group in shards.chunk_by(|a, b| a.0 == b.0) {
        let (label, _, total, _) = &group[0];
        let complete = group.len() == *total as usize && group.iter().enumerate().all(|(k, s)| s.1 == k as u32 + 1);
        if complete && listing.quant(label).is_none() {
            let files: Vec<RepoFile> = group.iter().map(|s| s.3.clone()).collect();
            listing.quants.push(Quant { label: label.clone(), size: files.iter().map(|f| f.size).sum(), files });
        }
    }
    listing.quants.sort_by_key(|q| q.size);
    listing
}

/// Average bits per weight.
pub fn bits_per_weight(size: u64, params: u64) -> Option<f32> {
    (params > 0).then(|| (size as f64 * 8.0 / params as f64) as f32)
}

/// Rough quality expectation from bits per weight (spec §4.4 "heuristic tiers").
pub fn quality_tier(bpw: f32) -> &'static str {
    match bpw {
        b if b >= 7.5 => "near-lossless",
        b if b >= 5.5 => "excellent",
        b if b >= 4.5 => "very good",
        b if b >= 3.5 => "good",
        b if b >= 2.7 => "degraded",
        _ => "experimental",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(path: &str, size: u64) -> RepoFile {
        RepoFile { path: path.into(), size, sha256: None }
    }

    #[test]
    fn labels() {
        for (path, want) in [
            ("Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf", Some("UD-Q4_K_XL")),
            ("gpt-oss-20b-MXFP4.gguf", Some("MXFP4")),
            ("Qwen3.6-35B-A3B-MXFP4_MOE.gguf", Some("MXFP4_MOE")),
            ("BF16/Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf", Some("BF16")),
            ("Llama-3.3-70B-Instruct-Q5_K_M/Llama-3.3-70B-Instruct-Q5_K_M-00002-of-00002.gguf", Some("Q5_K_M")),
            ("qwen3-8b-q4_k_m.gguf", Some("Q4_K_M")),
            ("embeddinggemma-300M-Q8_0.gguf", Some("Q8_0")),
            ("Qwen3-8B-IQ4_XS.gguf", Some("IQ4_XS")),
            ("Qwen3-4B.gguf", None),
        ] {
            assert_eq!(quant_label(path).as_deref(), want, "{path}");
        }
    }

    #[test]
    fn groups_quants_shards_and_projectors() {
        let tree = [
            f("M-UD-Q4_K_XL.gguf", 20),
            f("M-Q8_0.gguf", 36),
            f("BF16/M-BF16-00002-of-00002.gguf", 19),
            f("BF16/M-BF16-00001-of-00002.gguf", 50),
            f("Q6_K/M-Q6_K-00001-of-00002.gguf", 20), // second shard missing
            f("mmproj-F32.gguf", 2),
            f("mmproj-F16.gguf", 1),
            f("MTP/mtp-M-Q8_0.gguf", 1),
            f("mtp-M.gguf", 1),
            f("eagle3-M-Q8_0.gguf", 1),
            f("imatrix_unsloth.gguf", 1),
            f("M-Q4_0_4_4.gguf", 18),
            f("README.md", 1),
        ];
        let l = parse_tree(&tree);
        let labels: Vec<&str> = l.quants.iter().map(|q| q.label.as_str()).collect();
        assert_eq!(labels, ["UD-Q4_K_XL", "Q8_0", "BF16"]);
        let bf16 = l.quant("bf16").unwrap();
        assert_eq!(bf16.size, 69);
        assert!(bf16.files[0].path.contains("00001"), "shards in order");
        assert_eq!(l.mmproj.len(), 2);
        assert_eq!(l.preferred_mmproj().unwrap().path, "mmproj-F16.gguf");
    }

    #[test]
    fn tiers() {
        // Qwen3.6-35B-A3B: 34.66 B params
        let bpw = bits_per_weight(22_360_456_160, 34_660_610_688).unwrap();
        assert!((bpw - 5.16).abs() < 0.01, "{bpw}");
        assert_eq!(quality_tier(bpw), "very good");
        assert_eq!(quality_tier(8.5), "near-lossless");
        assert_eq!(quality_tier(2.0), "experimental");
        assert!(bits_per_weight(1, 0).is_none());
    }
}
