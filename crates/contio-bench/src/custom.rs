//! Custom prompt sets (spec §8.2): the user's own prompts with checks, as a
//! TOML file (`docs/bench-set.example.toml`):
//!
//! ```toml
//! name = "support-bot"
//! system = "You answer customer questions about the Framework 13."
//!
//! [[items]]
//! prompt = "How much RAM can the Ryzen AI 300 board take?"
//! contains = ["96"]           # all must appear (case-insensitive)
//! not_contains = ["I don't know"]
//! regex = "\\b96 ?GB\\b"
//! ```
//! An item may also set `json_schema` (sent as `response_format`, then
//! validated) and `max_tokens`.

use std::collections::BTreeMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::perf::{Progress, Target, or_cancel};
use crate::quality::{ItemResult, QualityReport};

const DEFAULT_MAX_TOKENS: u32 = 4096;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CustomSet {
    pub name: String,
    #[serde(default)]
    pub system: Option<String>,
    pub items: Vec<CustomItem>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CustomItem {
    #[serde(default)]
    pub id: Option<String>,
    pub prompt: String,
    #[serde(default)]
    pub contains: Vec<String>,
    #[serde(default)]
    pub not_contains: Vec<String>,
    #[serde(default)]
    pub regex: Option<String>,
    #[serde(default)]
    pub json_schema: Option<Value>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

impl CustomSet {
    /// Problems that would make every run fail (bad regex or schema, no items).
    pub fn validate(&self) -> Result<(), String> {
        if self.items.is_empty() {
            return Err("the set has no items".into());
        }
        for (i, it) in self.items.iter().enumerate() {
            let at = it.id.clone().unwrap_or_else(|| format!("item {}", i + 1));
            if let Some(r) = &it.regex {
                regex::Regex::new(r).map_err(|e| format!("{at}: bad regex: {e}"))?;
            }
            if let Some(s) = &it.json_schema {
                jsonschema::validator_for(s).map_err(|e| format!("{at}: bad JSON schema: {e}"))?;
            }
            if it.contains.is_empty() && it.not_contains.is_empty() && it.regex.is_none() && it.json_schema.is_none() {
                return Err(format!("{at}: needs at least one check (contains, not_contains, regex, json_schema)"));
            }
        }
        Ok(())
    }
}

/// Why `text` fails `item`'s checks, if it does.
pub fn check(item: &CustomItem, text: &str) -> Result<(), String> {
    let lower = text.to_lowercase();
    for c in &item.contains {
        if !lower.contains(&c.to_lowercase()) {
            return Err(format!("missing {c:?}"));
        }
    }
    for c in &item.not_contains {
        if lower.contains(&c.to_lowercase()) {
            return Err(format!("contains {c:?}"));
        }
    }
    if let Some(r) = &item.regex {
        let re = regex::Regex::new(r).map_err(|e| format!("bad regex: {e}"))?;
        if !re.is_match(text) {
            return Err(format!("no match for /{r}/"));
        }
    }
    if let Some(schema) = &item.json_schema {
        let v: Value = serde_json::from_str(text.trim()).map_err(|e| format!("not JSON: {e}"))?;
        let validator = jsonschema::validator_for(schema).map_err(|e| format!("bad schema: {e}"))?;
        if let Some(e) = validator.iter_errors(&v).next() {
            return Err(format!("schema: {e}"));
        }
    }
    Ok(())
}

pub async fn run(
    http: &reqwest::Client,
    target: &Target,
    set: &CustomSet,
    reps: u32,
    progress: &dyn Progress,
) -> Result<QualityReport, String> {
    let reps = reps.max(1);
    let mut results = Vec::new();
    let mut truncated = 0;
    for (i, item) in set.items.iter().enumerate() {
        let id = item.id.clone().unwrap_or_else(|| format!("item-{}", i + 1));
        let mut r = ItemResult { id: id.clone(), ..Default::default() };
        let mut elapsed = 0.0;
        for rep in 0..reps {
            progress.step(&format!("{} {}/{}: {id} (run {}/{reps})", set.name, i + 1, set.items.len(), rep + 1));
            let started = Instant::now();
            let mut messages = Vec::new();
            if let Some(s) = &set.system {
                messages.push(json!({ "role": "system", "content": s }));
            }
            messages.push(json!({ "role": "user", "content": item.prompt }));
            let max = item.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
            let mut body = json!({ "model": target.model, "messages": messages, "max_tokens": max });
            if let Some(schema) = &item.json_schema {
                body["response_format"] = json!({ "type": "json_schema", "json_schema": { "name": "answer", "schema": schema } });
            }
            target.apply_extra(&mut body);
            let req = async {
                let res = http
                    .post(format!("{}/v1/chat/completions", target.base_url))
                    .json(&body)
                    .send()
                    .await
                    .map_err(|e| format!("request failed: {e}"))?;
                let status = res.status();
                let v: Value = res.json().await.map_err(|e| format!("bad response: {e}"))?;
                if status.is_success() { Ok(v) } else { Err(format!("HTTP {status}: {}", v["error"]["message"])) }
            };
            let outcome = match or_cancel(progress, req).await {
                Err(e) if progress.cancelled() => return Err(e),
                Err(e) => Err(e),
                Ok(v) => {
                    r.completion_tokens += v["usage"]["completion_tokens"].as_u64().unwrap_or(0);
                    if v["choices"][0]["finish_reason"] == "length" {
                        truncated += 1;
                        Err(format!("hit the {max}-token limit"))
                    } else {
                        check(item, v["choices"][0]["message"]["content"].as_str().unwrap_or_default())
                    }
                }
            };
            elapsed += started.elapsed().as_secs_f64();
            r.runs += 1;
            match outcome {
                Ok(()) => r.passed += 1,
                Err(e) => r.failures.push(e),
            }
        }
        r.mean_s = elapsed / f64::from(reps);
        results.push(r);
    }
    let (p, n) = results.iter().fold((0, 0), |a, r| (a.0 + r.passed, a.1 + r.runs));
    Ok(QualityReport {
        suite: "custom".into(),
        pass_rate: if n == 0 { 0.0 } else { f64::from(p) / f64::from(n) },
        categories: BTreeMap::new(),
        reps,
        truncated,
        items: results,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(toml_src: &str) -> CustomItem {
        toml::from_str(toml_src).unwrap()
    }

    #[test]
    fn checks() {
        let it = item("prompt = 'x'\ncontains = ['96']\nnot_contains = [\"don't know\"]\nregex = '\\b96 ?GB\\b'\n");
        assert!(check(&it, "It supports up to 96 GB.").is_ok());
        assert!(check(&it, "Up to 96GB, but I don't know").unwrap_err().contains("don't know"));
        assert!(check(&it, "up to 64 GB").unwrap_err().contains("missing"));
        assert!(check(&it, "96 gigabytes").unwrap_err().contains("no match"));
        let js = item("prompt = 'x'\n[json_schema]\ntype = 'object'\nrequired = ['a']\n");
        assert!(check(&js, r#"{"a": 1}"#).is_ok());
        assert!(check(&js, r#"{"b": 1}"#).unwrap_err().starts_with("schema"));
    }

    #[test]
    fn example_file_is_valid() {
        let set: CustomSet = toml::from_str(include_str!("../../../docs/bench-set.example.toml")).unwrap();
        set.validate().unwrap();
        let bad: CustomSet = toml::from_str("name = 'b'\n[[items]]\nprompt = 'x'\nregex = '('\n").unwrap();
        assert!(bad.validate().unwrap_err().contains("bad regex"));
        let empty: CustomSet = toml::from_str("name = 'b'\n[[items]]\nprompt = 'x'\n").unwrap();
        assert!(empty.validate().unwrap_err().contains("at least one check"));
    }
}
