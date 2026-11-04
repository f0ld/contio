//! Answer-quality suites (spec §8.2) on small fixed subsets of public
//! benchmarks (`registry/bench/`, fetched by `scripts/bench/fetch_subsets.py`):
//! reasoning (GSM8K), knowledge (MMLU-Pro) and coding (HumanEval).
//!
//! Generated code runs only in a Podman sandbox: no network, read-only root,
//! no capabilities, CPU/memory/process/time limits, an unprivileged user.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Stdio;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::process::Command;

use crate::perf::{Progress, Target};

const GSM8K: &str = include_str!("../../../registry/bench/reasoning_gsm8k.json");
const MMLU_PRO: &str = include_str!("../../../registry/bench/knowledge_mmlu_pro.json");
const HUMANEVAL: &str = include_str!("../../../registry/bench/coding_humaneval.json");

/// Thinking models reason at length before answering.
const MAX_TOKENS: u32 = 8192;
/// Python for the coding sandbox (small; pulled once).
pub const SANDBOX_IMAGE: &str = "docker.io/library/python:3.12-alpine";
const RUN_TIMEOUT_SECS: u32 = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QualitySuite {
    Reasoning,
    Knowledge,
    Coding,
}

impl QualitySuite {
    pub fn name(self) -> &'static str {
        match self {
            Self::Reasoning => "reasoning",
            Self::Knowledge => "knowledge",
            Self::Coding => "coding",
        }
    }

    fn items(self) -> Vec<Item> {
        let raw = match self {
            Self::Reasoning => GSM8K,
            Self::Knowledge => MMLU_PRO,
            Self::Coding => HUMANEVAL,
        };
        let doc: Value = serde_json::from_str(raw).expect("registry/bench/ subsets are validated by tests");
        doc["items"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|v| match self {
                Self::Reasoning => Item::Math {
                    id: v["id"].as_str().unwrap_or_default().into(),
                    question: v["question"].as_str().unwrap_or_default().into(),
                    answer: v["answer"].as_f64().unwrap_or(f64::NAN),
                },
                Self::Knowledge => Item::Choice {
                    id: v["id"].as_str().unwrap_or_default().into(),
                    category: v["category"].as_str().unwrap_or_default().into(),
                    question: v["question"].as_str().unwrap_or_default().into(),
                    options: v["options"].as_array().into_iter().flatten().filter_map(|o| o.as_str().map(str::to_owned)).collect(),
                    answer: v["answer"].as_str().unwrap_or_default().into(),
                },
                Self::Coding => Item::Code {
                    id: v["id"].as_str().unwrap_or_default().into(),
                    prompt: v["prompt"].as_str().unwrap_or_default().into(),
                    test: v["test"].as_str().unwrap_or_default().into(),
                    entry_point: v["entry_point"].as_str().unwrap_or_default().into(),
                },
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
enum Item {
    Math { id: String, question: String, answer: f64 },
    Choice { id: String, category: String, question: String, options: Vec<String>, answer: String },
    Code { id: String, prompt: String, test: String, entry_point: String },
}

impl Item {
    fn id(&self) -> &str {
        match self {
            Self::Math { id, .. } | Self::Choice { id, .. } | Self::Code { id, .. } => id,
        }
    }

    fn category(&self) -> Option<&str> {
        match self {
            Self::Choice { category, .. } => Some(category),
            _ => None,
        }
    }

    fn prompt(&self) -> String {
        match self {
            Self::Math { question, .. } => format!(
                "{question}\n\nSolve the problem. End your reply with a final line of the form \"Answer: <number>\"."
            ),
            Self::Choice { question, options, .. } => {
                let opts: Vec<String> =
                    options.iter().enumerate().map(|(i, o)| format!("{}. {o}", letter(i))).collect();
                format!(
                    "{question}\n\nOptions:\n{}\n\nChoose the correct option. End your reply with a final line of the form \"Answer: <letter>\".",
                    opts.join("\n")
                )
            }
            Self::Code { prompt, .. } => format!(
                "Complete the following Python function. Reply with the complete function (including the signature and any imports it needs) in one ```python code block.\n\n```python\n{prompt}```"
            ),
        }
    }
}

fn letter(i: usize) -> char {
    char::from(b'A' + i as u8)
}

/// The number after the last "Answer:", else the last number in the text.
pub fn extract_number(text: &str) -> Option<f64> {
    let tail = text.rfind("Answer:").or_else(|| text.rfind("answer:")).map_or(text, |i| &text[i..]);
    let numbers = |s: &str| -> Vec<f64> {
        let mut out = Vec::new();
        let mut cur = String::new();
        for c in s.chars().chain(std::iter::once(' ')) {
            if c.is_ascii_digit() || (c == '.' && !cur.is_empty()) || (c == '-' && cur.is_empty()) {
                cur.push(c);
            } else if c == ',' && !cur.is_empty() {
                // thousands separator
            } else {
                let t = cur.trim_end_matches('.');
                if let Ok(v) = t.parse::<f64>() {
                    out.push(v);
                }
                cur.clear();
            }
        }
        out
    };
    let in_tail = numbers(tail);
    if tail.len() < text.len() && !in_tail.is_empty() {
        return in_tail.first().copied();
    }
    numbers(text).last().copied()
}

/// The option letter after the last "Answer:".
pub fn extract_letter(text: &str, options: usize) -> Option<char> {
    let i = text.rfind("Answer:").or_else(|| text.rfind("answer:"))?;
    let rest = text[i + 7..].replace("\\boxed{", "").replace("\\text{", "").replace("\\mathrm{", "");
    rest.chars()
        .filter(|c| !matches!(c, ' ' | '*' | '(' | '[' | '$' | '\\' | '{' | '}'))
        .find(|c| !c.is_whitespace())
        .map(|c| c.to_ascii_uppercase())
        .filter(|c| c.is_ascii_uppercase() && (*c as usize) < usize::from(b'A') + options)
}

/// The last fenced code block (```python preferred).
pub fn extract_code(text: &str) -> Option<String> {
    let mut blocks = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("```") {
        let after = &rest[start + 3..];
        let (lang, body) = after.split_once('\n')?;
        let Some(end) = body.find("```") else { break };
        blocks.push((lang.trim().to_lowercase(), body[..end].to_owned()));
        rest = &body[end + 3..];
    }
    blocks
        .iter()
        .rev()
        .find(|(l, _)| l == "python" || l == "py")
        .or(blocks.last())
        .map(|(_, b)| b.clone())
}

/// The program the sandbox runs: the prompt's imports, the model's code, the tests.
fn program(prompt: &str, code: &str, test: &str, entry_point: &str) -> String {
    let imports: Vec<&str> =
        prompt.lines().filter(|l| l.starts_with("import ") || l.starts_with("from ")).collect();
    // a reply with only the body: put it under the prompt's signature
    let body = if code.contains(&format!("def {entry_point}")) { code.to_owned() } else { format!("{prompt}{code}") };
    format!("{}\n\n{body}\n\n{test}\n\ncheck({entry_point})\n", imports.join("\n"))
}

/// What a coding run needs; checked before a coding run starts.
pub async fn sandbox_ready() -> Result<(), String> {
    let ok = Command::new("podman")
        .args(["image", "exists", SANDBOX_IMAGE])
        .status()
        .await
        .map_err(|e| format!("podman is needed to run generated code safely: {e}"))?;
    if ok.success() {
        return Ok(());
    }
    let out = Command::new("podman")
        .args(["pull", "-q", SANDBOX_IMAGE])
        .output()
        .await
        .map_err(|e| format!("podman pull: {e}"))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(format!("podman pull {SANDBOX_IMAGE} failed: {}", String::from_utf8_lossy(&out.stderr).trim()))
    }
}

/// Run `source` with Python in a locked-down container. `Err` describes the failure.
pub async fn run_sandboxed(source: &str, work: &Path) -> Result<(), String> {
    let dir = tempdir_in(work)?;
    std::fs::write(dir.join("main.py"), source).map_err(|e| e.to_string())?;
    let mount = format!("{}:/work:ro,Z", dir.display());
    let mut cmd = Command::new("podman");
    cmd.args([
        "run",
        "--rm",
        "--network=none",
        "--read-only",
        "--cap-drop=all",
        "--security-opt=no-new-privileges",
        "--pids-limit=64",
        "--memory=256m",
        "--cpus=1",
        "--user=65534:65534",
        "--tmpfs=/tmp:size=16m",
        "-v",
        &mount,
        SANDBOX_IMAGE,
        "timeout",
        &RUN_TIMEOUT_SECS.to_string(),
        "python3",
        "/work/main.py",
    ])
    .stdin(Stdio::null())
    .kill_on_drop(true);
    let out = tokio::time::timeout(Duration::from_secs(u64::from(RUN_TIMEOUT_SECS) + 30), cmd.output()).await;
    let _ = std::fs::remove_dir_all(&dir);
    let out = out.map_err(|_| "sandbox timed out".to_owned())?.map_err(|e| format!("podman: {e}"))?;
    if out.status.success() {
        return Ok(());
    }
    let err = String::from_utf8_lossy(&out.stderr);
    let last = err.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("").trim();
    Err(match out.status.code() {
        Some(124 | 143) => format!("timed out after {RUN_TIMEOUT_SECS} s"),
        _ => format!("tests failed: {}", last.chars().take(160).collect::<String>()),
    })
}

fn tempdir_in(base: &Path) -> Result<std::path::PathBuf, String> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let dir = base.join(format!("sandbox-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir_all(&dir).map_err(|e| format!("creating {}: {e}", dir.display()))?;
    Ok(dir)
}

// --- running --------------------------------------------------------------------

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ItemResult {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    pub runs: u32,
    pub passed: u32,
    pub failures: Vec<String>,
    pub mean_s: f64,
    pub completion_tokens: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct QualityReport {
    pub suite: String,
    pub items: Vec<ItemResult>,
    pub pass_rate: f64,
    /// Per subject (knowledge).
    pub categories: BTreeMap<String, f64>,
    pub reps: u32,
    /// Answers cut off by the token limit (a sign of runaway thinking).
    pub truncated: u32,
}

impl QualityReport {
    pub fn metrics(&self) -> BTreeMap<String, f64> {
        let mut m = BTreeMap::from([(format!("{}_pass_rate", self.suite), self.pass_rate)]);
        let tokens: u64 = self.items.iter().map(|i| i.completion_tokens).sum();
        let runs: u32 = self.items.iter().map(|i| i.runs).sum();
        if runs > 0 {
            m.insert(format!("{}_tokens_per_item", self.suite), tokens as f64 / f64::from(runs));
        }
        m
    }
}

/// How many items and repetitions a preset runs.
pub fn plan(preset: crate::Preset) -> (Option<usize>, u32) {
    match preset {
        crate::Preset::Quick => (Some(20), 1),
        crate::Preset::Standard => (None, 1),
        crate::Preset::Full => (None, 3),
    }
}

pub async fn run(
    http: &reqwest::Client,
    target: &Target,
    suite: QualitySuite,
    preset: crate::Preset,
    work_dir: &Path,
    progress: &dyn Progress,
) -> Result<QualityReport, String> {
    if suite == QualitySuite::Coding {
        progress.step("preparing the code sandbox (podman)");
        sandbox_ready().await?;
    }
    let (limit, reps) = plan(preset);
    let mut items = suite.items();
    if let Some(n) = limit {
        items.truncate(n);
    }
    let mut results = Vec::new();
    let mut truncated = 0;
    for (i, item) in items.iter().enumerate() {
        let mut r = ItemResult { id: item.id().into(), category: item.category().map(str::to_owned), ..Default::default() };
        let mut elapsed = 0.0;
        for rep in 0..reps {
            if progress.cancelled() {
                return Err("cancelled".into());
            }
            progress.step(&format!("{} {}/{}: {} (run {}/{reps})", suite.name(), i + 1, items.len(), item.id(), rep + 1));
            let started = Instant::now();
            let outcome = attempt(http, target, item, work_dir, &mut r.completion_tokens, &mut truncated).await;
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
    let rate = |rs: &[&ItemResult]| {
        let (p, n) = rs.iter().fold((0, 0), |a, r| (a.0 + r.passed, a.1 + r.runs));
        if n == 0 { 0.0 } else { f64::from(p) / f64::from(n) }
    };
    let mut by_cat: BTreeMap<String, Vec<&ItemResult>> = BTreeMap::new();
    for r in &results {
        if let Some(c) = &r.category {
            by_cat.entry(c.clone()).or_default().push(r);
        }
    }
    let categories = by_cat.iter().map(|(k, v)| (k.clone(), rate(v))).collect();
    let all: Vec<&ItemResult> = results.iter().collect();
    Ok(QualityReport { suite: suite.name().into(), pass_rate: rate(&all), categories, reps, truncated, items: results })
}

async fn attempt(
    http: &reqwest::Client,
    target: &Target,
    item: &Item,
    work_dir: &Path,
    tokens: &mut u64,
    truncated: &mut u32,
) -> Result<(), String> {
    let body = json!({
        "model": target.model,
        "messages": [{ "role": "user", "content": item.prompt() }],
        "max_tokens": MAX_TOKENS,
    });
    let res = http
        .post(format!("{}/v1/chat/completions", target.base_url))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    let status = res.status();
    let resp: Value = res.json().await.map_err(|e| format!("bad response: {e}"))?;
    if !status.is_success() {
        return Err(format!("HTTP {status}: {}", resp["error"]["message"].as_str().unwrap_or_default()));
    }
    *tokens += resp["usage"]["completion_tokens"].as_u64().unwrap_or(0);
    if resp["choices"][0]["finish_reason"] == "length" {
        *truncated += 1;
        return Err(format!("hit the {MAX_TOKENS}-token limit"));
    }
    let text = resp["choices"][0]["message"]["content"].as_str().unwrap_or_default();
    let tail = || {
        let t: String = text.chars().rev().take(60).collect::<Vec<_>>().into_iter().rev().collect();
        t.trim().replace('\n', " ")
    };
    match item {
        Item::Math { answer, .. } => match extract_number(text) {
            Some(v) if (v - answer).abs() < 1e-6 => Ok(()),
            Some(v) => Err(format!("answered {v}, expected {answer}")),
            None => Err(format!("no number in the answer: …{}", tail())),
        },
        Item::Choice { answer, options, .. } => match extract_letter(text, options.len()) {
            Some(l) if answer.starts_with(l) => Ok(()),
            Some(l) => Err(format!("answered {l}, expected {answer}")),
            None => Err(format!("no \"Answer: <letter>\": …{}", tail())),
        },
        Item::Code { prompt, test, entry_point, .. } => {
            let code = extract_code(text).ok_or("no code block in the reply")?;
            run_sandboxed(&program(prompt, &code, test, entry_point), work_dir).await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subsets_load() {
        assert_eq!(QualitySuite::Reasoning.items().len(), 50);
        let k = QualitySuite::Knowledge.items();
        assert_eq!(k.len(), 56);
        assert!(k.iter().all(|i| matches!(i, Item::Choice { options, answer, .. } if !options.is_empty() && answer.len() == 1)));
        assert_eq!(QualitySuite::Coding.items().len(), 40);
        assert!(QualitySuite::Reasoning.items().iter().all(|i| matches!(i, Item::Math { answer, .. } if answer.is_finite())));
    }

    #[test]
    fn numbers() {
        assert_eq!(extract_number("so 3 + 4 = 7.\nAnswer: 1,250"), Some(1250.0));
        assert_eq!(extract_number("Answer: $18.50."), Some(18.5));
        assert_eq!(extract_number("The total is 42 apples."), Some(42.0));
        assert_eq!(extract_number("Answer: -3"), Some(-3.0));
        assert_eq!(extract_number("**Answer:** 70000"), Some(70000.0));
        assert_eq!(extract_number("no digits"), None);
    }

    #[test]
    fn letters() {
        assert_eq!(extract_letter("... Answer: C", 10), Some('C'));
        assert_eq!(extract_letter("Answer: (b)", 10), Some('B'));
        assert_eq!(extract_letter("**Answer:** **J**", 10), Some('J'));
        assert_eq!(extract_letter("Answer: $\\boxed{D}$", 10), Some('D'));
        assert_eq!(extract_letter("Answer: K", 10), None, "only A–J");
        assert_eq!(extract_letter("I think it is C", 10), None);
    }

    #[test]
    fn code_blocks_and_programs() {
        let reply = "Here:\n```\nnot this\n```\n```python\ndef add(a, b):\n    return a + b\n```\nDone.";
        assert_eq!(extract_code(reply).unwrap(), "def add(a, b):\n    return a + b\n");
        let p = program("from typing import List\n\ndef add(a, b):\n    \"\"\"Add.\"\"\"\n", "    return a + b\n", "def check(f):\n    assert f(1, 2) == 3\n", "add");
        assert!(p.starts_with("from typing import List"));
        assert!(p.contains("\"\"\"Add.\"\"\"\n    return a + b"), "body under the prompt's signature:\n{p}");
        assert!(p.ends_with("check(add)\n"));
    }

    #[test]
    fn plans_grow() {
        assert_eq!(plan(crate::Preset::Quick), (Some(20), 1));
        assert_eq!(plan(crate::Preset::Full), (None, 3));
    }

    /// Needs podman; run with `cargo test -- --ignored`.
    #[tokio::test]
    #[ignore]
    async fn sandbox_isolates() {
        let dir = tempfile::tempdir().unwrap();
        sandbox_ready().await.unwrap();
        assert!(run_sandboxed("assert 1 + 1 == 2\n", dir.path()).await.is_ok());
        let e = run_sandboxed("assert 1 == 2\n", dir.path()).await.unwrap_err();
        assert!(e.contains("AssertionError"), "{e}");
        let net = "import urllib.request\nurllib.request.urlopen('http://1.1.1.1', timeout=3)\n";
        assert!(run_sandboxed(net, dir.path()).await.is_err(), "no network");
        let write = "open('/etc/x', 'w').write('x')\n";
        assert!(run_sandboxed(write, dir.path()).await.is_err(), "read-only root");
        let e = run_sandboxed("while True: pass\n", dir.path()).await.unwrap_err();
        assert!(e.contains("timed out"), "{e}");
    }
}
