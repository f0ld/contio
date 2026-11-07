//! Speed and energy through the OpenAI API (spec §8.1): prefill and decode
//! tok/s at several prompt lengths, time to first token, SoC power and joules
//! per token, and throughput with parallel requests.
//!
//! Every prompt starts with a unique nonce so the backend's prompt cache
//! can't serve it. llama.cpp reports its own timings in the last chunk; they
//! are preferred over client-side timing, which includes HTTP and gateway
//! overhead.

use std::time::Duration;

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::monitor::Monitor;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerfOptions {
    /// Prompt lengths to measure, in tokens.
    pub prompt_tokens: Vec<u32>,
    /// Tokens generated per request.
    pub gen_tokens: u32,
    /// Repetitions per prompt length (the median is reported).
    pub reps: u32,
    /// Parallel requests for the throughput test; 0 skips it.
    pub concurrency: u32,
    /// Agentic workloads (edit / code / prose) and a cached follow-up turn
    /// (docs/PERFORMANCE.md): show speculative decoding and prompt-cache reuse.
    #[serde(default)]
    pub workloads: bool,
}

/// Prompt lengths longer than this (minus the generation) are skipped.
#[derive(Debug, Clone)]
pub struct Target {
    pub base_url: String,
    /// API model name.
    pub model: String,
    pub context: Option<u32>,
    /// llama.cpp: send `ignore_eos` and `cache_prompt: false`, read `timings`.
    pub llama: bool,
    /// Extra fields for quality requests, e.g. `chat_template_kwargs` to turn
    /// thinking off.
    pub extra: serde_json::Map<String, Value>,
}

impl Target {
    /// Add the extra fields to a request body.
    pub fn apply_extra(&self, body: &mut Value) {
        for (k, v) in &self.extra {
            body[k] = v.clone();
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Sample {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub ttft_s: f64,
    pub total_s: f64,
    /// Client-side: prompt tokens / TTFT.
    pub prefill_tps: Option<f64>,
    /// Client-side: (completion tokens − 1) / time after the first token.
    pub decode_tps: Option<f64>,
    /// Reported by the backend (llama.cpp `timings`).
    pub backend_prefill_tps: Option<f64>,
    pub backend_decode_tps: Option<f64>,
    pub avg_watts: Option<f64>,
    pub prefill_joules: Option<f64>,
    pub decode_joules: Option<f64>,
    /// Energy of the whole request / generated tokens.
    pub joules_per_token: Option<f64>,
    pub finish_reason: Option<String>,
    /// Prompt tokens the backend took from its cache.
    #[serde(default)]
    pub cached_tokens: Option<u32>,
}

impl Sample {
    pub fn best_prefill_tps(&self) -> Option<f64> {
        self.backend_prefill_tps.or(self.prefill_tps)
    }

    pub fn best_decode_tps(&self) -> Option<f64> {
        self.backend_decode_tps.or(self.decode_tps)
    }
}

/// Medians over the repetitions of one prompt length.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    pub prompt_tokens: u32,
    pub ttft_s: Option<f64>,
    pub prefill_tps: Option<f64>,
    pub decode_tps: Option<f64>,
    pub avg_watts: Option<f64>,
    pub joules_per_token: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromptResult {
    /// The length asked for; `summary.prompt_tokens` is what the tokenizer made of it.
    pub target_tokens: u32,
    pub summary: Summary,
    pub samples: Vec<Sample>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConcurrencyResult {
    pub requests: u32,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub wall_s: f64,
    /// All generated tokens / wall time.
    pub aggregate_tps: f64,
    /// Median per-request decode speed while sharing the backend.
    pub per_request_decode_tps: Option<f64>,
    pub avg_watts: Option<f64>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PerfReport {
    /// SoC power before the first request, for context.
    pub idle_watts: Option<f64>,
    pub prompts: Vec<PromptResult>,
    /// Prompt lengths that don't fit the model's context.
    pub skipped: Vec<u32>,
    /// Measured prompt sizes: (characters, tokens) of the two calibration prompts.
    pub calibration: Vec<(usize, u32)>,
    pub concurrency: Option<ConcurrencyResult>,
    #[serde(default)]
    pub workloads: Vec<WorkloadResult>,
    #[serde(default)]
    pub followup: Option<FollowupResult>,
}

/// One realistic generation: what speculative decoding changes.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct WorkloadResult {
    /// `edit` (return a file with a small change), `code` (new program), `prose`.
    pub name: String,
    pub completion_tokens: u32,
    pub decode_tps: Option<f64>,
    /// Speculative decoding: tokens drafted and accepted (llama.cpp `timings`).
    pub draft_n: Option<u32>,
    pub draft_accepted: Option<u32>,
}

/// An agent's next turn: the same conversation plus a short message. With a
/// working prompt cache only the new tokens are read.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FollowupResult {
    pub context_tokens: u32,
    pub first_ttft_s: f64,
    pub followup_ttft_s: f64,
    /// Prompt tokens served from the cache in the follow-up.
    pub cached_tokens: Option<u32>,
}

/// Progress reporting and cancellation, implemented by the caller's job.
pub trait Progress: Sync {
    fn step(&self, message: &str);
    fn cancelled(&self) -> bool {
        false
    }
}

/// Run `fut` unless the job is cancelled first. Dropping a request closes its
/// connection, and the backend stops generating.
pub async fn or_cancel<T>(progress: &dyn Progress, fut: impl Future<Output = Result<T, String>>) -> Result<T, String> {
    let watch = async {
        loop {
            tokio::time::sleep(Duration::from_millis(250)).await;
            if progress.cancelled() {
                break;
            }
        }
    };
    tokio::select! {
        r = fut => r,
        () = watch => Err("cancelled".into()),
    }
}

const IDLE_WINDOW: Duration = Duration::from_secs(2);
/// Two calibration prompts (characters): the slope gives characters per
/// token, the intercept the template and instruction around the filler.
const CALIBRATION_CHARS: [usize; 2] = [1024, 4096];
/// Room for the chat template around the prompt.
const TEMPLATE_TOKENS: u32 = 64;

pub async fn run(
    http: &reqwest::Client,
    target: &Target,
    opts: &PerfOptions,
    monitor: &Monitor,
    progress: &dyn Progress,
) -> Result<PerfReport, String> {
    let mut report = PerfReport::default();
    let check = || if progress.cancelled() { Err("cancelled".to_owned()) } else { Ok(()) };

    progress.step("measuring idle power");
    let t0 = monitor.now();
    tokio::time::sleep(IDLE_WINDOW).await;
    report.idle_watts = monitor.avg_watts(t0, monitor.now());

    // warm-up (first-request costs like shader compilation) and calibration
    progress.step("warming up");
    let [c1, c2] = CALIBRATION_CHARS;
    let t1 = or_cancel(progress, request(http, target, &prompt(c1, u32::MAX), 8, monitor)).await?.prompt_tokens;
    let t2 = or_cancel(progress, request(http, target, &prompt(c2, u32::MAX - 1), 8, monitor)).await?.prompt_tokens;
    let cal = Calibration::new((c1, t1), (c2, t2));
    report.calibration = vec![(c1, t1), (c2, t2)];
    check()?;

    let mut nonce = 1;
    for &tokens in &opts.prompt_tokens {
        if target.context.is_some_and(|c| tokens + opts.gen_tokens + TEMPLATE_TOKENS > c) {
            report.skipped.push(tokens);
            continue;
        }
        let mut samples = Vec::new();
        for rep in 0..opts.reps.max(1) {
            check()?;
            progress.step(&format!("prompt {tokens} tokens, run {}/{}", rep + 1, opts.reps.max(1)));
            let text = prompt(cal.chars_for(tokens), nonce);
            nonce += 1;
            samples.push(or_cancel(progress, request(http, target, &text, opts.gen_tokens, monitor)).await?);
        }
        report.prompts.push(PromptResult { target_tokens: tokens, summary: summarize(&samples), samples });
    }

    if opts.workloads {
        for (name, text) in workloads() {
            check()?;
            progress.step(&format!("workload: {name}"));
            let messages = json!([{ "role": "user", "content": text }]);
            let (s, timings) =
                or_cancel(progress, chat(http, target, &messages, WORKLOAD_TOKENS, Mode::Natural, monitor)).await?;
            let n = |k: &str| timings[k].as_u64().map(|x| x as u32);
            report.workloads.push(WorkloadResult {
                name: name.to_owned(),
                completion_tokens: s.completion_tokens,
                decode_tps: s.best_decode_tps(),
                draft_n: n("draft_n"),
                draft_accepted: n("draft_n_accepted"),
            });
        }
        let tokens = FOLLOWUP_CONTEXT.min(target.context.unwrap_or(u32::MAX).saturating_sub(1024));
        if tokens >= 1024 {
            check()?;
            progress.step(&format!("follow-up turn on a {tokens}-token conversation"));
            let first = json!([{ "role": "user", "content": prompt(cal.chars_for(tokens), nonce) }]);
            let (a, _) = or_cancel(progress, chat(http, target, &first, 16, Mode::Cached, monitor)).await?;
            let mut next = first.as_array().cloned().unwrap_or_default();
            next.push(json!({ "role": "assistant", "content": "Understood." }));
            next.push(json!({ "role": "user", "content": "Now give the story a title. Only the title." }));
            let (b, _) = or_cancel(progress, chat(http, target, &Value::Array(next), 16, Mode::Cached, monitor)).await?;
            report.followup = Some(FollowupResult {
                context_tokens: a.prompt_tokens,
                first_ttft_s: a.ttft_s,
                followup_ttft_s: b.ttft_s,
                cached_tokens: b.cached_tokens,
            });
        }
    }

    if opts.concurrency > 1 {
        check()?;
        let n = opts.concurrency;
        progress.step(&format!("{n} parallel requests"));
        let tokens = opts.prompt_tokens.first().copied().unwrap_or(512).min(target.context.unwrap_or(u32::MAX) / 2);
        let texts: Vec<String> =
            (0..n).map(|i| prompt(cal.chars_for(tokens), nonce + i)).collect();
        let start = monitor.now();
        let all = futures_util::future::join_all(texts.iter().map(|t| request(http, target, t, opts.gen_tokens, monitor)));
        let results = or_cancel(progress, async { Ok(all.await) }).await?;
        let end = monitor.now();
        let samples: Vec<Sample> = results.into_iter().collect::<Result<_, _>>()?;
        let completion: u32 = samples.iter().map(|s| s.completion_tokens).sum();
        report.concurrency = Some(ConcurrencyResult {
            requests: n,
            prompt_tokens: samples.first().map_or(0, |s| s.prompt_tokens),
            completion_tokens: completion,
            wall_s: end - start,
            aggregate_tps: f64::from(completion) / (end - start).max(1e-9),
            per_request_decode_tps: median(samples.iter().filter_map(Sample::best_decode_tps).collect()),
            avg_watts: monitor.avg_watts(start, end),
        });
    }
    Ok(report)
}

/// Generated tokens per workload; enough for speculation to show.
const WORKLOAD_TOKENS: u32 = 600;
/// Conversation length for the follow-up turn.
const FOLLOWUP_CONTEXT: u32 = 8192;

/// The edit fixture: a real 140-line source file.
const EDIT_FIXTURE: &str = include_str!("../../../registry/bench/edit_fixture.rs.txt");

fn workloads() -> [(&'static str, String); 3] {
    [
        (
            "edit",
            format!(
                "Rename the function `check` to `check_packages` everywhere in this Rust file and return the \
                 complete updated file, nothing else.\n\n```rust\n{EDIT_FIXTURE}\n```"
            ),
        ),
        (
            "code",
            "Write a complete Python command-line program (argparse) that reads a CSV file and prints per-column \
             summary statistics: count, missing values, min, max, mean for numeric columns, and the top 3 values \
             for text columns. Include type hints and docstrings. Only the code."
                .into(),
        ),
        (
            "prose",
            "Explain to a new developer how a hash map works internally: hashing, buckets, collisions, load \
             factor and resizing. About 400 words, plain prose, no code."
                .into(),
        ),
    ]
}

/// How a request treats the backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Exactly `max_tokens`, never from the prompt cache (speed sweeps).
    Exact,
    /// Stop when the model is done, no prompt cache (workloads).
    Natural,
    /// Like a client: the prompt cache may serve the start of the conversation.
    Cached,
}

/// One streamed chat request, timed and metered.
pub async fn request(
    http: &reqwest::Client,
    target: &Target,
    prompt: &str,
    max_tokens: u32,
    monitor: &Monitor,
) -> Result<Sample, String> {
    let messages = json!([{ "role": "user", "content": prompt }]);
    chat(http, target, &messages, max_tokens, Mode::Exact, monitor).await.map(|(s, _)| s)
}

/// A streamed chat request; also returns llama.cpp's `timings`.
async fn chat(
    http: &reqwest::Client,
    target: &Target,
    messages: &Value,
    max_tokens: u32,
    mode: Mode,
    monitor: &Monitor,
) -> Result<(Sample, Value), String> {
    let mut body = json!({
        "model": target.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": true,
        "stream_options": { "include_usage": true },
    });
    if mode != Mode::Exact {
        target.apply_extra(&mut body); // e.g. thinking off
    }
    if target.llama {
        if mode == Mode::Exact {
            body["ignore_eos"] = json!(true);
        }
        body["cache_prompt"] = json!(mode == Mode::Cached);
    }
    let start = monitor.now();
    let res = http
        .post(format!("{}/v1/chat/completions", target.base_url))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    if !res.status().is_success() {
        let status = res.status();
        let text = res.text().await.unwrap_or_default();
        let msg = serde_json::from_str::<Value>(&text)
            .ok()
            .and_then(|v| v["error"]["message"].as_str().map(str::to_owned))
            .unwrap_or(text);
        return Err(format!("HTTP {status}: {msg}"));
    }

    let mut first: Option<f64> = None;
    let mut deltas = 0u32;
    let mut usage = Value::Null;
    let mut timings = Value::Null;
    let mut finish = None;
    // bytes, not text: a chunk may end inside a multi-byte character
    let mut buf: Vec<u8> = Vec::new();
    let mut stream = res.bytes_stream();
    while let Some(chunk) = stream.next().await {
        buf.extend_from_slice(&chunk.map_err(|e| format!("stream: {e}"))?);
        while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
            let raw: Vec<u8> = buf.drain(..=nl).collect();
            let line = String::from_utf8_lossy(&raw);
            let Some(data) = line.trim().strip_prefix("data:").map(str::trim) else { continue };
            if data == "[DONE]" {
                continue;
            }
            let Ok(v) = serde_json::from_str::<Value>(data) else { continue };
            if v["usage"].is_object() {
                usage = v["usage"].clone();
            }
            if v["timings"].is_object() {
                timings = v["timings"].clone();
            }
            let choice = &v["choices"][0];
            if let Some(r) = choice["finish_reason"].as_str() {
                finish = Some(r.to_owned());
            }
            let d = &choice["delta"];
            let produced = ["content", "reasoning_content", "reasoning"]
                .iter()
                .any(|k| d[k].as_str().is_some_and(|s| !s.is_empty()))
                || d["tool_calls"].is_array();
            if produced {
                first.get_or_insert_with(|| monitor.now());
                deltas += 1;
            }
        }
    }
    let end = monitor.now();
    let first = first.ok_or("the model produced no tokens")?;
    let n = |v: &Value| v.as_u64().map(|x| x as u32);
    let prompt_tokens = n(&usage["prompt_tokens"]).or(n(&timings["prompt_n"])).unwrap_or(0);
    let completion = n(&usage["completion_tokens"]).or(n(&timings["predicted_n"])).unwrap_or(deltas);
    let ttft = first - start;
    let decode_s = end - first;
    let prefill_j = monitor.energy(start, first);
    let decode_j = monitor.energy(first, end);
    let sample = Sample {
        prompt_tokens,
        completion_tokens: completion,
        ttft_s: ttft,
        total_s: end - start,
        prefill_tps: (prompt_tokens > 0 && ttft > 0.0).then(|| f64::from(prompt_tokens) / ttft),
        decode_tps: (completion > 1 && decode_s > 0.0).then(|| f64::from(completion - 1) / decode_s),
        backend_prefill_tps: timings["prompt_per_second"].as_f64(),
        backend_decode_tps: timings["predicted_per_second"].as_f64(),
        avg_watts: monitor.avg_watts(start, end),
        prefill_joules: prefill_j,
        decode_joules: decode_j,
        joules_per_token: monitor.energy(start, end).filter(|_| completion > 0).map(|j| j / f64::from(completion)),
        finish_reason: finish,
        cached_tokens: n(&usage["prompt_tokens_details"]["cached_tokens"]).or(n(&timings["cache_n"])),
    };
    Ok((sample, timings))
}

/// Prompt characters → tokens, as a line through two measurements.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Calibration {
    chars_per_token: f64,
    /// Tokens that aren't filler (template, instruction).
    overhead: f64,
}

impl Calibration {
    fn new((c1, t1): (usize, u32), (c2, t2): (usize, u32)) -> Self {
        let dt = f64::from(t2) - f64::from(t1);
        if dt <= 0.0 {
            // backend didn't report usage: assume ~4 characters per token
            return Self { chars_per_token: 4.0, overhead: 0.0 };
        }
        let chars_per_token = (c2 - c1) as f64 / dt;
        Self { chars_per_token, overhead: (f64::from(t1) - c1 as f64 / chars_per_token).max(0.0) }
    }

    fn chars_for(&self, tokens: u32) -> usize {
        ((f64::from(tokens) - self.overhead).max(1.0) * self.chars_per_token) as usize
    }
}

pub fn summarize(samples: &[Sample]) -> Summary {
    let m = |f: &dyn Fn(&Sample) -> Option<f64>| median(samples.iter().filter_map(f).collect());
    Summary {
        prompt_tokens: samples.first().map_or(0, |s| s.prompt_tokens),
        ttft_s: m(&|s| Some(s.ttft_s)),
        prefill_tps: m(&Sample::best_prefill_tps),
        decode_tps: m(&Sample::best_decode_tps),
        avg_watts: m(&|s| s.avg_watts),
        joules_per_token: m(&|s| s.joules_per_token),
    }
}

pub fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    let mid = v.len() / 2;
    Some(if v.len() % 2 == 1 { v[mid] } else { (v[mid - 1] + v[mid]) / 2.0 })
}

const FILLER: [&str; 8] = [
    "The harbour office logged {n} arrivals before noon, most of them small fishing boats returning early.",
    "In the archive, folder {n} held maps of the old railway line, annotated in three different inks.",
    "A survey team measured the river at point {n} and found the current slower than the season before.",
    "Bakery orders for district {n} doubled after the market moved to the square by the town hall.",
    "The observatory's night log, entry {n}, notes thin clouds and a bright pass of the space station.",
    "Workshop notes, page {n}: the gearbox was rebuilt with new bearings and tested for an hour.",
    "The library catalogued {n} donated books, mostly travel guides and a few rare atlases.",
    "Weather station {n} recorded gusts from the west and a sharp drop in pressure overnight.",
];

/// About `chars` characters of varied filler, a unique first line (defeats
/// prompt caches), and an instruction that keeps the model writing.
pub fn prompt(chars: usize, nonce: u32) -> String {
    let mut s = format!("Session {nonce:08x}. Background notes:\n");
    let mut i = 0usize;
    while s.len() < chars {
        // small numbers whatever the nonce: digits tokenize one by one, and the
        // calibration prompts must look like the measured ones
        s.push_str(&FILLER[i % FILLER.len()].replace("{n}", &((i * 7 + nonce as usize) % 1000).to_string()));
        s.push(' ');
        i += 1;
    }
    s.push_str("\n\nIgnore the notes. Write a long, detailed story about a lighthouse keeper. Keep writing.");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompts_are_unique_and_sized() {
        let a = prompt(4000, 1);
        let b = prompt(4000, 2);
        assert_ne!(a[..40], b[..40], "different nonce in the first line");
        assert!(a.len() >= 4000 && a.len() < 4300, "{}", a.len());
    }

    #[test]
    fn calibration_removes_the_template_overhead() {
        // 4 chars per token plus 40 tokens of template
        let c = Calibration::new((1000, 290), (4000, 1040));
        assert!((c.chars_per_token - 4.0).abs() < 1e-9 && (c.overhead - 40.0).abs() < 1e-9);
        assert_eq!(c.chars_for(512), (512 - 40) * 4);
        assert_eq!(Calibration::new((1000, 0), (4000, 0)).chars_per_token, 4.0, "no usage reported");
    }

    #[test]
    fn medians() {
        assert_eq!(median(vec![]), None);
        assert_eq!(median(vec![3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(vec![4.0, 1.0, 2.0, 3.0]), Some(2.5));
    }

    /// Streams like llama-server: tokens, a usage chunk, `timings` at the end.
    async fn mock_server() -> String {
        use axum::routing::post;
        let app = axum::Router::new().route(
            "/v1/chat/completions",
            post(|axum::Json(req): axum::Json<Value>| async move {
                let max = req["max_tokens"].as_u64().unwrap();
                let prompt_len = req["messages"][0]["content"].as_str().unwrap().len() as u64;
                // speed sweeps decode exactly max_tokens; workloads and follow-ups stop naturally
                let natural = max == 600 || req["cache_prompt"] == true;
                assert_eq!(req["ignore_eos"] == true, !natural);
                let followup = req["messages"].as_array().unwrap().len() == 3;
                let mut body = String::new();
                for _ in 0..max {
                    body.push_str("data: {\"choices\":[{\"delta\":{\"content\":\"ü\"}}]}\n\n");
                }
                let cached = if followup { prompt_len / 4 } else { 0 };
                let usage = json!({"choices": [], "usage": {"prompt_tokens": prompt_len / 4, "completion_tokens": max,
                        "prompt_tokens_details": {"cached_tokens": cached}},
                    "timings": {"prompt_per_second": 500.0, "predicted_per_second": 25.0, "draft_n": 10, "draft_n_accepted": 8}});
                body.push_str(&format!("data: {usage}\n\ndata: [DONE]\n\n"));
                tokio::time::sleep(Duration::from_millis(20)).await;
                ([("content-type", "text/event-stream")], body)
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        url
    }

    struct Quiet;
    impl Progress for Quiet {
        fn step(&self, _: &str) {}
    }

    #[tokio::test]
    async fn runs_against_a_streaming_server() {
        let target = Target { base_url: mock_server().await, model: "m".into(), context: Some(2048), llama: true, extra: Default::default() };
        let opts =
            PerfOptions { prompt_tokens: vec![256, 4096], gen_tokens: 16, reps: 2, concurrency: 3, workloads: true };
        let monitor = Monitor::start(None);
        let r = run(&reqwest::Client::new(), &target, &opts, &monitor, &Quiet).await.unwrap();
        assert_eq!(r.skipped, [4096], "longer than the context");
        let p = &r.prompts[0];
        assert_eq!(p.samples.len(), 2);
        // calibrated: ~256 tokens asked for, the mock counts 4 chars per token
        assert!((230..=290).contains(&p.summary.prompt_tokens), "{}", p.summary.prompt_tokens);
        assert_eq!(p.samples[0].completion_tokens, 16);
        assert_eq!(p.summary.prefill_tps, Some(500.0), "backend timings preferred");
        assert!(p.samples[0].decode_tps.is_some() && p.samples[0].ttft_s > 0.0);
        let c = r.concurrency.unwrap();
        assert_eq!((c.requests, c.completion_tokens), (3, 48));
        let names: Vec<&str> = r.workloads.iter().map(|w| w.name.as_str()).collect();
        assert_eq!(names, ["edit", "code", "prose"]);
        assert_eq!((r.workloads[0].draft_n, r.workloads[0].draft_accepted), (Some(10), Some(8)));
        let f = r.followup.expect("follow-up turn");
        assert!(f.context_tokens > 900 && f.cached_tokens.is_some_and(|c| c > 900), "{f:?}");
        assert!(r.idle_watts.is_none(), "no sensor without a probe");
    }

    #[test]
    fn backend_timings_win() {
        let s = Sample { prefill_tps: Some(100.0), backend_prefill_tps: Some(120.0), decode_tps: Some(20.0), ..Default::default() };
        assert_eq!(s.best_prefill_tps(), Some(120.0));
        assert_eq!(s.best_decode_tps(), Some(20.0));
    }
}
