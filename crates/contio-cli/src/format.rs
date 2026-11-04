//! Human-readable output.

use std::fmt::Write;

use contio_core::hw::{HardwareInfo, MemoryPool, MemoryPoolKind, PowerProfile};
use contio_core::model::{ModelKind, ModelState, ModelStatus};
use serde_json::Value;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / GIB)
}

fn pool_name(k: MemoryPoolKind) -> &'static str {
    match k {
        MemoryPoolKind::Gpu => "gpu",
        MemoryPoolKind::System => "system",
    }
}

fn state(s: &ModelState) -> String {
    match s {
        ModelState::Stopped => "stopped".into(),
        ModelState::Loading => "loading".into(),
        ModelState::Ready { port } => format!("ready :{port}"),
        ModelState::Failed { error } => {
            let first = error.lines().next().unwrap_or_default();
            format!("failed: {}", first.chars().take(60).collect::<String>())
        }
    }
}

pub fn models(models: &[ModelStatus]) -> String {
    if models.is_empty() {
        return "no models installed yet: try `contio search` and `contio install <model>`".into();
    }
    let rows: Vec<[String; 7]> = models
        .iter()
        .map(|m| {
            let mut device = format!("{:?}", m.device).to_lowercase();
            if m.kind != ModelKind::Chat {
                device = format!("{device} ({})", format!("{:?}", m.kind).to_lowercase());
            }
            let mut notes = Vec::new();
            if m.pinned {
                notes.push("pinned");
            }
            if !m.managed {
                notes.push("config");
            }
            [
                m.id.clone(),
                device,
                m.backend.clone().unwrap_or_else(|| "-".into()),
                m.context.map_or_else(|| "-".into(), |c| format!("{}k", c / 1024)),
                m.estimated_bytes.map_or_else(|| "-".into(), gib),
                if m.in_flight > 0 { format!("{} ({} active)", state(&m.state), m.in_flight) } else { state(&m.state) },
                notes.join(", "),
            ]
        })
        .collect();
    table(&["MODEL", "DEVICE", "BACKEND", "CTX", "EST. MEM", "STATE", ""], &rows)
}

/// `contio search` filters, applied client-side to `/api/catalog`.
pub struct CatalogFilter {
    pub query: String,
    pub device: Option<String>,
    pub fits: bool,
    pub kind: Option<String>,
}

fn fit_mark(fit: &Value) -> &'static str {
    match fit["verdict"].as_str() {
        Some("fits") => "✓",
        Some("tight") => "~",
        Some("too_big") => "✗",
        _ => "?",
    }
}

fn params(v: &Value) -> String {
    match (v["params"].as_u64(), v["active_b"].as_f64()) {
        (Some(p), Some(a)) => format!("{:.0}B-A{a}B", p as f64 / 1e9),
        (Some(p), None) if p < 1_000_000_000 => format!("{:.0}M", p as f64 / 1e6),
        (Some(p), None) => format!("{:.1}B", p as f64 / 1e9).replace(".0B", "B"),
        _ => "-".into(),
    }
}

pub fn strings(v: &Value) -> Vec<&str> {
    v.as_array().into_iter().flatten().filter_map(Value::as_str).collect()
}

pub fn catalog(raw: &Value, f: &CatalogFilter) -> String {
    let words: Vec<String> = f.query.to_lowercase().split_whitespace().map(str::to_owned).collect();
    let keep = |m: &Value| {
        let hay = format!("{} {} {}", m["id"], m["name"], m["family"]).to_lowercase();
        let device_ok = match f.device.as_deref() {
            Some("npu") => !strings(&m["npu"]).is_empty(),
            Some(d) => !m["suggested"].is_null() && m["suggested"]["fit"][d].is_object(),
            None => true,
        };
        let fit_device = f.device.as_deref().filter(|d| *d != "npu").unwrap_or("gpu");
        let fits_ok = !f.fits || m["suggested"]["fit"][fit_device]["verdict"].as_str().is_some_and(|v| v != "too_big");
        words.iter().all(|w| hay.contains(w.as_str()))
            && device_ok
            && fits_ok
            && f.kind.as_deref().is_none_or(|k| m["kind"] == k)
    };
    let mut out = Vec::new();
    for fam in raw["families"].as_array().into_iter().flatten() {
        let rows: Vec<[String; 7]> = raw["models"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|m| m["family"] == fam["id"] && keep(m))
            .map(|m| {
                let s = &m["suggested"];
                let quant = if s.is_null() {
                    m["error"].as_str().map_or("-".into(), |e| format!("({})", e.chars().take(30).collect::<String>()))
                } else {
                    format!("{} {}", s["label"].as_str().unwrap_or("?"), gib(s["size"].as_u64().unwrap_or(0)))
                };
                let fit = if s.is_null() {
                    "-".into()
                } else {
                    format!("gpu {} cpu {}", fit_mark(&s["fit"]["gpu"]), fit_mark(&s["fit"]["cpu"]))
                };
                let mut caps = strings(&m["capabilities"]).join(",");
                if !strings(&m["npu"]).is_empty() {
                    caps = if caps.is_empty() { "npu".into() } else { format!("{caps},npu") };
                }
                if m["kind"] != "chat" {
                    caps = format!("{}{}{caps}", m["kind"].as_str().unwrap_or_default(), if caps.is_empty() { "" } else { "," });
                }
                [
                    m["id"].as_str().unwrap_or_default().to_owned(),
                    params(m),
                    quant,
                    fit,
                    m["quant_count"].as_u64().filter(|n| *n > 0).map_or("-".into(), |n| n.to_string()),
                    caps,
                    if strings(&m["installed"]).is_empty() { String::new() } else { "installed".into() },
                ]
            })
            .collect();
        if rows.is_empty() {
            continue;
        }
        out.push(format!(
            "\x1b[1m{}\x1b[0m \x1b[2m{} · {}\x1b[0m",
            fam["name"].as_str().unwrap_or_default(),
            fam["vendor"].as_str().unwrap_or_default(),
            fam["license"].as_str().unwrap_or_default()
        ));
        out.push(indent(&table(&["MODEL", "PARAMS", "SUGGESTED", "FIT", "QUANTS", "CAPABILITIES", ""], &rows)));
    }
    if out.is_empty() {
        return "no matches (try `contio search --hf <words>` for all of Hugging Face)".into();
    }
    out.push("\x1b[2mfit: ✓ fits  ~ tight  ✗ too big (at the default context) · details: contio show <model>\x1b[0m".into());
    out.join("\n")
}

fn indent(s: &str) -> String {
    s.lines().map(|l| format!("  {l}")).collect::<Vec<_>>().join("\n")
}

fn num(v: Option<f64>, digits: usize) -> String {
    v.map_or("-".into(), |x| format!("{x:.digits$}"))
}

/// Memory a run used: VRAM growth for GPU models; otherwise the larger of
/// system-pool growth and the backend's resident memory (mapped weights count
/// as cache in the pool, but are resident).
fn mem_used(r: &Value) -> Option<f64> {
    let m = &r["metrics"];
    let f = |k: &str| m[k].as_f64().unwrap_or(0.0);
    let v = if r["target"]["device"] == "gpu" { f("mem_gpu_bytes") } else { f("mem_system_bytes").max(f("rss_bytes")) };
    (v > 0.0).then_some(v)
}

/// Perf and tool-calling runs, one table each.
pub fn bench_runs(runs: &[Value]) -> String {
    if runs.is_empty() {
        return "no benchmark runs yet: try `contio bench run <model>`".into();
    }
    let of = |s: &str| -> Vec<&Value> { runs.iter().filter(|r| r["suite"] == s).collect() };
    let quality: Vec<&Value> =
        runs.iter().filter(|r| matches!(r["suite"].as_str(), Some("reasoning" | "knowledge" | "coding" | "custom"))).collect();
    let mut parts = Vec::new();
    if !of("perf").is_empty() {
        parts.push(perf_runs(&of("perf")));
    }
    if !of("tools").is_empty() {
        parts.push(tool_runs(&of("tools")));
    }
    if !quality.is_empty() {
        parts.push(quality_runs(&quality));
    }
    if !of("speech").is_empty() {
        let rows: Vec<[String; 5]> = of("speech")
            .iter()
            .map(|r| {
                let m = &r["metrics"];
                [
                    r["id"].to_string(),
                    r["model"].as_str().unwrap_or_default().to_owned(),
                    device_quant(r),
                    m["speech_wer"].as_f64().map_or(status_note(r), |v| format!("{:.1}%", v * 100.0)),
                    num(m["speech_rtf"].as_f64(), 3),
                ]
            })
            .collect();
        parts.push(table(&["ID", "MODEL", "DEVICE", "WER", "REAL-TIME FACTOR"], &rows));
    }
    if !of("retrieval").is_empty() {
        let rows: Vec<[String; 6]> = of("retrieval")
            .iter()
            .map(|r| {
                let m = &r["metrics"];
                [
                    r["id"].to_string(),
                    r["model"].as_str().unwrap_or_default().to_owned(),
                    device_quant(r),
                    m["retrieval_recall1"].as_f64().map_or(status_note(r), |v| format!("{:.0}%", v * 100.0)),
                    num(m["retrieval_mrr"].as_f64(), 3),
                    num(m["retrieval_s"].as_f64(), 1),
                ]
            })
            .collect();
        parts.push(table(&["ID", "MODEL", "DEVICE", "RECALL@1", "MRR", "SEC"], &rows));
    }
    parts.join("\n\n")
}

fn quality_runs(runs: &[&Value]) -> String {
    let rows: Vec<[String; 7]> = runs
        .iter()
        .map(|r| {
            let suite = r["suite"].as_str().unwrap_or_default();
            let m = &r["metrics"];
            [
                r["id"].to_string(),
                r["model"].as_str().unwrap_or_default().to_owned(),
                device_quant(r),
                r["preset"].as_str().unwrap_or_default().to_owned(),
                suite.to_owned(),
                m[format!("{suite}_pass_rate")].as_f64().map_or(status_note(r), |v| format!("{:.0}%", v * 100.0)),
                num(m[format!("{suite}_tokens_per_item")].as_f64(), 0),
            ]
        })
        .collect();
    table(&["ID", "MODEL", "DEVICE", "PRESET", "SUITE", "PASS", "TOKENS/ITEM"], &rows)
}

fn device_quant(r: &Value) -> String {
    let no_think = if r["target"]["thinking"] == false { " no-think" } else { "" };
    format!(
        "{}{no_think}",
        device_quant_only(r)
    )
}

fn device_quant_only(r: &Value) -> String {
    format!(
        "{} {}",
        r["target"]["device"].as_str().unwrap_or("?"),
        r["target"]["origin"]["quant"].as_str().filter(|q| *q != "none").unwrap_or("")
    )
    .trim()
    .to_owned()
}

fn status_note(r: &Value) -> String {
    match r["status"].as_str() {
        Some("done") => String::new(),
        Some(s) => s.to_owned(),
        None => "?".into(),
    }
}

fn tool_runs(runs: &[&Value]) -> String {
    let pct = |v: Option<f64>| v.map_or("-".into(), |x| format!("{:.0}%", x * 100.0));
    let rows: Vec<[String; 12]> = runs
        .iter()
        .map(|r| {
            let m = &r["metrics"];
            let f = |k: &str| m[k].as_f64();
            let ready = match f("agent_ready") {
                Some(1.0) => "\x1b[32magent-ready\x1b[0m".to_owned(),
                Some(_) => "no".to_owned(),
                None => status_note(r),
            };
            [
                r["id"].to_string(),
                r["model"].as_str().unwrap_or_default().to_owned(),
                device_quant(r),
                r["preset"].as_str().unwrap_or_default().to_owned(),
                pct(f("tools_pass_rate")),
                pct(f("tools_simple")),
                pct(f("tools_multiple")),
                pct(f("tools_parallel")),
                pct(f("tools_multi_turn")),
                pct(f("tools_irrelevance")),
                pct(f("tools_json")),
                ready,
            ]
        })
        .collect();
    table(
        &["ID", "MODEL", "DEVICE", "PRESET", "PASS", "SIMPLE", "CHOOSE", "PARALLEL", "MULTI-TURN", "NO-CALL", "JSON", "AGENT"],
        &rows,
    )
}

fn perf_runs(runs: &[&Value]) -> String {
    let rows: Vec<[String; 12]> = runs
        .iter()
        .map(|r| {
            let m = &r["metrics"];
            let f = |k: &str| m[k].as_f64();
            let status = status_note(r);
            [
                r["id"].to_string(),
                r["model"].as_str().unwrap_or_default().to_owned(),
                device_quant(r),
                r["preset"].as_str().unwrap_or_default().to_owned(),
                num(f("tg_tps"), 1),
                num(f("pp512_tps"), 0),
                num(f("pp4096_tps"), 0),
                num(f("ttft512_s"), 2),
                num(f("j_per_token"), 2),
                num(f("load_s"), 1),
                mem_used(r).map_or("-".into(), |b| gib(b as u64)),
                [r["conditions"]["power_profile"].as_str().unwrap_or("?"), &status]
                    .iter()
                    .filter(|s| !s.is_empty())
                    .copied()
                    .collect::<Vec<_>>()
                    .join(" "),
            ]
        })
        .collect();
    let mut out = table(
        &["ID", "MODEL", "DEVICE", "PRESET", "DECODE", "PP512", "PP4K", "TTFT", "J/TOK", "LOAD s", "MEM", "PROFILE"],
        &rows,
    );
    out.push_str("\n\x1b[2mtok/s · TTFT at 512 tokens · J/TOK = SoC energy per generated token · MEM = VRAM growth (GPU) or resident memory\x1b[0m");
    out
}

pub fn bench_detail(r: &Value) -> String {
    let mut s = String::new();
    let t = &r["target"];
    let c = &r["conditions"];
    let _ = writeln!(
        s,
        "\x1b[1mrun {} · {}\x1b[0m  {} preset · {}",
        r["id"],
        r["model"].as_str().unwrap_or_default(),
        r["preset"].as_str().unwrap_or_default(),
        r["status"].as_str().unwrap_or_default()
    );
    let quant = t["origin"]["quant"].as_str().unwrap_or("-");
    let _ = writeln!(s, "{} · {} · {quant} · context {}", t["device"].as_str().unwrap_or("?"), t["backend"].as_str().unwrap_or("?"), t["context"]);
    let on_ac = match c["on_ac"].as_bool() {
        Some(true) => "AC".to_owned(),
        Some(false) => format!("battery {}%", c["battery_percent"]),
        None => "power source unknown".into(),
    };
    let _ = writeln!(s, "\x1b[2m{} · {on_ac} · {}\x1b[0m", c["power_profile"].as_str().unwrap_or("?"), c["backend"].as_str().unwrap_or_default());
    if let Some(o) = c["other_models"].as_array().filter(|o| !o.is_empty()) {
        let names: Vec<&str> = o.iter().filter_map(Value::as_str).collect();
        let _ = writeln!(s, "\x1b[33malso loaded: {}\x1b[0m", names.join(", "));
    }
    if let Some(e) = r["error"].as_str() {
        let _ = writeln!(s, "\x1b[31m{e}\x1b[0m");
    }
    let rep = &r["report"];
    if r["suite"] == "tools" {
        return tool_detail(s, rep);
    }
    if rep["items"].is_array() {
        return quality_detail(s, rep);
    }
    if rep["utterances"].is_array() {
        let _ = writeln!(
            s,
            "\x1b[1mWER {:.1}%\x1b[0m · real-time factor {}",
            rep["wer"].as_f64().unwrap_or(0.0) * 100.0,
            num(rep["rtf"].as_f64(), 3)
        );
        for u in rep["utterances"].as_array().into_iter().flatten().filter(|u| u["errors"] != 0) {
            let _ = writeln!(s, "\n{} ({} errors)\n  ref: {}\n  got: {}", u["id"].as_str().unwrap_or_default(), u["errors"],
                u["reference"].as_str().unwrap_or_default().to_lowercase(), u["hypothesis"].as_str().unwrap_or_default());
        }
        return s.trim_end().to_owned();
    }
    if rep["queries"].is_array() {
        let _ = writeln!(
            s,
            "\x1b[1mrecall@1 {:.0}%\x1b[0m · MRR {} · {} s",
            rep["recall_at_1"].as_f64().unwrap_or(0.0) * 100.0,
            num(rep["mrr"].as_f64(), 3),
            num(rep["seconds"].as_f64(), 1)
        );
        let rows: Vec<[String; 3]> = rep["queries"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|q| q["rank"] != 1)
            .map(|q| [q["query"].as_str().unwrap_or_default().to_owned(), q["rank"].to_string(), q["top"].as_str().unwrap_or_default().to_owned()])
            .collect();
        if !rows.is_empty() {
            let _ = writeln!(s, "\nmissed:\n{}", table(&["QUERY", "RANK", "TOP DOC"], &rows));
        }
        return s.trim_end().to_owned();
    }
    let load = &rep["load"];
    let mut facts = Vec::new();
    if let Some(l) = load["load_s"].as_f64() {
        facts.push(format!("load {l:.1} s"));
    }
    if let Some(d) = mem_used(r) {
        let est = load["estimated_bytes"].as_u64().map(|e| format!(" (estimated {})", gib(e))).unwrap_or_default();
        facts.push(format!("memory {}{est}", gib(d as u64)));
    }
    if let Some(w) = rep["perf"]["idle_watts"].as_f64() {
        facts.push(format!("idle {w:.1} W"));
    }
    if !facts.is_empty() {
        let _ = writeln!(s, "{}", facts.join(" · "));
    }
    let rows: Vec<[String; 7]> = rep["perf"]["prompts"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|p| {
            let m = &p["summary"];
            [
                m["prompt_tokens"].to_string(),
                num(m["ttft_s"].as_f64(), 2),
                num(m["prefill_tps"].as_f64(), 0),
                num(m["decode_tps"].as_f64(), 1),
                num(m["avg_watts"].as_f64(), 1),
                num(m["joules_per_token"].as_f64(), 2),
                p["samples"].as_array().map_or(0, Vec::len).to_string(),
            ]
        })
        .collect();
    if !rows.is_empty() {
        let _ = writeln!(s, "\n{}", table(&["PROMPT", "TTFT s", "PREFILL", "DECODE", "WATTS", "J/TOK", "RUNS"], &rows));
    }
    if let Some(sk) = rep["perf"]["skipped"].as_array().filter(|a| !a.is_empty()) {
        let _ = writeln!(s, "\x1b[2mskipped (longer than the context): {}\x1b[0m", sk.iter().map(Value::to_string).collect::<Vec<_>>().join(", "));
    }
    let cc = &rep["perf"]["concurrency"];
    if cc.is_object() {
        let _ = writeln!(
            s,
            "{} parallel requests: {} tok/s total, {} tok/s per request while generating",
            cc["requests"],
            num(cc["aggregate_tps"].as_f64(), 1),
            num(cc["per_request_decode_tps"].as_f64(), 1)
        );
    }
    s.trim_end().to_owned()
}

fn quality_detail(mut s: String, rep: &Value) -> String {
    let rate = rep["pass_rate"].as_f64().unwrap_or(0.0);
    let cats: Vec<String> = rep["categories"]
        .as_object()
        .into_iter()
        .flatten()
        .map(|(k, v)| format!("{k} {:.0}%", v.as_f64().unwrap_or(0.0) * 100.0))
        .collect();
    let _ = writeln!(s, "\x1b[1m{:.0}% passed\x1b[0m · {} runs per item{}", rate * 100.0, rep["reps"],
        if rep["truncated"].as_u64().unwrap_or(0) > 0 { format!(" · {} hit the token limit", rep["truncated"]) } else { String::new() });
    if !cats.is_empty() {
        let _ = writeln!(s, "\x1b[2m{}\x1b[0m", cats.join(" · "));
    }
    let rows: Vec<[String; 4]> = rep["items"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|i| i["passed"] != i["runs"])
        .map(|i| {
            let why = i["failures"][0].as_str().unwrap_or_default();
            [
                i["id"].as_str().unwrap_or_default().to_owned(),
                format!("{}/{}", i["passed"], i["runs"]),
                num(i["mean_s"].as_f64(), 1),
                why.chars().take(90).collect(),
            ]
        })
        .collect();
    if !rows.is_empty() {
        let _ = writeln!(s, "\nfailed:\n{}", table(&["ITEM", "PASS", "SEC", "FIRST FAILURE"], &rows));
    }
    s.trim_end().to_owned()
}

fn tool_detail(mut s: String, rep: &Value) -> String {
    if let Some(rate) = rep["pass_rate"].as_f64() {
        let ready = if rep["agent_ready"] == true { "agent-ready" } else { "not agent-ready" };
        let cats: Vec<String> = rep["categories"]
            .as_object()
            .into_iter()
            .flatten()
            .map(|(k, v)| format!("{k} {:.0}%", v.as_f64().unwrap_or(0.0) * 100.0))
            .collect();
        let _ = writeln!(s, "\x1b[1m{:.0}% passed\x1b[0m · {ready} · {}", rate * 100.0, cats.join(" · "));
    }
    let rows: Vec<[String; 5]> = rep["cases"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|c| {
            let fails: Vec<&str> = c["failures"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
            let mut why = fails.first().copied().unwrap_or_default().to_owned();
            if why.chars().count() > 90 {
                why = format!("{}…", why.chars().take(90).collect::<String>());
            }
            [
                c["id"].as_str().unwrap_or_default().to_owned(),
                format!("{}/{}", c["passed"], c["runs"]),
                num(c["mean_s"].as_f64(), 1),
                c["completion_tokens"].to_string(),
                why,
            ]
        })
        .collect();
    let _ = writeln!(s, "\n{}", table(&["CASE", "PASS", "SEC", "TOKENS", "FIRST FAILURE"], &rows));
    s.trim_end().to_owned()
}

pub fn aliases(v: &Value) -> String {
    let rows: Vec<[String; 3]> = v
        .as_array()
        .into_iter()
        .flatten()
        .map(|a| {
            let mut target = a["model"].as_str().unwrap_or_default().to_owned();
            if let Some(u) = a["upstream_model"].as_str() {
                target.push_str(&format!(" (as {u})"));
            }
            let from = if a["managed"] == true { "" } else { "config.toml" };
            [a["name"].as_str().unwrap_or_default().to_owned(), target, from.to_owned()]
        })
        .collect();
    if rows.is_empty() {
        return "no aliases (add one with `contio alias set <name> <model>`)".into();
    }
    table(&["ALIAS", "MODEL", ""], &rows)
}

pub fn detail(d: &Value) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "\x1b[1m{}\x1b[0m  {}", d["name"].as_str().unwrap_or_default(), d["repo"].as_str().unwrap_or_default());
    let speech = d["kind"] == "transcription";
    let mut facts = vec![params(d), d["kind"].as_str().unwrap_or_default().to_owned()];
    if !speech {
        facts.push(format!(
            "context {} (default {})",
            d["context_length"].as_u64().map_or("?".into(), |c| format!("{}k", c / 1024)),
            d["default_context"].as_u64().map_or("?".into(), |c| format!("{}k", c / 1024)),
        ));
    }
    let caps = strings(&d["capabilities"]).join(", ");
    if !caps.is_empty() {
        facts.push(caps);
    }
    let _ = writeln!(s, "{}", facts.join(" · "));
    if let Some(n) = d["notes"].as_str() {
        let _ = writeln!(s, "\x1b[2m{n}\x1b[0m");
    }
    if let Some(e) = d["error"].as_str() {
        let _ = writeln!(s, "\x1b[33mlisting failed: {e}\x1b[0m");
    }
    let rows: Vec<[String; 7]> = d["quants"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|q| {
            let mut label = q["label"].as_str().unwrap_or_default().to_owned();
            if q["recommended"] == true {
                label.push_str(" *");
            }
            let fit = |dev: &str| {
                let f = &q["fit"][dev];
                match f["needed_bytes"].as_u64() {
                    Some(n) => format!("{} {}", fit_mark(f), gib(n)),
                    None => "-".into(),
                }
            };
            let status = if !strings(&q["installed"]).is_empty() {
                let ids: Vec<String> = strings(&q["installed"])
                    .into_iter()
                    .map(|id| {
                        let m = q["measured"].as_array().into_iter().flatten().find(|m| m["model"] == id);
                        let mut facts = Vec::new();
                        if let Some(m) = m {
                            if let Some(t) = m["tg_tps"].as_f64() {
                                facts.push(format!("{t:.1} tok/s"));
                            }
                            if let Some(p) = m["tools_pass_rate"].as_f64() {
                                facts.push(format!("tools {:.0}%", p * 100.0));
                            }
                            if m["agent_ready"] == true {
                                facts.push("agent-ready".into());
                            }
                        }
                        if facts.is_empty() { id.to_owned() } else { format!("{id} ({})", facts.join(", ")) }
                    })
                    .collect();
                format!("installed: {}", ids.join(", "))
            } else if q["downloaded"] == true {
                "downloaded".into()
            } else {
                String::new()
            };
            [
                label,
                gib(q["size"].as_u64().unwrap_or(0)),
                q["bits_per_weight"].as_f64().map_or("-".into(), |b| format!("{b:.2}")),
                q["tier"].as_str().unwrap_or("-").to_owned(),
                fit("gpu"),
                fit("cpu"),
                status,
            ]
        })
        .collect();
    if !rows.is_empty() {
        let at = d["fit_context"].as_u64().filter(|_| !speech).map_or(String::new(), |c| format!(" @{}k", c / 1024));
        let (gpu, cpu) = (format!("GPU (needs{at})"), format!("CPU (needs{at})"));
        let _ = writeln!(s, "\n{}", table(&["QUANT", "SIZE", "BPW", "QUALITY", &gpu, &cpu, ""], &rows));
        if let Some(m) = d["mmproj"]["path"].as_str() {
            let _ = writeln!(s, "\x1b[2mvision projector {m} ({}) is included in sizes and installs\x1b[0m", gib(d["mmproj"]["size"].as_u64().unwrap_or(0)));
        }
    }
    let npu: Vec<[String; 4]> = d["npu"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|n| {
            let mut notes = Vec::new();
            if n["single_turn"] == true {
                notes.push("single-turn, 1k ctx".to_owned());
            }
            if n["unsupported"] == true {
                notes.push("runs next to an NPU LLM: set asr = true for FastFlowLM".to_owned());
            }
            if !strings(&n["installed"]).is_empty() {
                notes.push(format!("installed: {}", strings(&n["installed"]).join(", ")));
            } else if n["downloaded"] == true {
                notes.push("downloaded".into());
            }
            [
                format!("flm:{}", n["tag"].as_str().unwrap_or_default()),
                n["footprint_bytes"].as_u64().map_or("-".into(), gib),
                n["default_context"].as_u64().filter(|c| *c >= 1024).map_or("-".into(), |c| format!("{}k", c / 1024)),
                notes.join(", "),
            ]
        })
        .collect();
    if !npu.is_empty() {
        let _ = writeln!(s, "\nNPU (FastFlowLM)\n{}", table(&["TAG", "MEMORY", "CTX", ""], &npu));
    }
    let target = d["id"].as_str().or(d["repo"].as_str()).unwrap_or("<model>");
    if rows.is_empty() {
        let _ = write!(s, "\x1b[2minstall: contio install {target}\x1b[0m");
    } else {
        let _ = write!(
            s,
            "\x1b[2m* recommended · fit at the default context (change with -c) · install: contio install {target} [--quant Q] [--device gpu|cpu{}]\x1b[0m",
            if npu.is_empty() { "" } else { "|npu" }
        );
    }
    s
}

pub fn hf_hits(raw: &Value) -> String {
    let rows: Vec<[String; 5]> = raw
        .as_array()
        .into_iter()
        .flatten()
        .map(|h| {
            let mut caps = Vec::new();
            if h["tools"] == true {
                caps.push("tools");
            }
            if h["reasoning"] == true {
                caps.push("reasoning");
            }
            if h["gated"] == true {
                caps.push("gated");
            }
            [
                h["repo"].as_str().unwrap_or_default().to_owned(),
                params(h),
                h["arch"].as_str().unwrap_or("-").to_owned(),
                h["downloads"].as_u64().map_or("-".into(), |d| format!("{d}")),
                caps.join(","),
            ]
        })
        .collect();
    if rows.is_empty() {
        return "no GGUF repos found".into();
    }
    format!(
        "{}\n\x1b[2mcommunity uploads, not curated · details: contio show <owner/repo>\x1b[0m",
        table(&["REPO", "PARAMS", "ARCH", "DOWNLOADS", ""], &rows)
    )
}

pub fn status(url: &str, health: &Value, models: &[ModelStatus]) -> String {
    let loaded: Vec<&str> = models
        .iter()
        .filter(|m| matches!(m.state, ModelState::Ready { .. }))
        .map(|m| m.id.as_str())
        .collect();
    let mut s = format!(
        "contiod {} at {url}\nOpenAI API: {url}/v1\nmodels: {} configured, {} loaded",
        health["version"].as_str().unwrap_or("?"),
        models.len(),
        loaded.len()
    );
    if !loaded.is_empty() {
        let _ = write!(s, " ({})", loaded.join(", "));
    }
    s
}

pub fn hardware(hw: &HardwareInfo) -> String {
    let mut s = String::new();
    let classes = hw
        .cpu
        .core_classes
        .iter()
        .map(|c| format!("{}× {:.1} GHz", c.cores, f64::from(c.max_mhz) / 1000.0))
        .collect::<Vec<_>>()
        .join(" + ");
    let _ = writeln!(s, "CPU   {}", hw.cpu.model);
    let _ = writeln!(
        s,
        "      {} cores / {} threads ({classes}){}",
        hw.cpu.cores,
        hw.cpu.threads,
        if hw.cpu.avx512 { ", AVX-512" } else { "" }
    );
    match &hw.gpu {
        Some(g) => {
            let _ = writeln!(s, "GPU   {} {}  {} VRAM + {} GTT", g.driver, g.pci_id, gib(g.vram_total_bytes), gib(g.gtt_total_bytes));
        }
        None => s.push_str("GPU   not detected\n"),
    }
    match &hw.npu {
        Some(n) => {
            let access = if n.accessible { "" } else { "  (no access: check the `render` group)" };
            let _ = writeln!(s, "NPU   {} {}{access}", n.driver, n.device_path);
        }
        None => s.push_str("NPU   not detected\n"),
    }
    for p in &hw.pools {
        let _ = writeln!(s, "MEM   {:<6} {}", pool_name(p.kind), pool_bar(p));
    }
    let profile = match &hw.power.profile {
        PowerProfile::LowPower => "low-power".to_owned(),
        PowerProfile::Balanced => "balanced".to_owned(),
        PowerProfile::Performance => "performance".to_owned(),
        PowerProfile::Unknown(u) => format!("unknown ({u})"),
    };
    let ac = match hw.power.on_ac {
        Some(true) => ", on AC",
        Some(false) => ", on battery",
        None => "",
    };
    let bat = hw.power.battery_percent.map(|b| format!(", battery {b}%")).unwrap_or_default();
    let _ = write!(s, "POWER {profile}{ac}{bat}");
    s
}

pub fn job_line(job: &Value) -> String {
    let title = job["title"].as_str().unwrap_or_default();
    let step = job["step"].as_str().unwrap_or("");
    let progress = match (job["progress"]["done"].as_u64(), job["progress"]["total"].as_u64()) {
        (Some(d), Some(t)) if t > 0 => format!(" {:>5.1}% ({} / {})", 100.0 * d as f64 / t as f64, gib(d), gib(t)),
        (Some(d), _) => format!(" {}", gib(d)),
        _ => String::new(),
    };
    let state = job["state"].as_str().unwrap_or("?");
    format!("[{state}] {title}: {step}{progress}")
}

pub fn jobs(raw: &Value) -> String {
    let jobs = raw.as_array().cloned().unwrap_or_default();
    if jobs.is_empty() {
        return "no jobs".into();
    }
    jobs.iter()
        .map(|j| {
            let mut line = format!("{:>3}  {}", j["id"], job_line(j));
            if let Some(e) = j["error"].as_str() {
                let _ = write!(line, "\n     error: {}", e.lines().next().unwrap_or_default());
            }
            if let Some(n) = j["retried_as"].as_u64() {
                let _ = write!(line, " → retried as job {n}");
            }
            line
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn backends(raw: &Value) -> String {
    let mut s = String::new();
    let registered: Vec<&str> = raw["registered"].as_array().into_iter().flatten().filter_map(|v| v.as_str()).collect();
    if registered.is_empty() {
        s.push_str("registered: none\n");
    } else {
        s.push_str("registered (first match wins):\n");
        for r in &registered {
            let _ = writeln!(s, "  {r}");
        }
    }
    let installed = raw["installed"].as_array().cloned().unwrap_or_default();
    if installed.is_empty() {
        s.push_str("installed:  none (try `contio backends install llama.cpp --method prebuilt`)");
    } else {
        s.push_str("installed:");
        for b in installed {
            let at = b["bin_dir"].as_str().or(b["image"].as_str()).or(b["prefix"].as_str()).unwrap_or("?");
            let _ = write!(s, "\n  {} {} ({})  {at}", b["backend"].as_str().unwrap_or("?"), b["version"].as_str().unwrap_or("?"), b["method"].as_str().unwrap_or("?"));
        }
    }
    s
}

pub fn preflight(raw: &Value) -> String {
    if raw["ok"] == true {
        return "all prerequisites present".into();
    }
    let p = &raw["preflight"];
    let mut lines = Vec::new();
    let list = |v: &Value| v.as_array().into_iter().flatten().filter_map(|x| x.as_str().map(str::to_owned)).collect::<Vec<_>>();
    let pkgs = list(&p["missing_packages"]);
    if !pkgs.is_empty() {
        lines.push(format!("missing packages: {}", pkgs.join(", ")));
    }
    let tools = list(&p["missing_tools"]);
    if !tools.is_empty() {
        lines.push(format!("missing tools: {}", tools.join(", ")));
    }
    lines.extend(list(&p["notes"]));
    if let Some(h) = raw["install_hint"].as_str() {
        lines.push(format!("\nrun:  {h}"));
    }
    lines.join("\n")
}

fn pool_bar(p: &MemoryPool) -> String {
    const WIDTH: usize = 24;
    let frac = if p.total_bytes == 0 { 0.0 } else { p.used_bytes as f64 / p.total_bytes as f64 };
    let filled = ((frac * WIDTH as f64).round() as usize).min(WIDTH);
    format!("[{}{}] {} / {}", "█".repeat(filled), "·".repeat(WIDTH - filled), gib(p.used_bytes), gib(p.total_bytes))
}

fn table<const N: usize>(header: &[&str; N], rows: &[[String; N]]) -> String {
    let mut widths = header.map(|h| h.chars().count());
    for r in rows {
        for (w, c) in widths.iter_mut().zip(r) {
            *w = (*w).max(c.chars().count());
        }
    }
    let line = |cells: Vec<&str>| {
        cells
            .iter()
            .zip(widths)
            .map(|(c, w)| format!("{c:<w$}"))
            .collect::<Vec<_>>()
            .join("  ")
            .trim_end()
            .to_owned()
    };
    let mut out = vec![line(header.to_vec())];
    out.extend(rows.iter().map(|r| line(r.iter().map(String::as_str).collect())));
    out.join("\n")
}

#[cfg(test)]
mod tests {
    use contio_core::hw::DeviceKind;

    use super::*;

    fn status(id: &str, state: ModelState) -> ModelStatus {
        ModelStatus {
            id: id.into(),
            device: DeviceKind::Gpu,
            kind: ModelKind::Chat,
            context: Some(32768),
            managed: id != "a",
            origin: None,
            source: contio_core::model::ModelSource::Flm { tag: id.into() },
            pool: MemoryPoolKind::Gpu,
            backend: Some("llama.cpp".into()),
            pinned: id == "a",
            idle_ttl_secs: None,
            extra_args: vec![],
            estimated_bytes: Some(24 * 1024 * 1024 * 1024),
            state,
            in_flight: 0,
        }
    }

    #[test]
    fn models_table_aligns() {
        let t = models(&[
            status("a", ModelState::Ready { port: 4242 }),
            status("longer-name", ModelState::Failed { error: "boom\nmore".into() }),
        ]);
        let lines: Vec<&str> = t.lines().collect();
        assert!(lines[0].starts_with("MODEL        DEVICE"));
        assert!(lines[1].contains("ready :4242") && lines[1].ends_with("pinned, config"), "{}", lines[1]);
        assert!(lines[1].contains("32k"));
        assert!(lines[2].contains("failed: boom") && !lines[2].contains("more"));
        assert!(lines[1].contains("24.0 GiB"));
    }

    #[test]
    fn job_lines() {
        let j = serde_json::json!({"title": "x/y.gguf", "state": "running", "step": "downloading",
                                   "progress": {"done": 512u64 * 1024 * 1024, "total": 1024u64 * 1024 * 1024}});
        assert_eq!(job_line(&j), "[running] x/y.gguf: downloading  50.0% (0.5 GiB / 1.0 GiB)");
        let failed = serde_json::json!([{"id": 3, "title": "t", "state": "failed", "error": "boom\nmore"}]);
        assert!(jobs(&failed).contains("error: boom") && !jobs(&failed).contains("more"));
    }

    #[test]
    fn preflight_hint() {
        let raw = serde_json::json!({"ok": false, "install_hint": "sudo dnf install glslc",
                                     "preflight": {"missing_packages": ["glslc"], "missing_tools": [], "notes": []}});
        let t = preflight(&raw);
        assert!(t.contains("missing packages: glslc") && t.ends_with("run:  sudo dnf install glslc"));
    }

    #[test]
    fn empty_models_hint() {
        assert!(models(&[]).contains("contio search"));
    }

    fn overview() -> Value {
        serde_json::json!({
            "families": [{"id": "qwen", "name": "Qwen", "vendor": "Alibaba", "license": "Apache-2.0", "summary": ""},
                         {"id": "embeddings", "name": "Embeddings", "vendor": "v", "license": "l", "summary": ""}],
            "models": [
                {"id": "qwen3.6-35b-a3b", "name": "Qwen3.6 35B-A3B", "family": "qwen", "kind": "chat",
                 "params": 34660610688u64, "active_b": 3.0, "capabilities": ["tools", "vision"], "quant_count": 25,
                 "npu": ["qwen3.6-moe:35b-a3b"], "installed": ["qwen3.6-35b-a3b"],
                 "suggested": {"label": "UD-Q4_K_XL", "size": 22360456160u64,
                               "fit": {"gpu": {"verdict": "fits", "needed_bytes": 1, "budget_bytes": 2},
                                       "cpu": {"verdict": "too_big", "needed_bytes": 1, "budget_bytes": 2}}}},
                {"id": "nomic", "name": "Nomic", "family": "embeddings", "kind": "embedding", "params": 137000000,
                 "capabilities": [], "quant_count": 0, "npu": [], "installed": [], "suggested": null,
                 "error": "offline"}
            ]
        })
    }

    #[test]
    fn catalog_table_and_filters() {
        let f = |query: &str, device: Option<&str>, fits: bool| CatalogFilter {
            query: query.into(),
            device: device.map(str::to_owned),
            fits,
            kind: None,
        };
        let all = catalog(&overview(), &f("", None, false));
        assert!(all.contains("35B-A3B") && all.contains("UD-Q4_K_XL 20.8 GiB") && all.contains("gpu ✓ cpu ✗"));
        assert!(all.contains("tools,vision,npu") && all.contains("installed"));
        assert!(all.contains("137M") && all.contains("(offline)") && all.contains("embedding"));
        assert!(!catalog(&overview(), &f("qwen", None, false)).contains("Nomic"));
        let npu = catalog(&overview(), &f("", Some("npu"), false));
        assert!(npu.contains("qwen3.6") && !npu.contains("nomic"));
        let fits_cpu = catalog(&overview(), &f("", Some("cpu"), true));
        assert!(fits_cpu.contains("no matches"));
    }

    #[test]
    fn detail_table() {
        let d = serde_json::json!({
            "id": "m", "name": "M", "repo": "o/m", "params": 8000000000u64, "kind": "chat",
            "context_length": 40960, "default_context": 32768, "capabilities": ["tools"],
            "quants": [{"label": "Q4_K_M", "size": 5000000000u64, "bits_per_weight": 4.9, "tier": "very good",
                        "recommended": true, "installed": ["m"], "downloaded": true,
                        "fit": {"gpu": {"verdict": "fits", "needed_bytes": 6000000000u64, "budget_bytes": 1},
                                "cpu": null}}],
            "mmproj": null,
            "npu": [{"tag": "m:8b", "footprint_bytes": 5000000000u64, "default_context": 16384,
                     "single_turn": false, "unsupported": false, "installed": [], "downloaded": true}]
        });
        let t = detail(&d);
        assert!(t.contains("Q4_K_M *") && t.contains("✓ 5.6 GiB") && t.contains("installed: m"), "{t}");
        assert!(t.contains("flm:m:8b") && t.contains("downloaded"));
        assert!(t.contains("contio install m"));
    }

    #[test]
    fn pool_bar_is_bounded() {
        let p = MemoryPool { kind: MemoryPoolKind::Gpu, total_bytes: 100, used_bytes: 250 };
        assert_eq!(pool_bar(&p).chars().filter(|c| *c == '█').count(), 24);
    }
}
