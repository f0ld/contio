//! Benchmarks (spec §8): runs as background jobs, results in
//! `<data_dir>/bench.db`.
//!
//! A performance run unloads the model and loads it again (load time and
//! memory growth), keeps a lease so it can't be evicted midway, and sends its
//! requests through the daemon's own OpenAI endpoint, so the measured path is
//! the one clients use.

use std::time::{Duration, Instant};

use axum::Json;
use axum::extract::{Path as UrlPath, Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use contio_bench::store::{Filter, NewRun, Run, RunStatus};
use contio_bench::tools::Suite;
use contio_bench::{Conditions, LoadReport, Monitor, PerfRun, Preset, Progress, SuiteKind, Target};
use contio_core::model::{ModelKind, ModelState};
use contio_install::jobs::JobCtx;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::api::AppState;
use crate::install_api::{accepted, bad_request, not_found};

/// Lets memory settle after an unload before taking the baseline.
const SETTLE: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchRequest {
    /// Model ids or aliases, run one after another.
    pub models: Vec<String>,
    #[serde(default)]
    pub preset: Preset,
    /// Default: `["perf"]`.
    #[serde(default = "default_suites")]
    pub suites: Vec<SuiteKind>,
    /// Unload other (unpinned, idle) models first so they don't share memory
    /// bandwidth. Default: true.
    #[serde(default = "yes")]
    pub exclusive: bool,
    /// Unload and reload the model to measure load time and memory. Default: true.
    #[serde(default = "yes")]
    pub reload: bool,
    /// Switch to this power profile for the run and restore it afterwards
    /// (power-saver, balanced, performance). Default: leave it as it is.
    #[serde(default)]
    pub power_profile: Option<String>,
}

/// Firmware power limits take a moment to follow a profile switch.
const PROFILE_SETTLE: Duration = Duration::from_secs(3);

fn yes() -> bool {
    true
}

fn default_suites() -> Vec<SuiteKind> {
    vec![SuiteKind::Perf]
}

pub async fn start(State(state): State<AppState>, Json(req): Json<BenchRequest>) -> Response {
    if req.models.is_empty() || req.suites.is_empty() {
        return bad_request("`models` and `suites` must not be empty");
    }
    if let Some(p) = &req.power_profile
        && !contio_bench::power_profile::PROFILES.contains(&p.as_str())
    {
        return bad_request("power_profile: power-saver, balanced or performance");
    }
    // one benchmark at a time: parallel runs would measure each other
    if let Some(j) = state
        .installer
        .jobs
        .list()
        .into_iter()
        .find(|j| j.kind == "bench" && j.state == contio_install::jobs::JobState::Running)
    {
        let msg = format!("benchmark job {} is still running; wait for it or cancel it", j.id);
        return (StatusCode::CONFLICT, Json(json!({"error": {"message": msg, "type": "conflict", "job_id": j.id}})))
            .into_response();
    }
    let mut ids = Vec::new();
    for name in &req.models {
        let (id, _) = crate::gateway::resolve(&state, name);
        match state.manager.profile(&id) {
            Ok(p) if req.suites.iter().any(|s| s.applies_to(p.kind)) => ids.push(id),
            Ok(p) => {
                return bad_request(&format!(
                    "none of the chosen suites runs on `{name}` ({:?} model): chat models take perf, tools, \
                     reasoning, knowledge, coding; embedding and reranker models take retrieval",
                    p.kind
                ));
            }
            Err(e) => return crate::gateway::backend_error(&e),
        }
    }
    let suites: Vec<&str> = req.suites.iter().map(|s| s.name()).collect();
    let title = format!("benchmark {} ({}, {})", ids.join(", "), suites.join("+"), json!(req.preset).as_str().unwrap_or_default());
    let st = state.clone();
    // one benchmark at a time: parallel runs would measure each other
    let job = state.installer.jobs.spawn("bench", &title, Some("bench".into()), move |ctx| async move {
        let guard = match &req.power_profile {
            Some(p) => {
                ctx.step(&format!("switching to the {p} power profile"));
                let g = contio_bench::power_profile::apply(p).await?;
                if g.previous != *p {
                    ctx.log(&format!("power profile {} → {p} (restored afterwards)", g.previous));
                    tokio::time::sleep(PROFILE_SETTLE).await;
                }
                Some(g)
            }
            None => None,
        };
        let mut runs = Vec::new();
        let n = ids.len();
        for (i, id) in ids.iter().enumerate() {
            // no early return: the power profile must be restored below
            if ctx.is_cancelled() {
                break;
            }
            let sub = if n > 1 { ctx.with_prefix(&format!("[{}/{n} {id}] ", i + 1)) } else { ctx.clone() };
            let kind = st.manager.profile(id).map(|p| p.kind).unwrap_or_default();
            for &suite in req.suites.iter().filter(|s| s.applies_to(kind)) {
                runs.push(run_suite(&st, id, suite, &req, &sub).await);
            }
        }
        if let Some(g) = guard {
            let previous = g.previous.clone();
            match g.restore().await {
                Ok(()) => ctx.log(&format!("power profile restored to {previous}")),
                Err(e) => ctx.log(&format!("couldn't restore the power profile to {previous}: {e}")),
            }
        }
        ctx.check_cancelled()?;
        let failed: Vec<String> = runs.iter().filter_map(|r| r.as_ref().err().cloned()).collect();
        let ids: Vec<i64> = runs.iter().filter_map(|r| r.as_ref().ok().copied()).collect();
        if ids.is_empty() {
            return Err(failed.join("; "));
        }
        Ok(json!({ "runs": ids, "errors": failed }))
    });
    accepted(job)
}

struct JobProgress<'a>(&'a JobCtx);

impl Progress for JobProgress<'_> {
    fn step(&self, message: &str) {
        self.0.step(message);
    }

    fn cancelled(&self) -> bool {
        self.0.is_cancelled()
    }
}

/// One suite on one model; returns the run id (the run is stored either way).
async fn run_suite(state: &AppState, id: &str, suite: SuiteKind, req: &BenchRequest, ctx: &JobCtx) -> Result<i64, String> {
    let profile = state.manager.profile(id).map_err(|e| e.to_string())?;
    let (backend, backend_desc) = state.manager.backend_description(id).map_err(|e| e.to_string())?;

    // other models only matter for speed and energy
    if req.exclusive && suite == SuiteKind::Perf {
        for s in state.manager.statuses() {
            let loaded = matches!(s.state, ModelState::Ready { .. } | ModelState::Loading);
            if s.id != id && loaded && !s.pinned && s.in_flight == 0 {
                ctx.step(&format!("unloading {}", s.id));
                let _ = state.manager.unload(&s.id).await;
            }
        }
    }
    let others: Vec<String> = state
        .manager
        .statuses()
        .into_iter()
        .filter(|s| s.id != id && matches!(s.state, ModelState::Ready { .. }))
        .map(|s| s.id)
        .collect();
    let hw = state.probe.probe().ok();
    let conditions = Conditions {
        power_profile: hw.as_ref().map(|h| h.power.profile.clone()),
        on_ac: hw.as_ref().and_then(|h| h.power.on_ac),
        battery_percent: hw.as_ref().and_then(|h| h.power.battery_percent),
        backend: Some(backend_desc),
        contio_version: env!("CARGO_PKG_VERSION").into(),
        cpu: hw.as_ref().map(|h| h.cpu.model.clone()),
        other_models: others,
    };
    let target_info = json!({
        "id": profile.id, "device": profile.device, "backend": backend, "context": profile.context,
        "source": profile.source, "origin": profile.origin, "extra_args": profile.extra_args,
    });
    let conditions_json = serde_json::to_value(&conditions).unwrap_or_default();
    let preset = json!(req.preset).as_str().unwrap_or_default().to_owned();
    let run_id = state
        .bench
        .start(&NewRun {
            suite: suite.name(),
            model: id,
            preset: &preset,
            target: &target_info,
            conditions: &conditions_json,
            job_id: Some(ctx.id()),
        })?;

    let result = match suite {
        SuiteKind::Perf => measure(state, id, &profile, &backend, req, ctx)
            .await
            .map(|r| (serde_json::to_value(&r).unwrap_or_default(), r.metrics())),
        SuiteKind::Tools => tool_calls(state, id, req, ctx)
            .await
            .map(|r| (serde_json::to_value(&r).unwrap_or_default(), r.metrics())),
        SuiteKind::Reasoning | SuiteKind::Knowledge | SuiteKind::Coding => quality(state, id, suite, req, ctx)
            .await
            .map(|r| (serde_json::to_value(&r).unwrap_or_default(), r.metrics())),
        SuiteKind::Retrieval => retrieval(state, id, profile.kind, ctx)
            .await
            .map(|r| (serde_json::to_value(&r).unwrap_or_default(), r.metrics())),
    };
    let (status, error, report, metrics) = match &result {
        Ok((report, metrics)) => (RunStatus::Done, None, report.clone(), metrics.clone()),
        Err(e) if ctx.is_cancelled() => (RunStatus::Cancelled, Some(e.as_str()), Value::Null, Default::default()),
        Err(e) => (RunStatus::Failed, Some(e.as_str()), Value::Null, Default::default()),
    };
    state.bench.finish(run_id, status, error, &report, &metrics)?;
    result.map(|_| run_id).map_err(|e| format!("{id}: {e}"))
}

async fn tool_calls(
    state: &AppState,
    id: &str,
    req: &BenchRequest,
    ctx: &JobCtx,
) -> Result<contio_bench::tools::ToolsReport, String> {
    ctx.step("loading");
    // held for the whole run: the model can't be evicted midway
    let lease = state.manager.acquire(id).await.map_err(|e| e.to_string())?;
    let target = Target { base_url: state.self_url.clone(), model: id.to_owned(), context: None, llama: false };
    let report = contio_bench::tools::run(&state.http, &target, &Suite::builtin(), req.preset.quality_reps(), &JobProgress(ctx)).await;
    drop(lease);
    report
}

async fn retrieval(
    state: &AppState,
    id: &str,
    kind: ModelKind,
    ctx: &JobCtx,
) -> Result<contio_bench::retrieval::RetrievalReport, String> {
    use contio_bench::retrieval::Mode;
    let mode = if kind == ModelKind::Reranker { Mode::Reranker } else { Mode::Embedding };
    ctx.step("loading");
    let lease = state.manager.acquire(id).await.map_err(|e| e.to_string())?;
    let target = Target { base_url: state.self_url.clone(), model: id.to_owned(), context: None, llama: false };
    let report = contio_bench::retrieval::run(&state.http, &target, mode, &JobProgress(ctx)).await;
    drop(lease);
    report
}

async fn quality(
    state: &AppState,
    id: &str,
    suite: SuiteKind,
    req: &BenchRequest,
    ctx: &JobCtx,
) -> Result<contio_bench::quality::QualityReport, String> {
    let which = suite.quality().expect("a quality suite");
    ctx.step("loading");
    let lease = state.manager.acquire(id).await.map_err(|e| e.to_string())?;
    let target = Target { base_url: state.self_url.clone(), model: id.to_owned(), context: None, llama: false };
    let work = state.installer.data_dir.join("cache/sandbox");
    let report =
        contio_bench::quality::run(&state.http, &target, which, req.preset, &work, &JobProgress(ctx)).await;
    drop(lease);
    report
}

async fn measure(
    state: &AppState,
    id: &str,
    profile: &contio_core::model::ModelProfile,
    backend: &str,
    req: &BenchRequest,
    ctx: &JobCtx,
) -> Result<PerfRun, String> {
    let monitor = Monitor::start(Some(state.probe.clone()));
    let mut load = LoadReport {
        estimated_bytes: state.manager.statuses().into_iter().find(|s| s.id == id).and_then(|s| s.estimated_bytes),
        ..Default::default()
    };

    let baseline = if req.reload {
        if state.manager.unload(id).await.map_err(|e| e.to_string())? {
            ctx.step("unloading for a cold start");
        }
        tokio::time::sleep(SETTLE).await;
        let probe = state.probe.clone();
        tokio::task::spawn_blocking(move || probe.sample()).await.ok().and_then(Result::ok).map(|m| m.pools)
    } else {
        None
    };
    monitor.reset_peaks();
    ctx.step("loading");
    let started = Instant::now();
    // held for the whole run: the model can't be evicted or unloaded midway
    let lease = state.manager.acquire(id).await.map_err(|e| e.to_string())?;
    if req.reload {
        load.load_s = Some(started.elapsed().as_secs_f64());
    }
    if let Some(pid) = state.manager.pid(id) {
        monitor.watch_process(pid);
    }

    let target = Target {
        base_url: state.self_url.clone(),
        model: id.to_owned(),
        context: profile.context,
        llama: backend == "llama.cpp",
    };
    let perf = contio_bench::perf::run(&state.http, &target, &req.preset.perf_options(), &monitor, &JobProgress(ctx))
        .await?;
    drop(lease);
    load.peak_rss_bytes = monitor.peak_rss();

    if let Some(base) = baseline {
        for (kind, peak) in monitor.peak_used() {
            if let Some(b) = base.iter().find(|p| p.kind == kind) {
                load.memory_delta.insert(kind, peak.saturating_sub(b.used_bytes));
            }
        }
    }
    Ok(PerfRun { load, perf })
}

// --- results ------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ListQuery {
    pub model: Option<String>,
    pub suite: Option<String>,
    pub limit: Option<u32>,
}

pub async fn list(State(state): State<AppState>, Query(q): Query<ListQuery>) -> Response {
    match state.bench.list(&Filter { model: q.model, suite: q.suite, limit: q.limit }) {
        Ok(runs) => Json(runs).into_response(),
        Err(e) => internal(&e),
    }
}

pub async fn get(State(state): State<AppState>, UrlPath(id): UrlPath<i64>) -> Response {
    match state.bench.get(id) {
        Ok(Some(r)) => Json(r).into_response(),
        Ok(None) => not_found(&format!("no benchmark run {id}")),
        Err(e) => internal(&e),
    }
}

pub async fn delete(State(state): State<AppState>, UrlPath(id): UrlPath<i64>) -> Response {
    match state.bench.get(id) {
        Ok(Some(r)) if r.status == RunStatus::Running => bad_request("run is still going; cancel its job first"),
        Ok(Some(_)) => match state.bench.delete(id) {
            Ok(_) => Json(json!({ "deleted": id })).into_response(),
            Err(e) => internal(&e),
        },
        Ok(None) => not_found(&format!("no benchmark run {id}")),
        Err(e) => internal(&e),
    }
}

#[derive(Debug, Deserialize)]
pub struct ExportQuery {
    #[serde(default)]
    pub format: Option<String>,
}

/// All finished runs with their metrics as CSV (one row per run, one column
/// per metric) or JSON (full reports).
pub async fn export(State(state): State<AppState>, Query(q): Query<ExportQuery>) -> Response {
    let runs = match state.bench.list(&Filter::default()) {
        Ok(r) => r,
        Err(e) => return internal(&e),
    };
    match q.format.as_deref().unwrap_or("csv") {
        "json" => {
            let full: Vec<Run> = runs.iter().filter_map(|r| state.bench.get(r.id).ok().flatten()).collect();
            Json(full).into_response()
        }
        "csv" => ([(header::CONTENT_TYPE, "text/csv; charset=utf-8")], to_csv(&runs)).into_response(),
        other => bad_request(&format!("unknown format `{other}` (csv or json)")),
    }
}

fn to_csv(runs: &[Run]) -> String {
    let mut names: Vec<&String> = runs.iter().flat_map(|r| r.metrics.keys()).collect();
    names.sort();
    names.dedup();
    let text = |v: &Value| v.as_str().map(str::to_owned).unwrap_or_else(|| if v.is_null() { String::new() } else { v.to_string() });
    let quote = |s: &str| if s.contains([',', '"', '\n']) { format!("\"{}\"", s.replace('"', "\"\"")) } else { s.to_owned() };
    let mut out = ["id", "suite", "model", "preset", "status", "started_at", "device", "backend", "quant", "power_profile", "on_ac"]
        .iter()
        .map(|s| (*s).to_owned())
        .chain(names.iter().map(|n| (*n).clone()))
        .collect::<Vec<_>>()
        .join(",");
    out.push('\n');
    for r in runs {
        let mut row = vec![
            r.id.to_string(),
            r.suite.clone(),
            r.model.clone(),
            r.preset.clone(),
            text(&json!(r.status)),
            r.started_at.to_string(),
            text(&r.target["device"]),
            text(&r.target["backend"]),
            text(&r.target["origin"]["quant"]),
            text(&r.conditions["power_profile"]),
            text(&r.conditions["on_ac"]),
        ];
        row.extend(names.iter().map(|n| r.metrics.get(*n).map(|v| format!("{v:.4}")).unwrap_or_default()));
        out.push_str(&row.iter().map(|c| quote(c)).collect::<Vec<_>>().join(","));
        out.push('\n');
    }
    out
}

fn internal(msg: &str) -> Response {
    (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": {"message": msg, "type": "internal_error"}}))).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn csv_has_a_column_per_metric() {
        let run = |id, metrics: &[(&str, f64)]| Run {
            id,
            suite: "perf".into(),
            model: "m,1".into(),
            preset: "quick".into(),
            started_at: 0,
            finished_at: None,
            status: RunStatus::Done,
            error: None,
            target: json!({"device": "gpu", "backend": "llama.cpp", "origin": {"quant": "Q4_K_M"}}),
            conditions: json!({"power_profile": "balanced", "on_ac": true}),
            report: Value::Null,
            metrics: metrics.iter().map(|(k, v)| ((*k).to_owned(), *v)).collect::<BTreeMap<_, _>>(),
            job_id: None,
        };
        let csv = to_csv(&[run(2, &[("tg_tps", 22.0)]), run(1, &[("pp512_tps", 400.0)])]);
        let lines: Vec<&str> = csv.lines().collect();
        assert!(lines[0].ends_with(",on_ac,pp512_tps,tg_tps"));
        assert_eq!(lines[1], "2,perf,\"m,1\",quick,done,0,gpu,llama.cpp,Q4_K_M,balanced,true,,22.0000");
    }
}
