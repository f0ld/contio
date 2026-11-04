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
use contio_bench::{Conditions, LoadReport, Monitor, PerfRun, Preset, Progress, Target};
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
    /// Unload other (unpinned, idle) models first so they don't share memory
    /// bandwidth. Default: true.
    #[serde(default = "yes")]
    pub exclusive: bool,
    /// Unload and reload the model to measure load time and memory. Default: true.
    #[serde(default = "yes")]
    pub reload: bool,
}

fn yes() -> bool {
    true
}

pub async fn start(State(state): State<AppState>, Json(req): Json<BenchRequest>) -> Response {
    if req.models.is_empty() {
        return bad_request("`models` is empty");
    }
    let mut ids = Vec::new();
    for name in &req.models {
        let (id, _) = crate::gateway::resolve(&state, name);
        match state.manager.profile(&id) {
            Ok(p) if p.kind == ModelKind::Chat => ids.push(id),
            Ok(p) => return bad_request(&format!("`{name}` is a {:?} model; speed benchmarks cover chat models", p.kind)),
            Err(e) => return crate::gateway::backend_error(&e),
        }
    }
    let title = format!("benchmark {} ({})", ids.join(", "), json!(req.preset).as_str().unwrap_or_default());
    let st = state.clone();
    // one benchmark at a time: parallel runs would measure each other
    let job = state.installer.jobs.spawn("bench", &title, Some("bench".into()), move |ctx| async move {
        let mut runs = Vec::new();
        let n = ids.len();
        for (i, id) in ids.iter().enumerate() {
            ctx.check_cancelled()?;
            let sub = if n > 1 { ctx.with_prefix(&format!("[{}/{n} {id}] ", i + 1)) } else { ctx.clone() };
            runs.push(run_perf(&st, id, &req, &sub).await);
        }
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

/// One performance run; returns the run id (the run is stored either way).
async fn run_perf(state: &AppState, id: &str, req: &BenchRequest, ctx: &JobCtx) -> Result<i64, String> {
    let profile = state.manager.profile(id).map_err(|e| e.to_string())?;
    let (backend, backend_desc) = state.manager.backend_description(id).map_err(|e| e.to_string())?;

    if req.exclusive {
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
            suite: "perf",
            model: id,
            preset: &preset,
            target: &target_info,
            conditions: &conditions_json,
            job_id: Some(ctx.id()),
        })?;

    let result = measure(state, id, &profile, &backend, req, ctx).await;
    let (status, error, report, metrics) = match &result {
        Ok(r) => (RunStatus::Done, None, serde_json::to_value(r).unwrap_or_default(), r.metrics()),
        Err(e) if ctx.is_cancelled() => (RunStatus::Cancelled, Some(e.as_str()), Value::Null, Default::default()),
        Err(e) => (RunStatus::Failed, Some(e.as_str()), Value::Null, Default::default()),
    };
    state.bench.finish(run_id, status, error, &report, &metrics)?;
    result.map(|_| run_id).map_err(|e| format!("{id}: {e}"))
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
