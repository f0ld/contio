//! Catalog API (spec §4, §6): browse the curated catalog and Hugging Face,
//! check what fits, and install a model in one job: backend (if missing) →
//! download → register → optionally load.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::Json;
use axum::extract::{Path as UrlPath, Query, State};
use axum::response::{IntoResponse, Response};
use contio_backends::{FlmBackend, ModelManager};
use contio_catalog::fit::{Fit, Verdict, fit, fit_gguf, fit_whisper};
use contio_catalog::flm::{FlmModel, PLATFORM_XDNA2, parse_model_list};
use contio_catalog::hf::{HfClient, LISTING_TTL, RepoInfo, valid_repo};
use contio_catalog::quant::{Quant, RepoFile, RepoListing, bits_per_weight, quality_tier};
use contio_catalog::{Catalog, CatalogModel, Family};
use contio_core::hw::{DeviceKind, MemoryPoolKind};
use contio_core::model::{KvGeometry, ModelKind, ModelOrigin, ModelProfile, ModelSource};
use contio_install::download::{self, Remote};
use contio_install::jobs::JobCtx;
use contio_install::{BackendKind, FLM_DEFAULT_VERSION, LLAMACPP_DEFAULT_VERSION, WHISPER_DEFAULT_VERSION};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::api::AppState;
use crate::install_api::{JobSpec, accepted, bad_request, install_and_register, not_found};

/// Family id of FastFlowLM models that aren't linked to a curated model.
const NPU_FAMILY: &str = "npu";

#[derive(Clone)]
pub struct CatalogService {
    pub catalog: Arc<Catalog>,
    pub hf: HfClient,
}

impl CatalogService {
    pub fn new(catalog: Catalog, http: reqwest::Client, base: &str, cache_dir: PathBuf) -> Self {
        Self { catalog: Arc::new(catalog), hf: HfClient::new(http, base, cache_dir) }
    }
}

// --- views --------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct DeviceFits {
    pub gpu: Option<Fit>,
    pub cpu: Option<Fit>,
}

#[derive(Debug, Serialize)]
pub struct QuantView {
    pub label: String,
    pub size: u64,
    pub files: usize,
    pub bits_per_weight: Option<f32>,
    pub tier: Option<&'static str>,
    /// At `ModelDetail::fit_context`, projector included.
    pub fit: DeviceFits,
    pub recommended: bool,
    /// All files are on disk.
    pub downloaded: bool,
    /// Ids of registered models using this quant.
    pub installed: Vec<String>,
    /// Benchmark results of those models.
    pub measured: Vec<Measured>,
}

/// Latest benchmark results of an installed model (spec §4.3: measured
/// numbers next to the estimate).
#[derive(Debug, Clone, Default, Serialize)]
pub struct Measured {
    pub model: String,
    pub tg_tps: Option<f64>,
    pub pp512_tps: Option<f64>,
    pub j_per_token: Option<f64>,
    pub tools_pass_rate: Option<f64>,
    pub agent_ready: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct NpuView {
    pub tag: String,
    pub name: String,
    pub footprint_bytes: Option<u64>,
    pub default_context: Option<u32>,
    pub capabilities: Vec<String>,
    pub single_turn: bool,
    /// Not served by contio yet (speech).
    pub unsupported: bool,
    pub fit: Option<Fit>,
    pub downloaded: bool,
    pub installed: Vec<String>,
    pub measured: Vec<Measured>,
}

#[derive(Debug, Serialize)]
pub struct ModelDetail {
    /// Catalog id; `None` for an arbitrary Hugging Face repo.
    pub id: Option<String>,
    pub name: String,
    pub family: String,
    pub repo: Option<String>,
    pub kind: ModelKind,
    pub params: Option<u64>,
    pub active_b: Option<f32>,
    pub context_length: Option<u32>,
    pub default_context: u32,
    /// The context the quant fits were computed for.
    pub fit_context: u32,
    pub capabilities: Vec<String>,
    pub notes: Option<String>,
    pub recommended: Option<String>,
    /// Smallest first.
    pub quants: Vec<QuantView>,
    pub mmproj: Option<RepoFile>,
    pub npu: Vec<NpuView>,
    /// Listing problem (offline, gated repo, …).
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ModelSummary {
    pub id: String,
    pub name: String,
    pub family: String,
    pub repo: Option<String>,
    pub kind: ModelKind,
    pub params: Option<u64>,
    pub active_b: Option<f32>,
    pub capabilities: Vec<String>,
    pub default_context: u32,
    pub quant_count: usize,
    pub min_size: Option<u64>,
    pub max_size: Option<u64>,
    /// The quant installs default to, with its fit.
    pub suggested: Option<QuantView>,
    /// NPU variants (FastFlowLM tags).
    pub npu: Vec<String>,
    /// Registered model ids from this entry.
    pub installed: Vec<String>,
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Overview {
    pub families: Vec<Family>,
    pub models: Vec<ModelSummary>,
    /// FastFlowLM's model list was found (an FLM backend is set up).
    pub npu_available: bool,
}

// --- shared logic ---------------------------------------------------------------

/// What the views need to know about the machine and installed models.
struct Context {
    budgets: HashMap<MemoryPoolKind, u64>,
    profiles: Vec<ModelProfile>,
    models_dir: PathBuf,
    flm: Vec<FlmModel>,
    flm_model_dir: Option<PathBuf>,
    measured: HashMap<String, Measured>,
}

impl Context {
    fn new(state: &AppState) -> Self {
        let flm_config = state.installer.flm();
        let flm = flm_config
            .as_ref()
            .and_then(|c| std::fs::read_to_string(c.prefix.join("share/flm/model_list.json")).ok())
            .and_then(|json| parse_model_list(&json, PLATFORM_XDNA2).ok())
            .unwrap_or_default();
        Self {
            budgets: state.manager.pool_budgets().clone(),
            profiles: state.manager.profiles(),
            models_dir: state.installer.models_dir(),
            flm,
            flm_model_dir: flm_config.map(|c| c.model_path.join("models")),
            measured: measured(&state.bench),
        }
    }

    fn measured_for(&self, ids: &[String]) -> Vec<Measured> {
        ids.iter().filter_map(|id| self.measured.get(id).cloned()).collect()
    }

    fn budget(&self, device: DeviceKind) -> Option<u64> {
        let pool = ModelProfile::new("", ModelSource::Flm { tag: String::new() }, device).pool();
        self.budgets.get(&pool).copied()
    }

    fn quant_view(&self, repo: &str, q: &Quant, model: ModelInfo<'_>, recommended: Option<&str>) -> QuantView {
        let weights = q.size + model.mmproj_size;
        let fit_on = |d| {
            self.budget(d).map(|b| match model.kind {
                ModelKind::Transcription => fit_whisper(weights, b),
                _ => fit_gguf(weights, model.kv, model.context, b),
            })
        };
        let bpw = model.params.and_then(|p| bits_per_weight(q.size, p));
        let first_name = file_name(&q.files[0].path);
        QuantView {
            label: q.label.clone(),
            size: q.size,
            files: q.files.len(),
            bits_per_weight: bpw,
            tier: bpw.map(quality_tier),
            fit: DeviceFits { gpu: fit_on(DeviceKind::Gpu), cpu: fit_on(DeviceKind::Cpu) },
            recommended: recommended.is_some_and(|r| r.eq_ignore_ascii_case(&q.label)),
            downloaded: q.files.iter().all(|f| hf_path(&self.models_dir, repo, &f.path).is_file()),
            installed: self
                .profiles
                .iter()
                .filter(|p| {
                    let by_origin = p.origin.as_ref().is_some_and(|o| {
                        o.repo.as_deref() == Some(repo) && o.quant.as_deref().is_some_and(|l| l.eq_ignore_ascii_case(&q.label))
                    });
                    // models from config.toml have no origin: match the file name
                    let by_file = p.source.path().and_then(|path| path.file_name()).is_some_and(|n| n.to_string_lossy() == first_name);
                    by_origin || (p.origin.is_none() && by_file)
                })
                .map(|p| p.id.clone())
                .collect(),
            measured: vec![],
        }
        .with_measured(self)
    }

    fn npu_view(&self, m: &FlmModel) -> NpuView {
        let budget = self.budget(DeviceKind::Npu);
        let installed: Vec<String> = self
            .profiles
            .iter()
            .filter(|p| matches!(&p.source, ModelSource::Flm { tag } if tag == &m.tag))
            .map(|p| p.id.clone())
            .collect();
        NpuView {
            measured: self.measured_for(&installed),
            installed,
            tag: m.tag.clone(),
            name: m.name.clone(),
            footprint_bytes: m.footprint_bytes,
            default_context: m.default_context,
            capabilities: m.capabilities(),
            single_turn: m.single_turn(),
            unsupported: m.is_transcription(),
            fit: m.footprint_bytes.zip(budget).map(|(need, b)| fit(need, b)),
            downloaded: self.flm_model_dir.as_ref().is_some_and(|d| d.join(&m.name).is_dir()),
        }
    }

    fn flm_model(&self, tag: &str) -> Option<&FlmModel> {
        self.flm.iter().find(|m| m.tag == tag)
    }

    /// FLM tags not linked from any curated model.
    fn unlinked_flm<'a>(&'a self, catalog: &'a Catalog) -> impl Iterator<Item = &'a FlmModel> {
        self.flm.iter().filter(|m| !catalog.models.iter().any(|c| c.flm.contains(&m.tag)))
    }
}

impl QuantView {
    fn with_measured(mut self, ctx: &Context) -> Self {
        self.measured = ctx.measured_for(&self.installed);
        self
    }
}

/// Latest finished speed and tool-calling results per model id.
fn measured(store: &contio_bench::store::Store) -> HashMap<String, Measured> {
    use contio_bench::store::{Filter, RunStatus};
    let mut out: HashMap<String, Measured> = HashMap::new();
    // newest first: the first value seen per (model, metric) wins
    for r in store.list(&Filter::default()).unwrap_or_default().into_iter().filter(|r| r.status == RunStatus::Done) {
        let m = out.entry(r.model.clone()).or_insert_with(|| Measured { model: r.model.clone(), ..Default::default() });
        let get = |k: &str| r.metrics.get(k).copied();
        match r.suite.as_str() {
            "perf" if m.tg_tps.is_none() => {
                m.tg_tps = get("tg_tps");
                m.pp512_tps = get("pp512_tps");
                m.j_per_token = get("j_per_token");
            }
            "tools" if m.tools_pass_rate.is_none() => {
                m.tools_pass_rate = get("tools_pass_rate");
                m.agent_ready = get("agent_ready").map(|v| v >= 1.0);
            }
            _ => {}
        }
    }
    out
}

/// Per-model inputs to the fit estimate.
#[derive(Clone, Copy)]
struct ModelInfo<'a> {
    kind: ModelKind,
    kv: Option<&'a KvGeometry>,
    params: Option<u64>,
    context: u32,
    mmproj_size: u64,
}

fn file_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// Where a Hugging Face file is stored locally.
pub fn hf_path(models_dir: &Path, repo: &str, file: &str) -> PathBuf {
    models_dir.join("hf").join(repo).join(file)
}

/// The quant to install by default: the recommended one if it fits `device`,
/// else the largest one that does.
fn suggested<'a>(views: &'a [QuantView], device: DeviceKind) -> Option<&'a QuantView> {
    let fit_of = |q: &QuantView| match device {
        DeviceKind::Cpu => q.fit.cpu,
        _ => q.fit.gpu,
    };
    let fits = |q: &&QuantView| fit_of(q).is_none_or(|f| f.verdict != Verdict::TooBig);
    views
        .iter()
        .find(|q| q.recommended)
        .filter(fits)
        .or_else(|| views.iter().rev().find(|q| fits(q) && fit_of(q).is_none_or(|f| f.verdict == Verdict::Fits)))
        .or_else(|| views.iter().rev().find(fits))
}

/// `requested` (clamped to 512..=the model's maximum), else `default`.
fn fit_context(requested: Option<u32>, default: u32, max: Option<u32>) -> u32 {
    requested.map_or(default, |c| c.clamp(512, max.unwrap_or(u32::MAX).max(512)))
}

fn detail_for(ctx: &Context, m: &CatalogModel, listing: Result<RepoListing, String>, context: Option<u32>) -> ModelDetail {
    let g = m.geometry.as_ref();
    let (listing, error) = match listing {
        Ok(l) => (l, None),
        Err(e) => (RepoListing::default(), Some(e)),
    };
    let mmproj = listing.preferred_mmproj().cloned();
    let max = g.and_then(|g| g.context_length);
    let info = ModelInfo {
        kind: m.kind,
        kv: g.map(|g| &g.kv),
        params: m.params(),
        context: fit_context(context, m.default_context(), max),
        mmproj_size: mmproj.as_ref().map_or(0, |f| f.size),
    };
    ModelDetail {
        id: Some(m.id.clone()),
        name: m.name.clone(),
        family: m.family.clone(),
        repo: Some(m.repo.clone()),
        kind: m.kind,
        params: info.params,
        active_b: m.active_b,
        context_length: max,
        default_context: m.default_context(),
        fit_context: info.context,
        capabilities: m.capabilities(mmproj.is_some()),
        notes: m.notes.clone(),
        recommended: m.recommended.clone(),
        quants: listing.quants.iter().map(|q| ctx.quant_view(&m.repo, q, info, m.recommended.as_deref())).collect(),
        mmproj,
        npu: m.flm.iter().filter_map(|t| ctx.flm_model(t)).map(|f| ctx.npu_view(f)).collect(),
        error,
    }
}

fn npu_detail(ctx: &Context, f: &FlmModel) -> ModelDetail {
    ModelDetail {
        id: Some(format!("flm:{}", f.tag)),
        name: f.name.clone(),
        family: NPU_FAMILY.into(),
        repo: None,
        kind: if f.is_embedding() { ModelKind::Embedding } else { ModelKind::Chat },
        params: f.size_bytes,
        active_b: None,
        context_length: f.default_context,
        default_context: f.default_context.unwrap_or(8192),
        fit_context: f.default_context.unwrap_or(8192),
        capabilities: f.capabilities(),
        notes: f.repo_url.clone(),
        recommended: None,
        quants: vec![],
        mmproj: None,
        npu: vec![ctx.npu_view(f)],
        error: None,
    }
}

fn summary(d: ModelDetail) -> ModelSummary {
    let installed = d.quants.iter().flat_map(|q| q.installed.clone()).chain(d.npu.iter().flat_map(|n| n.installed.clone()));
    let mut installed: Vec<String> = installed.collect();
    installed.dedup();
    // nothing fits: still show the recommended (or smallest) quant with its ✗
    let suggested_label = suggested(&d.quants, DeviceKind::Gpu)
        .or_else(|| d.quants.iter().find(|q| q.recommended))
        .or_else(|| d.quants.first())
        .map(|q| q.label.clone());
    let mut quants = d.quants;
    let suggested = suggested_label.and_then(|l| quants.iter().position(|q| q.label == l)).map(|i| quants.swap_remove(i));
    ModelSummary {
        id: d.id.unwrap_or_default(),
        name: d.name,
        family: d.family,
        repo: d.repo,
        kind: d.kind,
        params: d.params,
        active_b: d.active_b,
        capabilities: d.capabilities,
        default_context: d.default_context,
        quant_count: quants.len() + usize::from(suggested.is_some()),
        min_size: quants.iter().map(|q| q.size).chain(suggested.as_ref().map(|q| q.size)).min(),
        max_size: quants.iter().map(|q| q.size).chain(suggested.as_ref().map(|q| q.size)).max(),
        suggested,
        npu: d.npu.into_iter().map(|n| n.tag).collect(),
        installed,
        error: d.error,
    }
}

// --- handlers -------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct RefreshQuery {
    #[serde(default)]
    pub refresh: bool,
}

/// All curated models (with their suggested quant and fit) plus NPU-only models.
pub async fn overview(State(state): State<AppState>, Query(q): Query<RefreshQuery>) -> Response {
    let ttl = if q.refresh { std::time::Duration::ZERO } else { LISTING_TTL };
    let svc = state.catalog.clone();
    let mut repos: Vec<String> = svc.catalog.models.iter().map(|m| m.repo.clone()).collect();
    repos.sort_unstable();
    repos.dedup(); // the Whisper models share one repo
    let trees: HashMap<String, Result<Vec<RepoFile>, String>> =
        futures_util::stream::iter(repos)
            .map(|repo| {
                let hf = svc.hf.clone();
                async move {
                    let l = hf.files(&repo, ttl).await;
                    (repo, l)
                }
            })
            .buffer_unordered(8)
            .collect()
            .await;
    let ctx = Context::new(&state);
    let mut models: Vec<ModelSummary> = svc
        .catalog
        .models
        .iter()
        .map(|m| {
            let listing = match trees.get(&m.repo) {
                Some(Ok(files)) => Ok(m.listing(files)),
                Some(Err(e)) => Err(e.clone()),
                None => Err("not listed".into()),
            };
            summary(detail_for(&ctx, m, listing, None))
        })
        .collect();
    let mut families = svc.catalog.families.clone();
    let npu_only: Vec<ModelSummary> = ctx.unlinked_flm(&svc.catalog).map(|f| summary(npu_detail(&ctx, f))).collect();
    if !npu_only.is_empty() {
        families.push(Family {
            id: NPU_FAMILY.into(),
            name: "NPU (FastFlowLM)".into(),
            vendor: "various".into(),
            license: "various".into(),
            summary: "More models built for the XDNA2 NPU; they run next to GPU models at a few watts.".into(),
        });
        models.extend(npu_only);
    }
    Json(Overview { families, models, npu_available: !ctx.flm.is_empty() }).into_response()
}

#[derive(Debug, Default, Deserialize)]
pub struct DetailQuery {
    /// Compute the fits for this context instead of the default.
    #[serde(default)]
    pub context: Option<u32>,
}

/// A curated model (`qwen3-8b`) or an NPU-only one (`flm:<tag>`).
pub async fn model_detail(
    State(state): State<AppState>,
    UrlPath(id): UrlPath<String>,
    Query(q): Query<DetailQuery>,
) -> Response {
    let ctx = Context::new(&state);
    if let Some(tag) = id.strip_prefix("flm:") {
        return match ctx.flm_model(tag) {
            Some(f) => Json(npu_detail(&ctx, f)).into_response(),
            None => not_found(&format!("no NPU model `{tag}` (is FastFlowLM set up?)")),
        };
    }
    let Some(m) = state.catalog.catalog.model(&id) else { return not_found(&format!("`{id}` is not in the catalog")) };
    let listing = state.catalog.hf.files(&m.repo, LISTING_TTL).await.map(|files| m.listing(&files));
    Json(detail_for(&ctx, m, listing, q.context)).into_response()
}

#[derive(Debug, Deserialize)]
pub struct SearchQuery {
    pub q: String,
    #[serde(default)]
    pub limit: Option<u32>,
}

/// GGUF repos on Hugging Face (spec §4.2, "community/unverified").
pub async fn search(State(state): State<AppState>, Query(q): Query<SearchQuery>) -> Response {
    match state.catalog.hf.search(&q.q, q.limit.unwrap_or(30)).await {
        Ok(hits) => Json(hits).into_response(),
        Err(e) => bad_request(&e),
    }
}

#[derive(Debug, Deserialize)]
pub struct RepoQuery {
    pub repo: String,
    #[serde(default)]
    pub context: Option<u32>,
}

/// Detail view for any Hugging Face GGUF repo. Without a GGUF header read, the
/// KV cache size is a conservative guess.
pub async fn repo_detail(State(state): State<AppState>, Query(q): Query<RepoQuery>) -> Response {
    if !valid_repo(&q.repo) {
        return bad_request("repo must be `owner/name`");
    }
    // a curated repo gets the curated view
    if let Some(m) = state.catalog.catalog.models.iter().find(|m| m.repo == q.repo) {
        return model_detail(State(state.clone()), UrlPath(m.id.clone()), Query(DetailQuery { context: q.context })).await;
    }
    let (info, listing) = tokio::join!(state.catalog.hf.info(&q.repo), state.catalog.hf.listing(&q.repo, LISTING_TTL));
    let ctx = Context::new(&state);
    Json(repo_view(&ctx, &q.repo, info, listing, q.context)).into_response()
}

fn repo_view(
    ctx: &Context,
    repo: &str,
    info: Result<RepoInfo, String>,
    listing: Result<RepoListing, String>,
    requested: Option<u32>,
) -> ModelDetail {
    let info = info.ok();
    let (listing, error) = match listing {
        Ok(l) => (l, None),
        Err(e) => (RepoListing::default(), Some(e)),
    };
    let mmproj = listing.preferred_mmproj().cloned();
    let max = info.as_ref().and_then(|i| i.context_length);
    let default = max.unwrap_or(u32::MAX).min(contio_core::model::DEFAULT_CONTEXT);
    let model = ModelInfo {
        kind: ModelKind::Chat,
        kv: None,
        params: info.as_ref().and_then(|i| i.params),
        context: fit_context(requested, default, max),
        mmproj_size: mmproj.as_ref().map_or(0, |f| f.size),
    };
    let mut caps = Vec::new();
    if info.as_ref().is_some_and(|i| i.tools) {
        caps.push("tools".to_owned());
    }
    if info.as_ref().is_some_and(|i| i.reasoning) {
        caps.push("reasoning".to_owned());
    }
    if mmproj.is_some() {
        caps.push("vision".to_owned());
    }
    let kind = match info.as_ref().and_then(|i| i.pipeline.as_deref()) {
        Some("feature-extraction" | "sentence-similarity") => ModelKind::Embedding,
        Some("text-ranking" | "text-classification") => ModelKind::Reranker,
        _ => ModelKind::Chat,
    };
    ModelDetail {
        id: None,
        name: repo.rsplit('/').next().unwrap_or(repo).trim_end_matches("-GGUF").to_owned(),
        family: "community".into(),
        repo: Some(repo.to_owned()),
        kind,
        params: model.params,
        active_b: None,
        context_length: max,
        default_context: default,
        fit_context: model.context,
        capabilities: caps,
        notes: Some("Community upload, not curated by contio.".into()),
        recommended: None,
        quants: listing.quants.iter().map(|q| ctx.quant_view(repo, q, model, None)).collect(),
        mmproj,
        npu: vec![],
        error,
    }
}

// --- install ------------------------------------------------------------------

/// Install a model: a catalog entry (`model`), any Hugging Face GGUF repo
/// (`repo`, with `quant` or a single `file`), or a FastFlowLM tag (`flm`, or
/// `device = "npu"` with a catalog entry that has one).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstallModelRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub repo: Option<String>,
    #[serde(default)]
    pub quant: Option<String>,
    #[serde(default)]
    pub file: Option<String>,
    #[serde(default)]
    pub flm: Option<String>,
    /// gpu (default), cpu or npu.
    #[serde(default)]
    pub device: Option<DeviceKind>,
    #[serde(default)]
    pub context: Option<u32>,
    /// Name in the API; derived from the catalog id if unset.
    #[serde(default)]
    pub id: Option<String>,
    /// Skip the vision projector of multimodal models.
    #[serde(default)]
    pub no_vision: bool,
    /// Start the model once installed (health check).
    #[serde(default)]
    pub load: bool,
}

pub async fn install(State(state): State<AppState>, Json(req): Json<InstallModelRequest>) -> Response {
    match start_install(&state, req) {
        Ok(id) => accepted(id),
        Err(e) => bad_request(&e),
    }
}

/// Model ids: what clients send in the `model` field.
pub fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id.starts_with(|c: char| c.is_ascii_alphanumeric())
        && id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || "._:-".contains(c))
}

/// A relative path inside a repo, without `..` or absolute components.
fn valid_file(file: &str) -> bool {
    file.split('/').all(|seg| !seg.is_empty() && seg != "." && seg != "..") && file.chars().all(|c| !c.is_control())
}

/// Validate and start an install job. `Err` is a client error.
pub fn start_install(state: &AppState, req: InstallModelRequest) -> Result<u64, String> {
    if let Some(id) = &req.id
        && !valid_id(id)
    {
        return Err("id: lowercase letters, digits and . _ : - (max 64)".into());
    }
    if req.file.as_deref().is_some_and(|f| !valid_file(f)) {
        return Err("file must be a relative path inside the repo".into());
    }
    let catalog = state.catalog.catalog.clone();
    let entry = match &req.model {
        Some(m) => Some(catalog.model(m).ok_or_else(|| format!("`{m}` is not in the catalog"))?.clone()),
        None => None,
    };
    let npu = req.flm.is_some() || req.device == Some(DeviceKind::Npu);
    let (title, key) = if npu {
        let tag = req
            .flm
            .clone()
            .or_else(|| entry.as_ref().and_then(|e| e.flm.first().cloned()))
            .ok_or("this model has no NPU (FastFlowLM) version")?;
        if !tag.chars().all(|c| c.is_ascii_alphanumeric() || "-_.:".contains(c)) {
            return Err("invalid FastFlowLM tag".into());
        }
        (format!("{tag} (npu)"), format!("model:flm:{tag}"))
    } else {
        let repo = req.repo.clone().or_else(|| entry.as_ref().map(|e| e.repo.clone())).ok_or("set `model`, `repo` or `flm`")?;
        if !valid_repo(&repo) {
            return Err("repo must be `owner/name`".into());
        }
        if req.device.is_some_and(|d| !matches!(d, DeviceKind::Gpu | DeviceKind::Cpu)) {
            return Err("GGUF models run on gpu or cpu".into());
        }
        let what = req.file.clone().or_else(|| req.quant.clone()).unwrap_or_else(|| "auto".into());
        let device = format!("{:?}", req.device.unwrap_or(DeviceKind::Gpu)).to_lowercase();
        let name = entry.as_ref().map_or(repo.clone(), |e| e.id.clone());
        (format!("{name} {what} ({device})"), format!("model:{repo}:{what}:{device}"))
    };
    let spec = serde_json::to_value(JobSpec::InstallModel(req.clone())).ok();
    let state = state.clone();
    Ok(state.installer.jobs.clone().spawn_with_spec("model", &title, Some(key), spec, move |ctx| async move {
        let profile =
            if npu { install_npu(&state, &req, entry.as_ref(), &ctx).await? } else { install_gguf(&state, &req, entry.as_ref(), &ctx).await? };
        let id = profile.id.clone();
        if req.load {
            ctx.step("starting model");
            drop(state.manager.acquire(&id).await.map_err(|e| e.to_string())?);
        }
        ctx.step("ready");
        Ok(json!({ "model": id, "kind": profile.kind, "device": profile.device, "source": profile.source }))
    }))
}

/// Register a llama.cpp/FLM backend if none is, with the method from settings.
async fn ensure_backend(state: &AppState, kind: BackendKind, ctx: &JobCtx) -> Result<(), String> {
    let (name, method, version) = match kind {
        BackendKind::LlamaCpp => ("llama.cpp", state.registry.settings().llamacpp_method, LLAMACPP_DEFAULT_VERSION),
        BackendKind::Flm => ("FastFlowLM", state.registry.settings().flm_method, FLM_DEFAULT_VERSION),
        BackendKind::Whisper => ("whisper.cpp", state.registry.settings().whisper_method, WHISPER_DEFAULT_VERSION),
    };
    if state.manager.has_backend(name) {
        return Ok(());
    }
    ctx.log(&format!("{name} isn't set up yet; installing it ({method:?})"));
    let sub = ctx.with_prefix(&format!("[{}] ", kind.dir_name()));
    install_and_register(&state.installer, &state.manager, kind, method, version, &sub).await?;
    Ok(())
}

async fn install_gguf(
    state: &AppState,
    req: &InstallModelRequest,
    entry: Option<&CatalogModel>,
    ctx: &JobCtx,
) -> Result<ModelProfile, String> {
    let repo = req.repo.clone().or_else(|| entry.map(|e| e.repo.clone())).expect("validated");
    let device = req.device.unwrap_or(DeviceKind::Gpu);
    ctx.step("looking up files");
    let hf = &state.catalog.hf;
    let listing = match entry {
        Some(e) => e.listing(&hf.files(&repo, LISTING_TTL).await?),
        None => hf.listing(&repo, LISTING_TTL).await?,
    };
    let (label, files) = match (&req.file, &req.quant) {
        (Some(file), _) => match listing.quants.iter().find(|q| q.files.iter().any(|f| &f.path == file)) {
            Some(q) => (q.label.clone(), q.files.clone()),
            None => {
                // a file that isn't a recognisable quant: take it as is
                let all = hf.files(&repo, LISTING_TTL).await?;
                let f = all.into_iter().find(|f| &f.path == file).ok_or_else(|| format!("{file} not found in {repo}"))?;
                (contio_catalog::quant::quant_label(&f.path).unwrap_or_else(|| "custom".into()), vec![f])
            }
        },
        (None, Some(label)) => {
            let q = listing.quant(label).ok_or_else(|| {
                let have: Vec<&str> = listing.quants.iter().map(|q| q.label.as_str()).collect();
                format!("{repo} has no {label} (available: {})", have.join(", "))
            })?;
            (q.label.clone(), q.files.clone())
        }
        (None, None) => {
            let ctx_view = Context::new(state);
            let detail = match entry {
                Some(e) => detail_for(&ctx_view, e, Ok(listing.clone()), None),
                None => repo_view(&ctx_view, &repo, hf.info(&repo).await, Ok(listing.clone()), None),
            };
            let pick = suggested(&detail.quants, device).ok_or("no quantization fits this machine")?.label.clone();
            ctx.log(&format!("picked {pick}"));
            let q = listing.quant(&pick).expect("from the same listing");
            (q.label.clone(), q.files.clone())
        }
    };
    let mmproj = if req.no_vision { None } else { listing.preferred_mmproj().cloned() };
    let speech = entry.is_some_and(|e| e.ggml.is_some());

    ensure_backend(state, if speech { BackendKind::Whisper } else { BackendKind::LlamaCpp }, ctx).await?;

    let downloads: Vec<&RepoFile> = files.iter().chain(mmproj.as_ref()).collect();
    let n = downloads.len();
    let token = std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty());
    let models_dir = state.installer.models_dir();
    for (i, f) in downloads.iter().enumerate() {
        let sub = if n > 1 { ctx.with_prefix(&format!("[{}/{n}] ", i + 1)) } else { ctx.clone() };
        sub.log(&format!("{} ({:.1} GiB)", f.path, f.size as f64 / (1u64 << 30) as f64));
        let remote = Remote {
            url: download::hf_url(&repo, &f.path, "main"),
            sha256: f.sha256.clone(),
            size: Some(f.size),
            token: token.clone(),
        };
        download::fetch(&state.installer.http, &remote, &hf_path(&models_dir, &repo, &f.path), &sub).await?;
    }

    let base = entry.map_or_else(|| repo_base_id(&repo), |e| e.id.clone());
    let path = hf_path(&models_dir, &repo, &files[0].path);
    if speech {
        let mut profile = ModelProfile::new(String::new(), ModelSource::Whisper { path }, device);
        profile.kind = ModelKind::Transcription;
        profile.origin = Some(ModelOrigin { catalog: entry.map(|e| e.id.clone()), repo: Some(repo), quant: Some(label.clone()) });
        return register(state, profile, req.id.as_deref(), &base, &label, ctx);
    }
    let mut profile = ModelProfile::new(
        String::new(),
        ModelSource::Gguf { path, mmproj: mmproj.as_ref().map(|m| hf_path(&models_dir, &repo, &m.path)) },
        device,
    );
    let geometry = entry.and_then(|e| e.geometry.as_ref());
    let max_ctx = geometry.and_then(|g| g.context_length).unwrap_or(u32::MAX);
    profile.kind = entry.map_or(ModelKind::Chat, |e| e.kind);
    profile.context = Some(req.context.unwrap_or_else(|| entry.map_or(contio_core::model::DEFAULT_CONTEXT, |e| e.default_context())).min(max_ctx));
    profile.kv = geometry.map(|g| g.kv);
    profile.origin = Some(ModelOrigin { catalog: entry.map(|e| e.id.clone()), repo: Some(repo), quant: Some(label.clone()) });
    register(state, profile, req.id.as_deref(), &base, &label, ctx)
}

async fn install_npu(
    state: &AppState,
    req: &InstallModelRequest,
    entry: Option<&CatalogModel>,
    ctx: &JobCtx,
) -> Result<ModelProfile, String> {
    let tag = req.flm.clone().or_else(|| entry.and_then(|e| e.flm.first().cloned())).expect("validated");
    ensure_backend(state, BackendKind::Flm, ctx).await?;
    let flm = state.installer.flm().ok_or("FastFlowLM is registered but its install path is unknown")?;
    let listed = Context::new(state).flm.into_iter().find(|m| m.tag == tag);
    let Some(listed) = listed else { return Err(format!("FastFlowLM has no model `{tag}` for this NPU")) };
    if listed.is_transcription() {
        return Err(format!(
            "FastFlowLM runs {tag} only next to an NPU LLM: set `asr = true` under [backends.flm] and add an alias \
             with upstream_model = \"{tag}\", or install the whisper.cpp version for GPU/CPU"
        ));
    }
    ctx.step("flm pull");
    let mut cmd = FlmBackend::new(flm).flm_command().map_err(|e| e.to_string())?;
    cmd.args(["pull", &tag]);
    contio_install::run_logged(ctx, &mut cmd).await?;

    let mut profile = ModelProfile::new(String::new(), ModelSource::Flm { tag: tag.clone() }, DeviceKind::Npu);
    profile.kind = if listed.is_embedding() { ModelKind::Embedding } else { ModelKind::Chat };
    profile.context = Some(req.context.or(listed.default_context).unwrap_or(8192));
    profile.origin = Some(ModelOrigin { catalog: entry.map(|e| e.id.clone()), repo: listed.repo_url, quant: listed.quant });
    let base = entry.map_or_else(|| tag.replace([':', '.'], "-"), |e| e.id.clone());
    register(state, profile, req.id.as_deref(), &format!("{base}-npu"), &tag, ctx)
}

/// `unsloth/Qwen3-8B-GGUF` → `qwen3-8b`.
fn repo_base_id(repo: &str) -> String {
    let name = repo.rsplit('/').next().unwrap_or(repo).to_ascii_lowercase();
    let name = name.trim_end_matches("-gguf").trim_end_matches("_gguf");
    name.chars().map(|c| if c.is_ascii_alphanumeric() || ".-".contains(c) { c } else { '-' }).collect()
}

/// Add the profile under `wanted` or a free id derived from `base`
/// (`base`, `base-cpu`, `base-<quant>`, `base-<quant>-2`, …). Installing the
/// same files for the same device again returns the existing model.
fn register(
    state: &AppState,
    mut profile: ModelProfile,
    wanted: Option<&str>,
    base: &str,
    variant: &str,
    ctx: &JobCtx,
) -> Result<ModelProfile, String> {
    let existing = state.manager.profiles();
    if let Some(same) = existing.iter().find(|p| p.source == profile.source && p.device == profile.device) {
        ctx.log(&format!("already registered as `{}`", same.id));
        return Ok(same.clone());
    }
    let taken = |id: &str| existing.iter().any(|p| p.id == id) || state.aliases.contains(id);
    profile.id = match wanted {
        Some(id) if taken(id) => return Err(format!("a model named `{id}` already exists")),
        Some(id) => id.to_owned(),
        None => {
            let base = match profile.device {
                DeviceKind::Cpu => format!("{base}-cpu"),
                _ => base.to_owned(),
            };
            let variant: String = variant.to_ascii_lowercase().replace([':', '/'], "-");
            let with_variant = format!("{base}-{variant}");
            [base.clone(), with_variant.clone()]
                .into_iter()
                .chain((2..).map(|n| format!("{with_variant}-{n}")))
                .find(|id| !taken(id))
                .expect("infinite candidates")
        }
    };
    state.manager.add_profile(profile.clone(), true).map_err(|e| e.to_string())?;
    state.registry.save_models(&state.manager);
    ctx.log(&format!("registered as `{}`", profile.id));
    Ok(profile)
}

// --- remove / update ----------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct RemoveQuery {
    #[serde(default)]
    pub delete_files: bool,
}

/// Unregister a model added through the API, optionally deleting its files
/// (only inside contio's model directory and only if no other model uses them).
pub async fn remove_model(
    State(state): State<AppState>,
    UrlPath(id): UrlPath<String>,
    Query(q): Query<RemoveQuery>,
) -> Response {
    let removed = match state.manager.remove_profile(&id).await {
        Ok(p) => p,
        Err(e) => return crate::gateway::backend_error(&e),
    };
    state.registry.save_models(&state.manager);
    let aliases = state.aliases.remove_for_model(&id);
    let deleted = if q.delete_files { delete_files(&state.manager, &state.installer.models_dir(), &removed) } else { vec![] };
    Json(json!({ "removed": id, "deleted_files": deleted, "removed_aliases": aliases })).into_response()
}

fn delete_files(manager: &ModelManager, models_dir: &Path, removed: &ModelProfile) -> Vec<PathBuf> {
    let files = match &removed.source {
        ModelSource::Gguf { path, mmproj } => shard_paths(path).into_iter().chain(mmproj.clone()).collect(),
        ModelSource::Whisper { path } => vec![path.clone()],
        ModelSource::Flm { .. } => return vec![],
    };
    let in_use = |p: &Path| {
        manager.profiles().iter().any(|o| match &o.source {
            ModelSource::Gguf { path: a, mmproj: b } => a == p || b.as_deref() == Some(p),
            ModelSource::Whisper { path: a } => a == p,
            ModelSource::Flm { .. } => false,
        })
    };
    let mut deleted = Vec::new();
    for f in files {
        if f.starts_with(models_dir) && !in_use(&f) && std::fs::remove_file(&f).is_ok() {
            tracing::info!(file = %f.display(), "deleted");
            deleted.push(f);
        }
    }
    deleted
}

/// `x-00001-of-00003.gguf` → all three shards; otherwise just `path`.
fn shard_paths(path: &Path) -> Vec<PathBuf> {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
    let parsed = name.strip_suffix(".gguf").and_then(|s| s.rsplit_once("-of-")).and_then(|(rest, total)| {
        let (prefix, index) = rest.rsplit_once('-')?;
        (index == "00001").then_some((prefix.to_owned(), total.parse::<u32>().ok()?))
    });
    match parsed {
        Some((prefix, total)) => {
            (1..=total).map(|i| path.with_file_name(format!("{prefix}-{i:05}-of-{total:05}.gguf"))).collect()
        }
        None => vec![path.to_owned()],
    }
}

/// Launch settings that can be changed on an installed model. Also the shape
/// of a `config.toml` model's saved overrides (`overrides.json`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelPatch {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pinned: Option<bool>,
    /// Seconds; 0 = use the daemon default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idle_ttl_secs: Option<u64>,
    /// gpu or cpu (GGUF models).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<DeviceKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_args: Option<Vec<String>>,
}

impl ModelPatch {
    /// Check the values against `p`, then apply them.
    pub fn apply(&self, p: &mut ModelProfile) -> Result<(), String> {
        if let Some(c) = self.context {
            if !(512..=4 << 20).contains(&c) {
                return Err("context must be between 512 and 4M tokens".into());
            }
            p.context = Some(c);
        }
        if let Some(pinned) = self.pinned {
            p.pinned = pinned;
        }
        if let Some(t) = self.idle_ttl_secs {
            p.idle_ttl_secs = (t > 0).then_some(t);
        }
        if let Some(d) = self.device {
            let local = !matches!(p.source, ModelSource::Flm { .. });
            if !local || !matches!(d, DeviceKind::Gpu | DeviceKind::Cpu) {
                return Err("only GGUF and Whisper models can move, and only between gpu and cpu".into());
            }
            p.device = d;
        }
        if let Some(a) = &self.extra_args {
            p.extra_args = a.clone();
        }
        Ok(())
    }

    /// Later changes win, field by field.
    pub fn merge(&mut self, newer: &ModelPatch) {
        self.context = newer.context.or(self.context);
        self.pinned = newer.pinned.or(self.pinned);
        self.idle_ttl_secs = newer.idle_ttl_secs.or(self.idle_ttl_secs);
        self.device = newer.device.or(self.device);
        self.extra_args = newer.extra_args.clone().or(self.extra_args.take());
    }
}

pub async fn update_model(
    State(state): State<AppState>,
    UrlPath(id): UrlPath<String>,
    Json(patch): Json<ModelPatch>,
) -> Response {
    let mut p = match state.manager.profile(&id) {
        Ok(p) => p,
        Err(e) => return crate::gateway::backend_error(&e),
    };
    if let Err(e) = patch.apply(&mut p) {
        return bad_request(&e);
    }
    if let Err(e) = state.manager.update_profile(p).await {
        return crate::gateway::backend_error(&e);
    }
    let managed = state.manager.managed_profiles().iter().any(|m| m.id == id);
    if managed {
        state.registry.save_models(&state.manager);
    } else if let Err(e) = state.registry.add_override(&id, &patch) {
        tracing::error!("saving overrides: {e}");
    }
    Json(crate::api::statuses(&state).into_iter().find(|s| s.id == id)).into_response()
}

/// `DELETE /api/models/{id}/overrides`: back to the `config.toml` settings.
pub async fn reset_overrides(State(state): State<AppState>, UrlPath(id): UrlPath<String>) -> Response {
    let Some(original) = state.registry.config_profile(&id) else {
        return bad_request(&format!("`{id}` isn't a config.toml model"));
    };
    if let Err(e) = state.manager.update_profile(original).await {
        return crate::gateway::backend_error(&e);
    }
    if let Err(e) = state.registry.clear_override(&id) {
        tracing::error!("saving overrides: {e}");
    }
    Json(crate::api::statuses(&state).into_iter().find(|s| s.id == id)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_and_paths() {
        assert_eq!(repo_base_id("unsloth/Qwen3-8B-GGUF"), "qwen3-8b");
        assert_eq!(repo_base_id("bartowski/Llama_3.2+X"), "llama-3.2-x");
        assert!(valid_id("qwen3.6-35b-a3b") && valid_id("a:b_c"));
        for bad in ["", "-x", "UPPER", "a b", &"x".repeat(65)] {
            assert!(!valid_id(bad), "{bad}");
        }
        assert!(valid_file("Q4_K_M/m-00001-of-00002.gguf"));
        for bad in ["../etc/passwd", "/etc/passwd", "a/../../b", "", "a/./b", "x\u{0}y"] {
            assert!(!valid_file(bad), "{bad:?}");
        }
        let shards = shard_paths(Path::new("/m/x-00001-of-00003.gguf"));
        assert_eq!(shards.len(), 3);
        assert_eq!(shards[2], Path::new("/m/x-00003-of-00003.gguf"));
        assert_eq!(shard_paths(Path::new("/m/x.gguf")), [Path::new("/m/x.gguf")]);
    }

    fn view(label: &str, recommended: bool, gpu: Verdict) -> QuantView {
        let f = Fit { verdict: gpu, needed_bytes: 0, budget_bytes: 0 };
        QuantView {
            label: label.into(),
            size: 0,
            files: 1,
            bits_per_weight: None,
            tier: None,
            fit: DeviceFits { gpu: Some(f), cpu: Some(Fit { verdict: Verdict::TooBig, ..f }) },
            recommended,
            downloaded: false,
            installed: vec![],
            measured: vec![],
        }
    }

    #[test]
    fn suggestion_prefers_recommended_then_largest_comfortable_fit() {
        use Verdict::*;
        let v = [view("Q2", false, Fits), view("Q4", true, Fits), view("Q6", false, Tight), view("Q8", false, TooBig)];
        assert_eq!(suggested(&v, DeviceKind::Gpu).unwrap().label, "Q4");
        let v = [view("Q2", false, Fits), view("Q3", false, Fits), view("Q4", true, TooBig), view("Q6", false, Tight)];
        assert_eq!(suggested(&v, DeviceKind::Gpu).unwrap().label, "Q3", "comfortable fit before tight");
        let v = [view("Q6", false, Tight), view("Q8", true, TooBig)];
        assert_eq!(suggested(&v, DeviceKind::Gpu).unwrap().label, "Q6", "tight beats nothing");
        assert!(suggested(&v, DeviceKind::Cpu).is_none(), "nothing fits the cpu pool");
    }
}
