//! Decides which models are running: on-demand loading, per-pool memory
//! budgets with LRU eviction, idle unloading, and request leases (spec §7.2).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use contio_core::hw::{GpuApi, MemoryPoolKind};
use contio_core::model::{ModelProfile, ModelSource, ModelState, ModelStatus};

use crate::instance::{Instance, free_port};
use crate::{Backend, BackendError};

#[derive(Debug, Clone)]
pub struct ManagerConfig {
    /// One `<model id>.log` per backend process.
    pub log_dir: PathBuf,
    /// Max time from spawn to healthy (large models load for ~20 s, m0-npu.md).
    pub startup_timeout: Duration,
    /// Unload unpinned models idle for longer than this (`None` = never).
    pub default_idle_ttl: Option<Duration>,
    /// Bytes contio may use per pool. Pools without an entry are unbudgeted.
    pub pool_budgets: HashMap<MemoryPoolKind, u64>,
}

/// Cheap to clone; all clones share state.
#[derive(Clone)]
pub struct ModelManager {
    inner: Arc<Inner>,
}

struct Inner {
    config: ManagerConfig,
    backends: RwLock<Vec<Arc<dyn Backend>>>,
    profiles: RwLock<Vec<Registered>>,
    slots: Mutex<HashMap<String, Slot>>,
    /// Serializes loads/unloads/evictions; requests to running models never wait on it.
    ops: tokio::sync::Mutex<()>,
    http: reqwest::Client,
    epoch: Instant,
    /// GPU API for GGUF models on the GPU that don't pick one.
    default_gpu_api: RwLock<GpuApi>,
}

struct Registered {
    profile: ModelProfile,
    /// Added through the API (persisted by the daemon, removable); `false` for
    /// models from `config.toml`.
    managed: bool,
}

/// Estimated bytes per memory pool; empty if unknown.
type Estimate = Vec<(MemoryPoolKind, u64)>;

fn in_pool(e: &Estimate, pool: MemoryPoolKind) -> u64 {
    e.iter().filter(|(p, _)| *p == pool).map(|(_, b)| b).sum()
}

struct Slot {
    state: SlotState,
    backend: &'static str,
    estimate: Estimate,
    in_flight: Arc<AtomicU32>,
    /// Milliseconds since `Inner::epoch`.
    last_used: Arc<AtomicU64>,
}

enum SlotState {
    Loading,
    Ready(Instance),
    Failed(String),
}

/// Proof that a request is being served by a running model. While any lease is
/// alive the model is neither evicted nor unloaded for idleness.
#[derive(Debug)]
pub struct Lease {
    pub base_url: String,
    pub upstream_model: String,
    /// `Backend::name` of the instance serving the request.
    pub backend: &'static str,
    in_flight: Arc<AtomicU32>,
    last_used: Arc<AtomicU64>,
    epoch: Instant,
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.last_used.store(millis_since(self.epoch), Ordering::Relaxed);
        self.in_flight.fetch_sub(1, Ordering::AcqRel);
    }
}

fn millis_since(epoch: Instant) -> u64 {
    epoch.elapsed().as_millis() as u64
}

const MIB: u64 = 1024 * 1024;

impl ModelManager {
    /// `profiles` are the fixed ones from the config file; see [`Self::add_profile`].
    pub fn new(config: ManagerConfig, backends: Vec<Arc<dyn Backend>>, profiles: Vec<ModelProfile>) -> Self {
        let profiles = profiles.into_iter().map(|profile| Registered { profile, managed: false }).collect();
        Self {
            inner: Arc::new(Inner {
                config,
                backends: RwLock::new(backends),
                profiles: RwLock::new(profiles),
                slots: Mutex::new(HashMap::new()),
                ops: tokio::sync::Mutex::new(()),
                http: reqwest::Client::new(),
                epoch: Instant::now(),
                default_gpu_api: RwLock::default(),
            }),
        }
    }

    pub fn profiles(&self) -> Vec<ModelProfile> {
        self.read_profiles().iter().map(|r| r.profile.clone()).collect()
    }

    /// Profiles added through the API, for persisting them.
    pub fn managed_profiles(&self) -> Vec<ModelProfile> {
        self.read_profiles().iter().filter(|r| r.managed).map(|r| r.profile.clone()).collect()
    }

    pub fn profile(&self, id: &str) -> Result<ModelProfile, BackendError> {
        self.read_profiles()
            .iter()
            .find(|r| r.profile.id == id)
            .map(|r| r.profile.clone())
            .ok_or_else(|| BackendError::UnknownModel(id.to_owned()))
    }

    /// Register a model at runtime (`managed`: may later be updated/removed).
    pub fn add_profile(&self, profile: ModelProfile, managed: bool) -> Result<(), BackendError> {
        let mut profiles = self.write_profiles();
        if profiles.iter().any(|r| r.profile.id == profile.id) {
            return Err(BackendError::DuplicateModel(profile.id));
        }
        tracing::info!(model = %profile.id, "model registered");
        profiles.push(Registered { profile, managed });
        Ok(())
    }

    /// Replace a managed profile. A running instance is stopped so the next
    /// request starts it with the new settings.
    pub async fn update_profile(&self, profile: ModelProfile) -> Result<(), BackendError> {
        let _ops = self.inner.ops.lock().await;
        {
            let mut profiles = self.write_profiles();
            let r = profiles
                .iter_mut()
                .find(|r| r.profile.id == profile.id)
                .ok_or_else(|| BackendError::UnknownModel(profile.id.clone()))?;
            // config.toml models too: the daemon saves their changes as overrides
            r.profile = profile.clone();
        }
        self.stop_slot(&profile.id).await;
        Ok(())
    }

    /// Unregister a managed model, stopping it first.
    pub async fn remove_profile(&self, id: &str) -> Result<ModelProfile, BackendError> {
        let _ops = self.inner.ops.lock().await;
        let removed = {
            let mut profiles = self.write_profiles();
            let i = profiles
                .iter()
                .position(|r| r.profile.id == id)
                .ok_or_else(|| BackendError::UnknownModel(id.to_owned()))?;
            if !profiles[i].managed {
                return Err(BackendError::NotManaged(id.to_owned()));
            }
            profiles.remove(i).profile
        };
        self.stop_slot(id).await;
        tracing::info!(model = id, "model removed");
        Ok(removed)
    }

    /// Register a backend at runtime (e.g. right after it was installed).
    /// It takes precedence over existing backends for the formats it supports.
    pub fn add_backend(&self, backend: Arc<dyn Backend>) {
        tracing::info!(backend = %backend.describe(), "backend registered");
        self.inner.backends.write().expect("backends lock poisoned").insert(0, backend);
    }

    /// Unregister the backends `remove` matches. Loaded models keep running
    /// until [`Self::restart_gpu_models`] or their next start.
    pub fn remove_backends(&self, remove: impl Fn(&dyn Backend) -> bool) {
        self.inner.backends.write().expect("backends lock poisoned").retain(|b| {
            let gone = remove(b.as_ref());
            if gone {
                tracing::info!(backend = %b.describe(), "backend removed");
            }
            !gone
        });
    }

    /// Stop loaded GPU models running on an `api` build; the next request
    /// starts them on whichever build is registered first now.
    pub async fn restart_gpu_models(&self, api: GpuApi) {
        let _ops = self.inner.ops.lock().await;
        let default = self.default_gpu_api();
        let affected: Vec<String> = self
            .profiles()
            .into_iter()
            .filter(|p| p.uses_gpu_api() && p.gpu_api.unwrap_or(default) == api)
            .map(|p| p.id)
            .collect();
        for id in affected {
            self.stop_slot(&id).await;
        }
    }

    pub fn default_gpu_api(&self) -> GpuApi {
        *self.inner.default_gpu_api.read().expect("gpu api lock poisoned")
    }

    /// Change the default GPU API. Loaded models that follow the default and
    /// now run elsewhere are stopped; the next request starts them on the new one.
    pub async fn set_default_gpu_api(&self, api: GpuApi) {
        let _ops = self.inner.ops.lock().await;
        let old = std::mem::replace(&mut *self.inner.default_gpu_api.write().expect("gpu api lock poisoned"), api);
        if old == api {
            return;
        }
        tracing::info!(%api, "default GPU API changed");
        let affected: Vec<String> =
            self.profiles().into_iter().filter(|p| p.uses_gpu_api() && p.gpu_api.is_none()).map(|p| p.id).collect();
        for id in affected {
            self.stop_slot(&id).await;
        }
    }

    /// Whether a llama.cpp build for `api` is registered.
    pub fn has_gpu_build(&self, api: GpuApi) -> bool {
        self.inner.backends.read().expect("backends lock poisoned").iter().any(|b| b.gpu_api() == Some(api))
    }

    /// Whether a backend with this [`Backend::name`] is registered.
    pub fn has_backend(&self, name: &str) -> bool {
        self.inner.backends.read().expect("backends lock poisoned").iter().any(|b| b.name() == name)
    }

    /// Bytes contio may use per pool (fit estimates use the same numbers).
    pub fn pool_budgets(&self) -> &HashMap<MemoryPoolKind, u64> {
        &self.inner.config.pool_budgets
    }

    /// Descriptions of the registered backends, in priority order.
    pub fn backend_names(&self) -> Vec<String> {
        self.inner.backends.read().expect("backends lock poisoned").iter().map(|b| b.describe()).collect()
    }

    /// Process id of the running server for `id`.
    pub fn pid(&self, id: &str) -> Option<u32> {
        match &self.lock_slots().get(id)?.state {
            SlotState::Ready(i) => i.pid(),
            _ => None,
        }
    }

    /// The backend that runs `id` (name, how it runs), e.g. for benchmark records.
    pub fn backend_description(&self, id: &str) -> Result<(String, String), BackendError> {
        let b = self.backend_for(&self.profile(id)?)?;
        Ok((b.name().to_owned(), b.describe()))
    }

    /// Get a lease on a running instance of `id`, loading it (and evicting
    /// others from its pool) if needed.
    pub async fn acquire(&self, id: &str) -> Result<Lease, BackendError> {
        let profile = self.profile(id)?;
        if let Some(lease) = self.try_lease(id) {
            return Ok(lease);
        }

        let _ops = self.inner.ops.lock().await;
        if let Some(lease) = self.try_lease(id) {
            return Ok(lease); // loaded by a concurrent request while we waited
        }

        let backend = self.backend_for(&profile)?;
        let estimate = backend.estimate_pools(&profile);
        for victim in self.make_room(&profile, &estimate)? {
            tracing::info!(model = %victim.0, "evicting to make room for {id}");
            victim.1.stop().await;
        }

        self.set_slot(id, SlotState::Loading, backend.name(), estimate.clone());
        let port = free_port()?;
        let cmd = match backend.command(&profile, port) {
            Ok(cmd) => cmd,
            Err(e) => {
                self.set_slot(id, SlotState::Failed(e.to_string()), backend.name(), estimate.clone());
                return Err(e);
            }
        };
        tracing::info!(model = id, backend = backend.name(), port, "starting");
        let log_path = self.inner.config.log_dir.join(format!("{id}.log"));
        let started = Instant::now();
        match Instance::spawn(
            cmd,
            port,
            backend.health_path(),
            &log_path,
            self.inner.config.startup_timeout,
            &self.inner.http,
        )
        .await
        {
            Ok(instance) => {
                tracing::info!(model = id, secs = started.elapsed().as_secs_f32(), "ready");
                self.set_slot(id, SlotState::Ready(instance), backend.name(), estimate.clone());
                Ok(self.try_lease(id).expect("just marked ready"))
            }
            Err(reason) => {
                tracing::warn!(model = id, "failed to start: {reason}");
                self.set_slot(id, SlotState::Failed(reason.clone()), backend.name(), estimate);
                Err(BackendError::StartFailed { model: id.to_owned(), reason })
            }
        }
    }

    /// Stop `id` if running. An explicit user action: takes effect immediately,
    /// even with requests in flight.
    pub async fn unload(&self, id: &str) -> Result<bool, BackendError> {
        self.profile(id)?;
        let _ops = self.inner.ops.lock().await;
        Ok(self.stop_slot(id).await)
    }

    /// Forget `id`'s slot, stopping it if running. Caller holds `ops`.
    async fn stop_slot(&self, id: &str) -> bool {
        let taken = self.lock_slots().remove(id);
        match taken {
            Some(Slot { state: SlotState::Ready(instance), .. }) => {
                tracing::info!(model = id, "unloading");
                instance.stop().await;
                true
            }
            _ => false,
        }
    }

    /// Unload idle models and forget crashed ones. Call periodically.
    pub async fn reap(&self) {
        let Ok(_ops) = self.inner.ops.try_lock() else {
            return; // a load is in progress; try again next tick
        };
        let now = millis_since(self.inner.epoch);
        let mut stopped = Vec::new();
        let profiles = self.profiles();
        {
            let mut slots = self.lock_slots();
            for profile in &profiles {
                let Some(slot) = slots.get_mut(&profile.id) else { continue };
                let SlotState::Ready(instance) = &mut slot.state else { continue };
                if !instance.is_alive() {
                    tracing::warn!(model = %profile.id, log = %instance.log_path().display(), "backend exited");
                    slot.state = SlotState::Failed("backend process exited".into());
                    continue;
                }
                let ttl = profile
                    .idle_ttl_secs
                    .map(Duration::from_secs)
                    .or(self.inner.config.default_idle_ttl);
                let idle = Duration::from_millis(now.saturating_sub(slot.last_used.load(Ordering::Relaxed)));
                if !profile.pinned
                    && slot.in_flight.load(Ordering::Acquire) == 0
                    && ttl.is_some_and(|ttl| idle >= ttl)
                    && let Some(Slot { state: SlotState::Ready(i), .. }) = slots.remove(&profile.id)
                {
                    stopped.push((profile.id.clone(), i));
                }
            }
        }
        for (id, instance) in stopped {
            tracing::info!(model = %id, "unloading idle model");
            instance.stop().await;
        }
    }

    /// Stop every backend (daemon shutdown).
    pub async fn shutdown(&self) {
        let _ops = self.inner.ops.lock().await;
        let all: Vec<_> = self.lock_slots().drain().collect();
        for (id, slot) in all {
            if let SlotState::Ready(instance) = slot.state {
                tracing::info!(model = %id, "stopping");
                instance.stop().await;
            }
        }
    }

    pub fn statuses(&self) -> Vec<ModelStatus> {
        let profiles = self.read_profiles();
        let slots = self.lock_slots();
        profiles
            .iter()
            .map(|r| {
                let p = &r.profile;
                let backend = self.backend_for(p).ok();
                let slot = slots.get(&p.id);
                let estimate = slot
                    .map(|s| s.estimate.clone())
                    .filter(|e| !e.is_empty())
                    .or_else(|| backend.as_ref().map(|b| b.estimate_pools(p)))
                    .unwrap_or_default();
                ModelStatus {
                    id: p.id.clone(),
                    device: p.device,
                    gpu_api: p.gpu_api,
                    gpu_api_used: p.uses_gpu_api().then(|| p.gpu_api.unwrap_or(self.default_gpu_api())),
                    kind: p.kind,
                    context: p.context,
                    managed: r.managed,
                    overridden: false,
                    origin: p.origin.clone(),
                    source: p.source.clone(),
                    pool: p.pool(),
                    backend: backend.as_ref().map(|b| b.name().to_owned()),
                    pinned: p.pinned,
                    idle_ttl_secs: p.idle_ttl_secs,
                    extra_args: p.extra_args.clone(),
                    speculative: p.speculative.clone(),
                    vocab_size: match &p.source {
                        ModelSource::Gguf { path, .. } => crate::gguf::vocab_size(path),
                        _ => None,
                    },
                    tuning: p.tuning,
                    has_mtp: matches!(&p.source, ModelSource::Gguf { path, .. } if crate::gguf::has_mtp(path)),
                    moe_layers: match &p.source {
                        ModelSource::Gguf { path, .. } => {
                            crate::gguf::expert_bytes_per_layer(path).iter().filter(|&&b| b > 0).count() as u32
                        }
                        _ => 0,
                    },
                    estimated_bytes: (!estimate.is_empty()).then(|| estimate.iter().map(|(_, b)| b).sum()),
                    offloaded_bytes: (p.pool() == MemoryPoolKind::Gpu)
                        .then(|| in_pool(&estimate, MemoryPoolKind::System))
                        .filter(|&b| b > 0),
                    state: match slot.map(|s| &s.state) {
                        None => ModelState::Stopped,
                        Some(SlotState::Loading) => ModelState::Loading,
                        Some(SlotState::Ready(i)) => ModelState::Ready { port: i.port() },
                        Some(SlotState::Failed(e)) => ModelState::Failed { error: e.clone() },
                    },
                    in_flight: slot.map_or(0, |s| s.in_flight.load(Ordering::Relaxed)),
                }
            })
            .collect()
    }

    // --- internals ---------------------------------------------------------------

    fn read_profiles(&self) -> std::sync::RwLockReadGuard<'_, Vec<Registered>> {
        self.inner.profiles.read().expect("profiles lock poisoned")
    }

    fn write_profiles(&self) -> std::sync::RwLockWriteGuard<'_, Vec<Registered>> {
        self.inner.profiles.write().expect("profiles lock poisoned")
    }

    fn backend_for(&self, profile: &ModelProfile) -> Result<Arc<dyn Backend>, BackendError> {
        // backends see the default GPU API filled in
        let profile: std::borrow::Cow<'_, ModelProfile> = if profile.uses_gpu_api() && profile.gpu_api.is_none() {
            let mut p = profile.clone();
            p.gpu_api = Some(self.default_gpu_api());
            std::borrow::Cow::Owned(p)
        } else {
            std::borrow::Cow::Borrowed(profile)
        };
        let found =
            self.inner.backends.read().expect("backends lock poisoned").iter().find(|b| b.supports(&profile)).cloned();
        found.ok_or_else(|| match profile.gpu_api {
            Some(api) if profile.uses_gpu_api() && self.has_backend("llama.cpp") => {
                BackendError::MissingGpuBuild { model: profile.id.clone(), api }
            }
            _ => BackendError::Unsupported(profile.id.clone()),
        })
    }

    fn lock_slots(&self) -> std::sync::MutexGuard<'_, HashMap<String, Slot>> {
        self.inner.slots.lock().expect("model slot mutex poisoned")
    }

    // Lock order everywhere: profiles before slots.
    fn try_lease(&self, id: &str) -> Option<Lease> {
        let profile = self.profile(id).ok()?;
        let upstream_model = self.backend_for(&profile).ok()?.upstream_model(&profile);
        let mut slots = self.lock_slots();
        let slot = slots.get_mut(id)?;
        let SlotState::Ready(instance) = &mut slot.state else { return None };
        if !instance.is_alive() {
            slot.state = SlotState::Failed("backend process exited".into());
            return None;
        }
        slot.in_flight.fetch_add(1, Ordering::AcqRel);
        slot.last_used.store(millis_since(self.inner.epoch), Ordering::Relaxed);
        Some(Lease {
            base_url: format!("http://127.0.0.1:{}", instance.port()),
            upstream_model,
            backend: slot.backend,
            in_flight: slot.in_flight.clone(),
            last_used: slot.last_used.clone(),
            epoch: self.inner.epoch,
        })
    }

    fn set_slot(&self, id: &str, state: SlotState, backend: &'static str, estimate: Estimate) {
        let mut slots = self.lock_slots();
        let now = millis_since(self.inner.epoch);
        let slot = slots.entry(id.to_owned()).or_insert_with(|| Slot {
            state: SlotState::Loading,
            backend,
            estimate: estimate.clone(),
            in_flight: Arc::new(AtomicU32::new(0)),
            last_used: Arc::new(AtomicU64::new(now)),
        });
        slot.state = state;
        slot.backend = backend;
        slot.estimate = estimate;
        slot.last_used.store(now, Ordering::Relaxed);
    }

    /// Pick LRU victims so `estimate` fits in every pool it needs (usually
    /// one; MoE offload uses both), and remove them from the slot table.
    /// Errors without touching anything if even evicting every candidate
    /// would not be enough.
    fn make_room(&self, profile: &ModelProfile, estimate: &Estimate) -> Result<Vec<(String, Instance)>, BackendError> {
        let pinned: Vec<String> =
            self.read_profiles().iter().filter(|r| r.profile.pinned).map(|r| r.profile.id.clone()).collect();
        let mut slots = self.lock_slots();
        // (last_used, id) of models that may go, oldest first
        let mut candidates: Vec<(u64, String)> = slots
            .iter()
            .filter(|(id, slot)| {
                *id != &profile.id
                    && matches!(slot.state, SlotState::Ready(_))
                    && !pinned.contains(id)
                    && slot.in_flight.load(Ordering::Acquire) == 0
            })
            .map(|(id, slot)| (slot.last_used.load(Ordering::Relaxed), id.clone()))
            .collect();
        candidates.sort_unstable();

        let mut evict: Vec<String> = Vec::new();
        for &(pool, need) in estimate {
            let Some(&budget) = self.inner.config.pool_budgets.get(&pool) else { continue };
            let usage = |slots: &HashMap<String, Slot>, skip: &[String]| -> u64 {
                slots
                    .iter()
                    .filter(|(id, s)| *id != &profile.id && !skip.contains(id) && !matches!(s.state, SlotState::Failed(_)))
                    .map(|(_, s)| in_pool(&s.estimate, pool))
                    .sum()
            };
            let mut used = usage(&slots, &evict);
            for (_, id) in &candidates {
                if used + need <= budget {
                    break;
                }
                let bytes = slots.get(id).map_or(0, |s| in_pool(&s.estimate, pool));
                if bytes > 0 && !evict.contains(id) {
                    evict.push(id.clone());
                    used -= bytes;
                }
            }
            if used + need > budget {
                return Err(BackendError::InsufficientMemory {
                    model: profile.id.clone(),
                    pool: format!("{pool:?}").to_lowercase(),
                    needed_mib: need / MIB,
                    available_mib: budget.saturating_sub(used) / MIB,
                });
            }
        }
        let mut victims = Vec::new();
        for id in evict {
            if let Some(Slot { state: SlotState::Ready(i), .. }) = slots.remove(&id) {
                victims.push((id, i));
            }
        }
        Ok(victims)
    }
}
