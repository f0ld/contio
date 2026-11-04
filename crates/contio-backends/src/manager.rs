//! Decides which models are running: on-demand loading, per-pool memory
//! budgets with LRU eviction, idle unloading, and request leases (spec §7.2).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use contio_core::hw::MemoryPoolKind;
use contio_core::model::{ModelProfile, ModelState, ModelStatus};

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
}

struct Registered {
    profile: ModelProfile,
    /// Added through the API (persisted by the daemon, removable); `false` for
    /// models from `config.toml`.
    managed: bool,
}

struct Slot {
    state: SlotState,
    backend: &'static str,
    estimate: Option<u64>,
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
            if !r.managed {
                return Err(BackendError::NotManaged(profile.id));
            }
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
        let estimate = backend.estimate_bytes(&profile);
        for victim in self.make_room(&profile, estimate)? {
            tracing::info!(model = %victim.0, "evicting to make room for {id}");
            victim.1.stop().await;
        }

        self.set_slot(id, SlotState::Loading, backend.name(), estimate);
        let port = free_port()?;
        let cmd = match backend.command(&profile, port) {
            Ok(cmd) => cmd,
            Err(e) => {
                self.set_slot(id, SlotState::Failed(e.to_string()), backend.name(), estimate);
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
                self.set_slot(id, SlotState::Ready(instance), backend.name(), estimate);
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
                ModelStatus {
                    id: p.id.clone(),
                    device: p.device,
                    kind: p.kind,
                    context: p.context,
                    managed: r.managed,
                    origin: p.origin.clone(),
                    source: p.source.clone(),
                    pool: p.pool(),
                    backend: backend.as_ref().map(|b| b.name().to_owned()),
                    pinned: p.pinned,
                    idle_ttl_secs: p.idle_ttl_secs,
                    extra_args: p.extra_args.clone(),
                    estimated_bytes: slot
                        .and_then(|s| s.estimate)
                        .or_else(|| backend.and_then(|b| b.estimate_bytes(p))),
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
        self.inner
            .backends
            .read()
            .expect("backends lock poisoned")
            .iter()
            .find(|b| b.supports(profile))
            .cloned()
            .ok_or_else(|| BackendError::Unsupported(profile.id.clone()))
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
            in_flight: slot.in_flight.clone(),
            last_used: slot.last_used.clone(),
            epoch: self.inner.epoch,
        })
    }

    fn set_slot(&self, id: &str, state: SlotState, backend: &'static str, estimate: Option<u64>) {
        let mut slots = self.lock_slots();
        let now = millis_since(self.inner.epoch);
        let slot = slots.entry(id.to_owned()).or_insert_with(|| Slot {
            state: SlotState::Loading,
            backend,
            estimate,
            in_flight: Arc::new(AtomicU32::new(0)),
            last_used: Arc::new(AtomicU64::new(now)),
        });
        slot.state = state;
        slot.backend = backend;
        slot.estimate = estimate;
        slot.last_used.store(now, Ordering::Relaxed);
    }

    /// Pick LRU victims in `profile`'s pool so `estimate` fits, and remove them
    /// from the slot table. Errors without touching anything if even evicting
    /// every candidate would not be enough.
    fn make_room(
        &self,
        profile: &ModelProfile,
        estimate: Option<u64>,
    ) -> Result<Vec<(String, Instance)>, BackendError> {
        let pool = profile.pool();
        let (Some(need), Some(&budget)) = (estimate, self.inner.config.pool_budgets.get(&pool)) else {
            return Ok(Vec::new());
        };
        let (in_pool, pinned): (Vec<String>, Vec<String>) = {
            let profiles = self.read_profiles();
            let ids = |f: &dyn Fn(&ModelProfile) -> bool| {
                profiles.iter().filter(|r| f(&r.profile)).map(|r| r.profile.id.clone()).collect()
            };
            (ids(&|p| p.pool() == pool), ids(&|p| p.pinned))
        };
        let in_pool = |id: &str| in_pool.iter().any(|p| p == id);
        let mut slots = self.lock_slots();
        let mut used = 0u64;
        let mut candidates: Vec<(u64, u64, String)> = Vec::new(); // (last_used, bytes, id)
        for (id, slot) in slots.iter() {
            if id == &profile.id || !in_pool(id) || matches!(slot.state, SlotState::Failed(_)) {
                continue;
            }
            let bytes = slot.estimate.unwrap_or(0);
            used += bytes;
            let evictable = matches!(slot.state, SlotState::Ready(_))
                && !pinned.contains(id)
                && slot.in_flight.load(Ordering::Acquire) == 0;
            if evictable {
                candidates.push((slot.last_used.load(Ordering::Relaxed), bytes, id.clone()));
            }
        }
        if used + need <= budget {
            return Ok(Vec::new());
        }
        let reclaimable: u64 = candidates.iter().map(|c| c.1).sum();
        if used - reclaimable + need > budget {
            return Err(BackendError::InsufficientMemory {
                model: profile.id.clone(),
                pool: format!("{pool:?}").to_lowercase(),
                needed_mib: need / MIB,
                available_mib: budget.saturating_sub(used - reclaimable) / MIB,
            });
        }

        candidates.sort_unstable();
        let mut victims = Vec::new();
        for (_, bytes, id) in candidates {
            if used + need <= budget {
                break;
            }
            if let Some(Slot { state: SlotState::Ready(i), .. }) = slots.remove(&id) {
                used -= bytes;
                victims.push((id, i));
            }
        }
        Ok(victims)
    }
}
