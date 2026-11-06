//! ModelManager against real child processes (`contio-fake-backend`).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use contio_backends::{Backend, BackendError, ManagerConfig, ModelManager};
use contio_core::hw::{DeviceKind, MemoryPoolKind};
use contio_core::model::{ModelProfile, ModelSource, ModelState};
use tokio::process::Command;

const GIB: u64 = 1024 * 1024 * 1024;

/// Runs the fake server; estimates are exactly the (sparse) model file size.
struct FakeBackend {
    env: HashMap<String, Vec<(&'static str, &'static str)>>,
}

impl Backend for FakeBackend {
    fn name(&self) -> &'static str {
        "fake"
    }
    fn supports(&self, p: &ModelProfile) -> bool {
        matches!(p.source, ModelSource::Gguf { .. })
    }
    fn command(&self, p: &ModelProfile, port: u16) -> Result<Command, BackendError> {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_contio-fake-backend"));
        cmd.args(["--port", &port.to_string()]);
        cmd.envs(self.env.get(&p.id).cloned().unwrap_or_default());
        Ok(cmd)
    }
    fn health_path(&self) -> &'static str {
        "/health"
    }
    fn upstream_model(&self, p: &ModelProfile) -> String {
        format!("upstream-{}", p.id)
    }
    fn estimate_bytes(&self, p: &ModelProfile) -> Option<u64> {
        let ModelSource::Gguf { path, .. } = &p.source else { return None };
        Some(std::fs::metadata(path).ok()?.len())
    }
}

struct Fixture {
    _dir: tempfile::TempDir,
    manager: ModelManager,
}

/// `models`: (id, device, size GiB, pinned)
fn fixture(
    models: &[(&str, DeviceKind, u64, bool)],
    gpu_budget_gib: u64,
    idle_ttl: Option<Duration>,
    env: HashMap<String, Vec<(&'static str, &'static str)>>,
) -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let profiles = models
        .iter()
        .map(|&(id, device, gib, pinned)| {
            let path = dir.path().join(format!("{id}.gguf"));
            std::fs::File::create(&path).unwrap().set_len(gib * GIB).unwrap(); // sparse
            let mut p = ModelProfile::new(id, ModelSource::Gguf { path, mmproj: None }, device);
            p.pinned = pinned;
            p
        })
        .collect();
    let config = ManagerConfig {
        log_dir: dir.path().join("logs"),
        startup_timeout: Duration::from_secs(10),
        default_idle_ttl: idle_ttl,
        pool_budgets: HashMap::from([(MemoryPoolKind::Gpu, gpu_budget_gib * GIB)]),
    };
    let manager = ModelManager::new(config, vec![Arc::new(FakeBackend { env })], profiles);
    Fixture { _dir: dir, manager }
}

fn state(m: &ModelManager, id: &str) -> ModelState {
    m.statuses().into_iter().find(|s| s.id == id).unwrap().state
}

async fn healthy(base_url: &str) -> bool {
    reqwest::get(format!("{base_url}/health")).await.is_ok_and(|r| r.status().is_success())
}

#[tokio::test]
async fn loads_on_demand_and_reuses_instance() {
    let f = fixture(&[("a", DeviceKind::Gpu, 1, false)], 10, None, HashMap::new());
    assert_eq!(state(&f.manager, "a"), ModelState::Stopped);

    let lease = f.manager.acquire("a").await.unwrap();
    assert!(healthy(&lease.base_url).await);
    assert_eq!(lease.upstream_model, "upstream-a");
    let first = lease.base_url.clone();
    drop(lease);

    let again = f.manager.acquire("a").await.unwrap();
    assert_eq!(again.base_url, first, "second acquire must not respawn");
    assert!(matches!(state(&f.manager, "a"), ModelState::Ready { .. }));
    drop(again);
    f.manager.shutdown().await;
    assert!(!healthy(&first).await);
}

#[tokio::test]
async fn concurrent_acquires_spawn_once() {
    let mut env = HashMap::new();
    env.insert("a".to_owned(), vec![("FAKE_STARTUP_MS", "300")]);
    let f = fixture(&[("a", DeviceKind::Gpu, 1, false)], 10, None, env);
    let m = &f.manager;
    let (r1, r2, r3) = tokio::join!(m.acquire("a"), m.acquire("a"), m.acquire("a"));
    let urls = [r1.unwrap().base_url.clone(), r2.unwrap().base_url.clone(), r3.unwrap().base_url.clone()];
    assert!(urls.iter().all(|u| u == &urls[0]));
    m.shutdown().await;
}

#[tokio::test]
async fn evicts_least_recently_used_in_same_pool_only() {
    let f = fixture(
        &[
            ("a", DeviceKind::Gpu, 4, false),
            ("b", DeviceKind::Gpu, 4, false),
            ("c", DeviceKind::Gpu, 4, false),
            ("cpu", DeviceKind::Cpu, 4, false),
        ],
        10,
        None,
        HashMap::new(),
    );
    let m = &f.manager;
    let a = m.acquire("a").await.unwrap().base_url.clone();
    tokio::time::sleep(Duration::from_millis(5)).await;
    drop(m.acquire("b").await.unwrap());
    drop(m.acquire("cpu").await.unwrap()); // other pool: never a victim

    drop(m.acquire("c").await.unwrap()); // 4+4+4 > 10 → evict LRU = a

    assert_eq!(state(m, "a"), ModelState::Stopped);
    assert!(!healthy(&a).await, "evicted process must be stopped");
    for id in ["b", "c", "cpu"] {
        assert!(matches!(state(m, id), ModelState::Ready { .. }), "{id}");
    }
    m.shutdown().await;
}

#[tokio::test]
async fn pinned_and_in_flight_models_block_eviction() {
    let f = fixture(
        &[("pinned", DeviceKind::Gpu, 4, true), ("busy", DeviceKind::Gpu, 4, false), ("new", DeviceKind::Gpu, 4, false)],
        10,
        None,
        HashMap::new(),
    );
    let m = &f.manager;
    drop(m.acquire("pinned").await.unwrap());
    let busy = m.acquire("busy").await.unwrap(); // held = request in flight

    let err = m.acquire("new").await.unwrap_err();
    assert!(
        matches!(err, BackendError::InsufficientMemory { needed_mib: 4096, available_mib: 2048, .. }),
        "{err}"
    );
    assert!(healthy(&busy.base_url).await, "nothing may be evicted on failure");

    drop(busy); // request finished → now evictable
    drop(m.acquire("new").await.unwrap());
    assert_eq!(state(m, "busy"), ModelState::Stopped);
    assert!(matches!(state(m, "pinned"), ModelState::Ready { .. }));
    m.shutdown().await;
}

#[tokio::test]
async fn model_larger_than_pool_is_rejected() {
    let f = fixture(&[("huge", DeviceKind::Gpu, 12, false)], 10, None, HashMap::new());
    let err = f.manager.acquire("huge").await.unwrap_err();
    assert!(matches!(err, BackendError::InsufficientMemory { .. }), "{err}");
}

#[tokio::test]
async fn startup_failure_reports_log_tail() {
    let mut env = HashMap::new();
    env.insert("bad".to_owned(), vec![("FAKE_EXIT", "1")]);
    let f = fixture(&[("bad", DeviceKind::Gpu, 1, false)], 10, None, env);
    let err = f.manager.acquire("bad").await.unwrap_err();
    let BackendError::StartFailed { reason, .. } = &err else { panic!("{err}") };
    assert!(reason.contains("simulated startup failure"), "{reason}");
    assert!(matches!(state(&f.manager, "bad"), ModelState::Failed { .. }));
}

#[tokio::test]
async fn idle_models_are_reaped_but_not_while_leased() {
    let f = fixture(&[("a", DeviceKind::Gpu, 1, false)], 10, Some(Duration::ZERO), HashMap::new());
    let m = &f.manager;
    let lease = m.acquire("a").await.unwrap();
    m.reap().await;
    assert!(healthy(&lease.base_url).await, "in-flight model must survive reaping");

    let url = lease.base_url.clone();
    drop(lease);
    m.reap().await;
    assert_eq!(state(m, "a"), ModelState::Stopped);
    assert!(!healthy(&url).await);
}

#[tokio::test]
async fn unload_and_unknown_models() {
    let f = fixture(&[("a", DeviceKind::Gpu, 1, false)], 10, None, HashMap::new());
    let m = &f.manager;
    assert!(!m.unload("a").await.unwrap(), "not running");
    let url = m.acquire("a").await.unwrap().base_url.clone();
    assert!(m.unload("a").await.unwrap());
    assert!(!healthy(&url).await);
    assert!(matches!(m.acquire("nope").await, Err(BackendError::UnknownModel(_))));
}

#[tokio::test]
async fn runtime_registration() {
    let f = fixture(&[("cfg", DeviceKind::Gpu, 1, false)], 10, None, HashMap::new());
    let m = &f.manager;
    let path = f._dir.path().join("cfg.gguf");
    let new = ModelProfile::new("api", ModelSource::Gguf { path, mmproj: None }, DeviceKind::Gpu);

    m.add_profile(new.clone(), true).unwrap();
    assert!(matches!(m.add_profile(new.clone(), true), Err(BackendError::DuplicateModel(_))));
    assert_eq!(m.managed_profiles(), vec![new.clone()]);
    let st = m.statuses();
    assert!(!st[0].managed && st[1].managed);

    // config models can be retuned through the API, not removed
    let cfg = m.profile("cfg").unwrap();
    m.update_profile(cfg).await.unwrap(); // launch settings of config models can change (saved as overrides)
    assert!(matches!(m.remove_profile("cfg").await, Err(BackendError::NotManaged(_))));

    // updating a running model restarts it with the new settings
    let first = m.acquire("api").await.unwrap().base_url.clone();
    let mut changed = new.clone();
    changed.context = Some(4096);
    m.update_profile(changed).await.unwrap();
    assert_eq!(state(m, "api"), ModelState::Stopped);
    assert!(!healthy(&first).await, "old instance stopped");
    assert_eq!(m.profile("api").unwrap().context, Some(4096));

    m.acquire("api").await.unwrap();
    let removed = m.remove_profile("api").await.unwrap();
    assert_eq!(removed.id, "api");
    assert!(matches!(m.acquire("api").await, Err(BackendError::UnknownModel(_))));
    assert!(m.managed_profiles().is_empty());
    m.shutdown().await;
}
