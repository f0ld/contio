//! Background jobs with progress, logs and cancellation (spec §6).
//!
//! With [`Jobs::persistent`] the job list is kept in a JSON file. Each job can
//! carry the request that started it (`spec`), so jobs cut off by a daemon
//! restart can be started again; downloads then resume from their `.part`
//! files.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

const LOG_LINES: usize = 200;
/// Finished jobs kept in the history file.
const HISTORY: usize = 100;
pub const INTERRUPTED: &str = "interrupted by a daemon restart";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobState {
    Running,
    Done,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Progress {
    pub done: u64,
    pub total: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobInfo {
    pub id: u64,
    pub kind: String,
    pub title: String,
    pub state: JobState,
    /// Current step, e.g. "downloading", "building".
    pub step: Option<String>,
    pub progress: Option<Progress>,
    pub error: Option<String>,
    pub result: Option<Value>,
    pub started_at: u64,
    pub finished_at: Option<u64>,
    /// Last lines of tool output (builds, podman pull).
    pub log: VecDeque<String>,
    /// The request that started the job, to retry or resume it.
    #[serde(default)]
    pub spec: Option<Value>,
    /// Set when this job was retried/resumed as another job.
    #[serde(default)]
    pub retried_as: Option<u64>,
    #[serde(skip)]
    key: Option<String>,
}

#[derive(Clone, Default)]
pub struct Jobs {
    inner: Arc<Inner>,
}

#[derive(Default)]
struct Inner {
    jobs: Mutex<BTreeMap<u64, JobInfo>>,
    cancel: Mutex<HashMap<u64, Arc<AtomicBool>>>,
    next: AtomicU64,
    /// Where the job list is saved; `None` keeps it in memory only.
    store: Option<PathBuf>,
    /// One writer of the store file at a time.
    saving: Mutex<()>,
}

/// Handed to a running job to report progress and observe cancellation.
#[derive(Clone)]
pub struct JobCtx {
    id: u64,
    jobs: Jobs,
    cancelled: Arc<AtomicBool>,
    /// Prepended to step names, e.g. `[2/3] ` for multi-file downloads.
    prefix: Option<Arc<str>>,
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or_default()
}

impl Jobs {
    /// Jobs saved to `path`. Returns the jobs that were still running when the
    /// daemon stopped (now marked failed with [`INTERRUPTED`]) so the caller
    /// can start them again from their `spec`.
    pub fn persistent(path: PathBuf) -> (Self, Vec<JobInfo>) {
        let mut jobs: BTreeMap<u64, JobInfo> = match std::fs::read(&path) {
            Ok(bytes) => serde_json::from_slice::<Vec<JobInfo>>(&bytes)
                .map(|v| v.into_iter().map(|j| (j.id, j)).collect())
                .unwrap_or_else(|e| {
                    tracing::warn!("ignoring unreadable job history {}: {e}", path.display());
                    BTreeMap::new()
                }),
            Err(_) => BTreeMap::new(),
        };
        let mut interrupted = Vec::new();
        for j in jobs.values_mut().filter(|j| j.state == JobState::Running) {
            j.state = JobState::Failed;
            j.error = Some(INTERRUPTED.into());
            j.finished_at = Some(now());
            interrupted.push(j.clone());
        }
        let next = jobs.keys().max().copied().unwrap_or(0);
        let me = Self {
            inner: Arc::new(Inner {
                jobs: Mutex::new(jobs),
                cancel: Default::default(),
                next: AtomicU64::new(next),
                store: Some(path),
                saving: Mutex::new(()),
            }),
        };
        me.save();
        (me, interrupted)
    }

    /// Start `work` as a job. If `key` is set and a running job has the same
    /// key (e.g. the same download target), that job's id is returned instead,
    /// so there is never more than one writer per target.
    pub fn spawn<F, Fut>(&self, kind: &str, title: &str, key: Option<String>, work: F) -> u64
    where
        F: FnOnce(JobCtx) -> Fut + Send + 'static,
        Fut: Future<Output = Result<Value, String>> + Send + 'static,
    {
        self.spawn_with_spec(kind, title, key, None, work)
    }

    /// [`Self::spawn`], remembering the request (`spec`) for retries.
    pub fn spawn_with_spec<F, Fut>(
        &self,
        kind: &str,
        title: &str,
        key: Option<String>,
        spec: Option<Value>,
        work: F,
    ) -> u64
    where
        F: FnOnce(JobCtx) -> Fut + Send + 'static,
        Fut: Future<Output = Result<Value, String>> + Send + 'static,
    {
        let id = {
            let mut jobs = self.lock();
            if let Some(k) = &key
                && let Some(running) =
                    jobs.values().find(|j| j.state == JobState::Running && j.key.as_ref() == Some(k))
            {
                return running.id;
            }
            let id = self.inner.next.fetch_add(1, Ordering::Relaxed) + 1;
            jobs.insert(
                id,
                JobInfo {
                    id,
                    kind: kind.to_owned(),
                    title: title.to_owned(),
                    state: JobState::Running,
                    step: None,
                    progress: None,
                    error: None,
                    result: None,
                    started_at: now(),
                    finished_at: None,
                    log: VecDeque::new(),
                    spec,
                    retried_as: None,
                    key,
                },
            );
            id
        };
        self.save();
        let cancelled = Arc::new(AtomicBool::new(false));
        self.inner.cancel.lock().expect("cancel map poisoned").insert(id, cancelled.clone());
        let ctx = JobCtx { id, jobs: self.clone(), cancelled: cancelled.clone(), prefix: None };
        let jobs = self.clone();
        tracing::info!(id, kind, title, "job started");
        tokio::spawn(async move {
            let result = work(ctx).await;
            let mut map = jobs.lock();
            let Some(job) = map.get_mut(&id) else { return };
            job.finished_at = Some(now());
            match result {
                Ok(v) => {
                    job.state = JobState::Done;
                    job.result = Some(v);
                    tracing::info!(id, "job done");
                }
                Err(e) if cancelled.load(Ordering::Relaxed) => {
                    job.state = JobState::Cancelled;
                    job.error = Some(e);
                    tracing::info!(id, "job cancelled");
                }
                Err(e) => {
                    tracing::warn!(id, "job failed: {e}");
                    job.state = JobState::Failed;
                    job.error = Some(e);
                }
            }
            drop(map);
            jobs.inner.cancel.lock().expect("cancel map poisoned").remove(&id);
            jobs.save();
        });
        id
    }

    /// Record that `old` was retried as `new`.
    pub fn mark_retried(&self, old: u64, new: u64) {
        self.update(old, |j| j.retried_as = Some(new));
        self.save();
    }

    /// Write the job list (running jobs + recent history) to the store.
    fn save(&self) {
        let Some(path) = &self.inner.store else { return };
        let _saving = self.inner.saving.lock().expect("save mutex poisoned");
        let snapshot: Vec<JobInfo> = {
            let jobs = self.lock();
            let mut finished = jobs.values().filter(|j| j.state != JobState::Running).count();
            jobs.values()
                .filter(|j| {
                    let keep = j.state == JobState::Running || finished <= HISTORY;
                    if j.state != JobState::Running {
                        finished -= 1; // oldest first: drop until HISTORY remain
                    }
                    keep
                })
                .cloned()
                .collect()
        };
        if let Err(e) = write_json(path, &snapshot) {
            tracing::warn!("saving jobs to {}: {e}", path.display());
        }
    }

    pub fn list(&self) -> Vec<JobInfo> {
        self.lock().values().rev().cloned().collect()
    }

    pub fn get(&self, id: u64) -> Option<JobInfo> {
        self.lock().get(&id).cloned()
    }

    /// Request cancellation; the job stops at its next checkpoint.
    pub fn cancel(&self, id: u64) -> bool {
        match self.inner.cancel.lock().expect("cancel map poisoned").get(&id) {
            Some(flag) => {
                flag.store(true, Ordering::Relaxed);
                true
            }
            None => false,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, BTreeMap<u64, JobInfo>> {
        self.inner.jobs.lock().expect("jobs mutex poisoned")
    }

    fn update(&self, id: u64, f: impl FnOnce(&mut JobInfo)) {
        if let Some(j) = self.lock().get_mut(&id) {
            f(j);
        }
    }
}

fn write_json(path: &Path, value: &impl Serialize) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, serde_json::to_vec(value).map_err(std::io::Error::other)?)?;
    std::fs::rename(&tmp, path)
}

impl JobCtx {
    pub fn id(&self) -> u64 {
        self.id
    }

    /// A handle whose step names start with `prefix`.
    pub fn with_prefix(&self, prefix: &str) -> Self {
        Self { prefix: Some(prefix.into()), ..self.clone() }
    }

    pub fn step(&self, step: &str) {
        let step = match &self.prefix {
            Some(p) => format!("{p}{step}"),
            None => step.to_owned(),
        };
        tracing::info!(job = self.id, "{step}");
        self.jobs.update(self.id, |j| {
            j.step = Some(step);
            j.progress = None;
        });
        self.jobs.save();
    }

    pub fn progress(&self, done: u64, total: Option<u64>) {
        self.jobs.update(self.id, |j| j.progress = Some(Progress { done, total }));
    }

    pub fn log(&self, line: &str) {
        let line = line.to_owned();
        self.jobs.update(self.id, |j| {
            j.log.push_back(line);
            while j.log.len() > LOG_LINES {
                j.log.pop_front();
            }
        });
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    pub fn check_cancelled(&self) -> Result<(), String> {
        if self.is_cancelled() { Err("cancelled".into()) } else { Ok(()) }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    async fn wait(jobs: &Jobs, id: u64) -> JobInfo {
        for _ in 0..200 {
            let j = jobs.get(id).unwrap();
            if j.state != JobState::Running {
                return j;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("job {id} did not finish");
    }

    #[tokio::test]
    async fn runs_reports_and_finishes() {
        let jobs = Jobs::default();
        let id = jobs.spawn("test", "t", None, |ctx| async move {
            ctx.step("working");
            ctx.progress(5, Some(10));
            ctx.log("line");
            Ok(serde_json::json!({"ok": true}))
        });
        let j = wait(&jobs, id).await;
        assert_eq!(j.state, JobState::Done);
        assert_eq!(j.result.unwrap()["ok"], true);
        assert_eq!(j.progress, Some(Progress { done: 5, total: Some(10) }));
        assert_eq!(j.log, ["line"]);
        assert!(j.finished_at.is_some());
    }

    #[tokio::test]
    async fn same_key_joins_running_job() {
        let jobs = Jobs::default();
        let slow = |_ctx: JobCtx| async {
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok(Value::Null)
        };
        let a = jobs.spawn("dl", "x", Some("/models/a".into()), slow);
        let b = jobs.spawn("dl", "x", Some("/models/a".into()), slow);
        let c = jobs.spawn("dl", "y", Some("/models/c".into()), slow);
        assert_eq!(a, b);
        assert_ne!(a, c);
        wait(&jobs, a).await;
        let d = jobs.spawn("dl", "x", Some("/models/a".into()), slow);
        assert_ne!(a, d, "finished jobs don't block new ones");
    }

    #[tokio::test]
    async fn cancellation_and_failure() {
        let jobs = Jobs::default();
        let id = jobs.spawn("t", "loop", None, |ctx| async move {
            loop {
                ctx.check_cancelled()?;
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        });
        assert!(jobs.cancel(id));
        assert_eq!(wait(&jobs, id).await.state, JobState::Cancelled);
        assert!(!jobs.cancel(id), "finished jobs can't be cancelled");

        let id = jobs.spawn("t", "fail", None, |_| async { Err("boom".into()) });
        let j = wait(&jobs, id).await;
        assert_eq!((j.state, j.error.as_deref()), (JobState::Failed, Some("boom")));
    }

    #[tokio::test]
    async fn persists_and_reports_interrupted_jobs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("jobs.json");
        let (jobs, interrupted) = Jobs::persistent(path.clone());
        assert!(interrupted.is_empty());
        let done = jobs.spawn_with_spec("t", "done", None, Some(serde_json::json!({"n": 1})), |ctx| async move {
            ctx.with_prefix("[1/2] ").step("downloading");
            Ok(Value::Null)
        });
        wait(&jobs, done).await;
        assert_eq!(jobs.get(done).unwrap().step.as_deref(), Some("[1/2] downloading"));
        let running = jobs.spawn_with_spec("t", "slow", None, Some(serde_json::json!({"n": 2})), |_| async {
            tokio::time::sleep(Duration::from_secs(60)).await;
            Ok(Value::Null)
        });
        drop(jobs); // "daemon stops" with one job running

        let (jobs, interrupted) = Jobs::persistent(path.clone());
        assert_eq!(interrupted.len(), 1);
        assert_eq!((interrupted[0].id, interrupted[0].spec.clone()), (running, Some(serde_json::json!({"n": 2}))));
        let j = jobs.get(running).unwrap();
        assert_eq!((j.state, j.error.as_deref()), (JobState::Failed, Some(INTERRUPTED)));
        assert_eq!(jobs.get(done).unwrap().state, JobState::Done);
        let next = jobs.spawn("t", "new", None, |_| async { Ok(Value::Null) });
        assert!(next > running, "ids continue after a restart");
        jobs.mark_retried(running, next);
        wait(&jobs, next).await;

        let (jobs, interrupted) = Jobs::persistent(path);
        assert!(interrupted.is_empty(), "already handled");
        assert_eq!(jobs.get(running).unwrap().retried_as, Some(next));
    }

    #[test]
    fn history_is_capped() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("jobs.json");
        let (jobs, _) = Jobs::persistent(path.clone());
        {
            let mut map = jobs.lock();
            for id in 1..=HISTORY as u64 + 5 {
                let mut j = map.values().next().cloned().unwrap_or_else(|| JobInfo {
                    id: 0,
                    kind: "t".into(),
                    title: "t".into(),
                    state: JobState::Done,
                    step: None,
                    progress: None,
                    error: None,
                    result: None,
                    started_at: 0,
                    finished_at: None,
                    log: VecDeque::new(),
                    spec: None,
                    retried_as: None,
                    key: None,
                });
                j.id = id;
                map.insert(id, j);
            }
        }
        jobs.save();
        let (jobs, _) = Jobs::persistent(path);
        let ids: Vec<u64> = jobs.list().iter().map(|j| j.id).collect();
        assert_eq!(ids.len(), HISTORY);
        assert_eq!(ids.last(), Some(&6), "oldest dropped");
    }
}
