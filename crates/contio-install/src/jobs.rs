//! Background jobs with progress, logs and cancellation (spec §6).
//!
//! Jobs live in memory; downloads resume from their `.part` files after a
//! restart, so nothing is lost except the job record (persistence: M2).

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::Value;

const LOG_LINES: usize = 200;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum JobState {
    Running,
    Done,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Progress {
    pub done: u64,
    pub total: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
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
}

/// Handed to a running job to report progress and observe cancellation.
#[derive(Clone)]
pub struct JobCtx {
    id: u64,
    jobs: Jobs,
    cancelled: Arc<AtomicBool>,
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or_default()
}

impl Jobs {
    /// Start `work` as a job. If `key` is set and a running job has the same
    /// key (e.g. the same download target), that job's id is returned instead,
    /// so there is never more than one writer per target.
    pub fn spawn<F, Fut>(&self, kind: &str, title: &str, key: Option<String>, work: F) -> u64
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
                    key,
                },
            );
            id
        };
        let cancelled = Arc::new(AtomicBool::new(false));
        self.inner.cancel.lock().expect("cancel map poisoned").insert(id, cancelled.clone());
        let ctx = JobCtx { id, jobs: self.clone(), cancelled: cancelled.clone() };
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
        });
        id
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

impl JobCtx {
    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn step(&self, step: &str) {
        tracing::info!(job = self.id, "{step}");
        let step = step.to_owned();
        self.jobs.update(self.id, |j| {
            j.step = Some(step);
            j.progress = None;
        });
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
}
