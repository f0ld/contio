//! Benchmark results in SQLite (spec §8.3): one row per run with its full
//! report as JSON, plus the headline metrics as rows for sorting/filtering.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RunStatus {
    Running,
    Done,
    Failed,
    Cancelled,
}

impl RunStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Done => "done",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    fn parse(s: &str) -> Self {
        match s {
            "running" => Self::Running,
            "done" => Self::Done,
            "cancelled" => Self::Cancelled,
            _ => Self::Failed,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Run {
    pub id: i64,
    /// `perf`, later `tools`, `coding`, …
    pub suite: String,
    /// API model id at the time of the run.
    pub model: String,
    pub preset: String,
    /// Unix seconds.
    pub started_at: i64,
    pub finished_at: Option<i64>,
    pub status: RunStatus,
    pub error: Option<String>,
    /// Device, backend, quant, context, files: what exactly was measured.
    pub target: Value,
    pub conditions: Value,
    /// The full report; `Null` in listings.
    pub report: Value,
    pub metrics: BTreeMap<String, f64>,
    /// Background job running it, while it runs.
    pub job_id: Option<u64>,
}

/// What a new run is about.
#[derive(Debug, Clone)]
pub struct NewRun<'a> {
    pub suite: &'a str,
    pub model: &'a str,
    pub preset: &'a str,
    pub target: &'a Value,
    pub conditions: &'a Value,
    pub job_id: Option<u64>,
}

#[derive(Debug, Clone, Default)]
pub struct Filter {
    pub model: Option<String>,
    pub suite: Option<String>,
    pub limit: Option<u32>,
}

#[derive(Clone)]
pub struct Store {
    conn: Arc<Mutex<Connection>>,
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY,
    suite       TEXT NOT NULL,
    model       TEXT NOT NULL,
    preset      TEXT NOT NULL,
    started_at  INTEGER NOT NULL,
    finished_at INTEGER,
    status      TEXT NOT NULL,
    error       TEXT,
    target      TEXT NOT NULL,
    conditions  TEXT NOT NULL,
    report      TEXT,
    job_id      INTEGER
);
CREATE INDEX IF NOT EXISTS runs_model ON runs(model);
CREATE TABLE IF NOT EXISTS metrics (
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    name   TEXT NOT NULL,
    value  REAL NOT NULL,
    PRIMARY KEY (run_id, name)
);
CREATE TABLE IF NOT EXISTS arena_votes (
    id         INTEGER PRIMARY KEY,
    created_at INTEGER NOT NULL,
    prompt     TEXT NOT NULL,
    model_a    TEXT NOT NULL,
    model_b    TEXT NOT NULL,
    -- a, b, tie, both_bad
    winner     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS usage_daily (
    day               TEXT NOT NULL,   -- YYYY-MM-DD, local time
    model             TEXT NOT NULL,
    requests          INTEGER NOT NULL DEFAULT 0,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    audio_seconds     REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (day, model)
);
PRAGMA user_version = 2;
";

/// API traffic of one model (a day, a period, or since the daemon started).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct UsageCounts {
    pub model: String,
    pub requests: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub audio_seconds: f64,
}

impl UsageCounts {
    pub fn add(&mut self, o: &UsageCounts) {
        self.requests += o.requests;
        self.prompt_tokens += o.prompt_tokens;
        self.completion_tokens += o.completion_tokens;
        self.audio_seconds += o.audio_seconds;
    }
}

/// One blind side-by-side comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Vote {
    #[serde(default)]
    pub id: i64,
    #[serde(default)]
    pub created_at: i64,
    pub prompt: String,
    pub model_a: String,
    pub model_b: String,
    pub winner: Winner,
    /// Model that judged; `None` for a human vote.
    #[serde(default)]
    pub judge: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Winner {
    A,
    B,
    Tie,
    BothBad,
}

impl Winner {
    fn as_str(self) -> &'static str {
        match self {
            Self::A => "a",
            Self::B => "b",
            Self::Tie => "tie",
            Self::BothBad => "both_bad",
        }
    }

    fn parse(s: &str) -> Self {
        match s {
            "a" => Self::A,
            "b" => Self::B,
            "both_bad" => Self::BothBad,
            _ => Self::Tie,
        }
    }
}

type Result<T> = std::result::Result<T, String>;

fn err(e: impl std::fmt::Display) -> String {
    format!("benchmark store: {e}")
}

fn now() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or_default()
}

impl Store {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(err)?;
        }
        Self::init(Connection::open(path).map_err(err)?)
    }

    pub fn in_memory() -> Result<Self> {
        Self::init(Connection::open_in_memory().map_err(err)?)
    }

    fn init(conn: Connection) -> Result<Self> {
        conn.execute_batch("PRAGMA foreign_keys = ON; PRAGMA journal_mode = WAL;").map_err(err)?;
        conn.execute_batch(SCHEMA).map_err(err)?;
        // v3: judge votes (added to databases created before)
        let has_judge = conn
            .prepare("SELECT judge FROM arena_votes LIMIT 0")
            .is_ok();
        if !has_judge {
            conn.execute_batch("ALTER TABLE arena_votes ADD COLUMN judge TEXT; PRAGMA user_version = 3;").map_err(err)?;
        }
        Ok(Self { conn: Arc::new(Mutex::new(conn)) })
    }

    fn conn(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().expect("store lock poisoned")
    }

    /// Runs left `running` by a previous daemon become `failed`. Returns how many.
    pub fn fail_interrupted(&self) -> Result<usize> {
        self.conn()
            .execute(
                "UPDATE runs SET status = 'failed', error = 'interrupted by a daemon restart', finished_at = ?1
                 WHERE status = 'running'",
                params![now()],
            )
            .map_err(err)
    }

    pub fn start(&self, r: &NewRun<'_>) -> Result<i64> {
        let c = self.conn();
        c.execute(
            "INSERT INTO runs (suite, model, preset, started_at, status, target, conditions, job_id)
             VALUES (?1, ?2, ?3, ?4, 'running', ?5, ?6, ?7)",
            params![r.suite, r.model, r.preset, now(), r.target.to_string(), r.conditions.to_string(), r.job_id.map(|j| j as i64)],
        )
        .map_err(err)?;
        Ok(c.last_insert_rowid())
    }

    pub fn finish(
        &self,
        id: i64,
        status: RunStatus,
        error: Option<&str>,
        report: &Value,
        metrics: &BTreeMap<String, f64>,
    ) -> Result<()> {
        let mut c = self.conn();
        let tx = c.transaction().map_err(err)?;
        tx.execute(
            "UPDATE runs SET status = ?2, error = ?3, report = ?4, finished_at = ?5, job_id = NULL WHERE id = ?1",
            params![id, status.as_str(), error, report.to_string(), now()],
        )
        .map_err(err)?;
        for (name, value) in metrics {
            tx.execute("INSERT OR REPLACE INTO metrics (run_id, name, value) VALUES (?1, ?2, ?3)", params![id, name, value])
                .map_err(err)?;
        }
        tx.commit().map_err(err)
    }

    /// Newest first, without reports.
    pub fn list(&self, f: &Filter) -> Result<Vec<Run>> {
        let c = self.conn();
        let mut stmt = c
            .prepare(
                "SELECT id, suite, model, preset, started_at, finished_at, status, error, target, conditions, NULL, job_id
                 FROM runs WHERE (?1 IS NULL OR model = ?1) AND (?2 IS NULL OR suite = ?2)
                 ORDER BY id DESC LIMIT ?3",
            )
            .map_err(err)?;
        let rows = stmt
            .query_map(params![f.model, f.suite, f.limit.unwrap_or(500)], row_to_run)
            .map_err(err)?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(err)?;
        drop(stmt);
        rows.into_iter().map(|mut r| {
            r.metrics = metrics_of(&c, r.id)?;
            Ok(r)
        })
        .collect()
    }

    pub fn get(&self, id: i64) -> Result<Option<Run>> {
        let c = self.conn();
        let run = c
            .query_row(
                "SELECT id, suite, model, preset, started_at, finished_at, status, error, target, conditions, report, job_id
                 FROM runs WHERE id = ?1",
                params![id],
                row_to_run,
            )
            .optional()
            .map_err(err)?;
        match run {
            Some(mut r) => {
                r.metrics = metrics_of(&c, id)?;
                Ok(Some(r))
            }
            None => Ok(None),
        }
    }

    /// Add one request's traffic to `day` (YYYY-MM-DD).
    pub fn add_usage(&self, day: &str, u: &UsageCounts) -> Result<()> {
        self.conn()
            .execute(
                "INSERT INTO usage_daily (day, model, requests, prompt_tokens, completion_tokens, audio_seconds)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                 ON CONFLICT (day, model) DO UPDATE SET
                    requests = requests + excluded.requests,
                    prompt_tokens = prompt_tokens + excluded.prompt_tokens,
                    completion_tokens = completion_tokens + excluded.completion_tokens,
                    audio_seconds = audio_seconds + excluded.audio_seconds",
                params![day, u.model, u.requests as i64, u.prompt_tokens as i64, u.completion_tokens as i64, u.audio_seconds],
            )
            .map(|_| ())
            .map_err(err)
    }

    /// One row per (day, model) for days >= `since`, oldest first.
    pub fn usage_by_day(&self, since: &str) -> Result<Vec<(String, UsageCounts)>> {
        let c = self.conn();
        let mut stmt = c
            .prepare(
                "SELECT day, model, requests, prompt_tokens, completion_tokens, audio_seconds
                 FROM usage_daily WHERE day >= ?1 ORDER BY day, model",
            )
            .map_err(err)?;
        stmt.query_map(params![since], |r| {
            Ok((
                r.get(0)?,
                UsageCounts {
                    model: r.get(1)?,
                    requests: r.get::<_, i64>(2)? as u64,
                    prompt_tokens: r.get::<_, i64>(3)? as u64,
                    completion_tokens: r.get::<_, i64>(4)? as u64,
                    audio_seconds: r.get(5)?,
                },
            ))
        })
        .map_err(err)?
        .collect::<std::result::Result<_, _>>()
        .map_err(err)
    }

    /// Totals per model for days >= `since` (YYYY-MM-DD), or all days.
    pub fn usage_totals(&self, since: Option<&str>) -> Result<Vec<UsageCounts>> {
        let c = self.conn();
        let mut stmt = c
            .prepare(
                "SELECT model, SUM(requests), SUM(prompt_tokens), SUM(completion_tokens), SUM(audio_seconds)
                 FROM usage_daily WHERE (?1 IS NULL OR day >= ?1) GROUP BY model ORDER BY model",
            )
            .map_err(err)?;
        stmt.query_map(params![since], |r| {
            Ok(UsageCounts {
                model: r.get(0)?,
                requests: r.get::<_, i64>(1)? as u64,
                prompt_tokens: r.get::<_, i64>(2)? as u64,
                completion_tokens: r.get::<_, i64>(3)? as u64,
                audio_seconds: r.get(4)?,
            })
        })
        .map_err(err)?
        .collect::<std::result::Result<_, _>>()
        .map_err(err)
    }

    pub fn add_vote(&self, v: &Vote) -> Result<i64> {
        let c = self.conn();
        c.execute(
            "INSERT INTO arena_votes (created_at, prompt, model_a, model_b, winner, judge) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![now(), v.prompt, v.model_a, v.model_b, v.winner.as_str(), v.judge],
        )
        .map_err(err)?;
        Ok(c.last_insert_rowid())
    }

    /// Oldest first (Elo depends on order).
    pub fn votes(&self) -> Result<Vec<Vote>> {
        let c = self.conn();
        let mut stmt = c
            .prepare("SELECT id, created_at, prompt, model_a, model_b, winner, judge FROM arena_votes ORDER BY id")
            .map_err(err)?;
        stmt.query_map([], |r| {
            Ok(Vote {
                id: r.get(0)?,
                created_at: r.get(1)?,
                prompt: r.get(2)?,
                model_a: r.get(3)?,
                model_b: r.get(4)?,
                winner: Winner::parse(&r.get::<_, String>(5)?),
                judge: r.get(6)?,
            })
        })
        .map_err(err)?
        .collect::<std::result::Result<_, _>>()
        .map_err(err)
    }

    pub fn delete(&self, id: i64) -> Result<bool> {
        Ok(self.conn().execute("DELETE FROM runs WHERE id = ?1", params![id]).map_err(err)? > 0)
    }
}

fn row_to_run(row: &rusqlite::Row<'_>) -> rusqlite::Result<Run> {
    let json = |i: usize| -> rusqlite::Result<Value> {
        Ok(row.get::<_, Option<String>>(i)?.and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(Value::Null))
    };
    Ok(Run {
        id: row.get(0)?,
        suite: row.get(1)?,
        model: row.get(2)?,
        preset: row.get(3)?,
        started_at: row.get(4)?,
        finished_at: row.get(5)?,
        status: RunStatus::parse(&row.get::<_, String>(6)?),
        error: row.get(7)?,
        target: json(8)?,
        conditions: json(9)?,
        report: json(10)?,
        metrics: BTreeMap::new(),
        job_id: row.get::<_, Option<i64>>(11)?.map(|j| j as u64),
    })
}

fn metrics_of(c: &Connection, id: i64) -> Result<BTreeMap<String, f64>> {
    let mut stmt = c.prepare_cached("SELECT name, value FROM metrics WHERE run_id = ?1").map_err(err)?;
    stmt.query_map(params![id], |r| Ok((r.get(0)?, r.get(1)?)))
        .map_err(err)?
        .collect::<std::result::Result<_, _>>()
        .map_err(err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn usage_accumulates_per_day_and_model() {
        let store = Store::in_memory().unwrap();
        let u = |m: &str, p, c| UsageCounts { model: m.into(), requests: 1, prompt_tokens: p, completion_tokens: c, audio_seconds: 0.0 };
        store.add_usage("2026-11-05", &u("a", 10, 5)).unwrap();
        store.add_usage("2026-11-06", &u("a", 20, 7)).unwrap();
        store.add_usage("2026-11-06", &u("a", 1, 1)).unwrap();
        store.add_usage("2026-11-06", &u("b", 3, 0)).unwrap();
        let all = store.usage_totals(None).unwrap();
        assert_eq!((all[0].requests, all[0].prompt_tokens, all[0].completion_tokens), (3, 31, 13));
        let days = store.usage_by_day("2026-11-06").unwrap();
        assert_eq!(days.len(), 2, "two models on the 6th");
        assert_eq!((days[0].0.as_str(), days[0].1.prompt_tokens), ("2026-11-06", 21));
        let today = store.usage_totals(Some("2026-11-06")).unwrap();
        assert_eq!(today.iter().map(|x| (x.model.as_str(), x.requests)).collect::<Vec<_>>(), [("a", 2), ("b", 1)]);
    }

    #[test]
    fn run_lifecycle() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&dir.path().join("bench.db")).unwrap();
        let target = json!({"device": "gpu"});
        let cond = json!({"power_profile": "balanced"});
        let new = |model| NewRun { suite: "perf", model, preset: "quick", target: &target, conditions: &cond, job_id: Some(7) };
        let a = store.start(&new("a")).unwrap();
        let b = store.start(&new("b")).unwrap();

        let metrics = BTreeMap::from([("tg_tps".to_owned(), 22.5)]);
        store.finish(a, RunStatus::Done, None, &json!({"perf": {}}), &metrics).unwrap();
        let got = store.get(a).unwrap().unwrap();
        assert_eq!(got.status, RunStatus::Done);
        assert_eq!(got.metrics["tg_tps"], 22.5);
        assert_eq!(got.report, json!({"perf": {}}));
        assert_eq!(got.job_id, None);

        let all = store.list(&Filter::default()).unwrap();
        assert_eq!(all.iter().map(|r| r.id).collect::<Vec<_>>(), [b, a], "newest first");
        assert_eq!(all[1].report, Value::Null, "listings skip reports");
        assert_eq!(store.list(&Filter { model: Some("a".into()), ..Default::default() }).unwrap().len(), 1);

        // a restart fails the run that was still going
        drop(store);
        let store = Store::open(&dir.path().join("bench.db")).unwrap();
        assert_eq!(store.fail_interrupted().unwrap(), 1);
        assert_eq!(store.get(b).unwrap().unwrap().status, RunStatus::Failed);

        assert!(store.delete(a).unwrap());
        assert!(store.get(a).unwrap().is_none());
        assert!(!store.delete(a).unwrap());
    }
}
