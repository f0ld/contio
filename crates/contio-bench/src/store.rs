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
PRAGMA user_version = 1;
";

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
