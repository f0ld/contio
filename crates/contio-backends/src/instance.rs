//! A supervised backend server process.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant as StdInstant};

use nix::sys::signal::{Signal, kill};
use nix::unistd::Pid;
use tokio::process::{Child, Command};

/// How long a SIGTERM'd server gets before SIGKILL.
const STOP_GRACE: Duration = Duration::from_secs(10);
const POLL: Duration = Duration::from_millis(200);

#[derive(Debug)]
pub struct Instance {
    child: Child,
    port: u16,
    log_path: PathBuf,
}

impl Instance {
    /// Spawn `cmd`, logging stdout/stderr to `log_path`, and wait until
    /// `http://127.0.0.1:{port}{health_path}` answers 2xx.
    ///
    /// On failure the process is killed and the error includes the log tail.
    pub async fn spawn(
        mut cmd: Command,
        port: u16,
        health_path: &str,
        log_path: &Path,
        timeout: Duration,
        http: &reqwest::Client,
    ) -> Result<Self, String> {
        if let Some(dir) = log_path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("creating {}: {e}", dir.display()))?;
        }
        let log = std::fs::File::create(log_path).map_err(|e| format!("creating {}: {e}", log_path.display()))?;
        let log_err = log.try_clone().map_err(|e| e.to_string())?;
        cmd.stdin(Stdio::null())
            .stdout(log)
            .stderr(log_err)
            .kill_on_drop(true)
            // own process group: a Ctrl-C in the daemon's terminal must not hit
            // backends directly; the daemon stops them in order on shutdown
            .process_group(0);

        let child = cmd.spawn().map_err(|e| format!("spawn failed: {e}"))?;
        let mut instance = Self { child, port, log_path: log_path.to_owned() };
        let url = format!("http://127.0.0.1:{port}{health_path}");
        let deadline = StdInstant::now() + timeout;

        loop {
            if let Some(status) = instance.child.try_wait().map_err(|e| e.to_string())? {
                return Err(format!("exited during startup ({status})\n{}", instance.log_tail(15)));
            }
            if let Ok(res) = http.get(&url).timeout(Duration::from_secs(2)).send().await
                && res.status().is_success()
            {
                return Ok(instance);
            }
            if StdInstant::now() >= deadline {
                let tail = instance.log_tail(15);
                instance.stop().await;
                return Err(format!("not healthy after {}s\n{tail}", timeout.as_secs()));
            }
            tokio::time::sleep(POLL).await;
        }
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn log_path(&self) -> &Path {
        &self.log_path
    }

    /// Whether the process is still running.
    pub fn is_alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    /// SIGTERM, then SIGKILL after a grace period.
    pub async fn stop(mut self) {
        if let Some(pid) = self.child.id() {
            let _ = kill(Pid::from_raw(pid as i32), Signal::SIGTERM);
            if tokio::time::timeout(STOP_GRACE, self.child.wait()).await.is_ok() {
                return;
            }
            tracing::warn!(pid, "backend ignored SIGTERM, killing");
        }
        let _ = self.child.kill().await;
    }

    fn log_tail(&self, lines: usize) -> String {
        let text = std::fs::read_to_string(&self.log_path).unwrap_or_default();
        let all: Vec<&str> = text.lines().collect();
        all[all.len().saturating_sub(lines)..].join("\n")
    }
}

/// An unused localhost port. Racy by nature (the port is released before the
/// backend binds it); a failed bind surfaces as a startup error.
pub(crate) fn free_port() -> std::io::Result<u16> {
    Ok(std::net::TcpListener::bind("127.0.0.1:0")?.local_addr()?.port())
}
