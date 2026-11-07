//! Power while serving (docs/PERFORMANCE.md §2e): hold the `performance`
//! platform profile while a model is working, give the user's profile back
//! shortly after. The hold goes through power-profiles-daemon, which drops it
//! by itself if contiod dies.

use std::time::{Duration, Instant};

use contio_bench::power_profile::{self, ProfileGuard};
use contio_core::hw::Metrics;
use contio_install::jobs::{JobState, Jobs};
use serde::{Deserialize, Serialize};
use tokio::sync::watch;

use crate::registry::Registry;
use contio_backends::ModelManager;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServePower {
    /// Leave the power profile alone.
    #[default]
    Off,
    /// Hold `performance` while serving, on battery too.
    Always,
    /// Only when plugged in.
    OnAc,
}

const POLL: Duration = Duration::from_millis(500);
/// Keep the hold this long after the last request, so agent turns that
/// follow each other don't flip the profile back and forth.
const LINGER: Duration = Duration::from_secs(10);

/// Whether to hold right now.
fn wanted(setting: ServePower, busy: bool, on_ac: Option<bool>, benchmarking: bool) -> bool {
    busy && !benchmarking
        && match setting {
            ServePower::Off => false,
            ServePower::Always => true,
            ServePower::OnAc => on_ac == Some(true),
        }
}

pub fn spawn(manager: ModelManager, registry: Registry, jobs: Jobs, metrics: watch::Receiver<Option<Metrics>>) {
    tokio::spawn(async move {
        let mut guard: Option<ProfileGuard> = None;
        let mut last_busy = Instant::now() - LINGER;
        let mut warned = false;
        let mut tick = tokio::time::interval(POLL);
        loop {
            tick.tick().await;
            let busy = manager.statuses().iter().any(|s| s.in_flight > 0);
            if busy {
                last_busy = Instant::now();
            }
            let recent = last_busy.elapsed() < LINGER;
            let on_ac = metrics.borrow().as_ref().and_then(|m| m.power.on_ac);
            // benchmarks pick their own profile
            let benchmarking = jobs.list().iter().any(|j| j.kind == "bench" && j.state == JobState::Running);
            let hold = wanted(registry.settings().serve_power, recent, on_ac, benchmarking);
            match (hold, guard.is_some()) {
                (true, false) => match power_profile::apply("performance").await {
                    Ok(g) => {
                        tracing::info!(previous = %g.previous, "serving: holding the performance power profile");
                        guard = Some(g);
                        warned = false;
                    }
                    Err(e) if !warned => {
                        tracing::warn!("can't switch the power profile while serving: {e}");
                        warned = true;
                    }
                    Err(_) => {}
                },
                (false, true) => {
                    if let Some(g) = guard.take()
                        && let Err(e) = g.restore().await
                    {
                        tracing::warn!("{e}");
                    }
                    tracing::info!("serving done: power profile released");
                }
                _ => {}
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holds_only_when_wanted() {
        use ServePower::*;
        assert!(wanted(Always, true, Some(false), false));
        assert!(!wanted(Always, false, Some(true), false), "idle");
        assert!(!wanted(Always, true, Some(true), true), "a benchmark sets its own profile");
        assert!(wanted(OnAc, true, Some(true), false));
        assert!(!wanted(OnAc, true, Some(false), false) && !wanted(OnAc, true, None, false), "battery or unknown");
        assert!(!wanted(Off, true, Some(true), false));
    }
}
