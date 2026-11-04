//! Background sampling during a benchmark: SoC power at 10 Hz (amdgpu `PPT`
//! sensor, no root needed) and memory pool usage at 2 Hz.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use contio_core::hw::MemoryPoolKind;
use contio_hw::{HwProbe, read_microwatts};
use tokio::task::JoinHandle;

const POWER_INTERVAL: Duration = Duration::from_millis(100);
const MEMORY_INTERVAL: Duration = Duration::from_millis(500);

#[derive(Default)]
struct Data {
    /// (seconds since start, watts)
    power: Vec<(f64, f64)>,
    peak_used: HashMap<MemoryPoolKind, u64>,
    /// Process whose resident memory is tracked, and its peak.
    pid: Option<u32>,
    peak_rss: u64,
}

/// Samples until dropped.
pub struct Monitor {
    started: Instant,
    data: Arc<Mutex<Data>>,
    tasks: Vec<JoinHandle<()>>,
    has_power: bool,
}

impl Monitor {
    /// `probe: None` samples nothing (tests, or no AMD GPU).
    pub fn start(probe: Option<HwProbe>) -> Self {
        let started = Instant::now();
        let data = Arc::new(Mutex::new(Data::default()));
        let mut tasks = Vec::new();
        let sensor: Option<PathBuf> = probe.as_ref().and_then(HwProbe::soc_power_sensor);
        let has_power = sensor.is_some();
        if let Some(path) = sensor {
            let data = data.clone();
            tasks.push(tokio::spawn(async move {
                let mut tick = tokio::time::interval(POWER_INTERVAL);
                loop {
                    tick.tick().await;
                    // a sysfs read takes microseconds; no need for spawn_blocking
                    if let Some(w) = read_microwatts(&path) {
                        let t = started.elapsed().as_secs_f64();
                        data.lock().expect("monitor lock").power.push((t, w));
                    }
                }
            }));
        }
        if let Some(probe) = probe {
            let data = data.clone();
            tasks.push(tokio::spawn(async move {
                let mut tick = tokio::time::interval(MEMORY_INTERVAL);
                loop {
                    tick.tick().await;
                    let p = probe.clone();
                    let pid = data.lock().expect("monitor lock").pid;
                    let Ok(Ok((m, rss))) =
                        tokio::task::spawn_blocking(move || p.sample().map(|m| (m, pid.and_then(rss_bytes)))).await
                    else {
                        continue;
                    };
                    let mut d = data.lock().expect("monitor lock");
                    d.peak_rss = d.peak_rss.max(rss.unwrap_or(0));
                    for pool in m.pools {
                        let peak = d.peak_used.entry(pool.kind).or_default();
                        *peak = (*peak).max(pool.used_bytes);
                    }
                }
            }));
        }
        Self { started, data, tasks, has_power }
    }

    /// Seconds since the monitor started; the time base of all queries.
    pub fn now(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    pub fn has_power(&self) -> bool {
        self.has_power
    }

    /// Mean power between two instants, if any samples fall in or around them.
    pub fn avg_watts(&self, from: f64, to: f64) -> Option<f64> {
        let d = self.data.lock().expect("monitor lock");
        avg_watts(&d.power, from, to)
    }

    /// Joules used between two instants (mean power × duration).
    pub fn energy(&self, from: f64, to: f64) -> Option<f64> {
        self.avg_watts(from, to).map(|w| w * (to - from).max(0.0))
    }

    /// Highest used bytes per pool since the last reset.
    pub fn peak_used(&self) -> HashMap<MemoryPoolKind, u64> {
        self.data.lock().expect("monitor lock").peak_used.clone()
    }

    pub fn reset_peaks(&self) {
        let mut d = self.data.lock().expect("monitor lock");
        d.peak_used.clear();
        d.peak_rss = 0;
    }

    /// Also track this process's resident memory. It includes memory-mapped
    /// weights, which pool usage counts as cache (llama.cpp on CPU).
    pub fn watch_process(&self, pid: u32) {
        let mut d = self.data.lock().expect("monitor lock");
        d.pid = Some(pid);
        d.peak_rss = d.peak_rss.max(rss_bytes(pid).unwrap_or(0));
    }

    pub fn peak_rss(&self) -> Option<u64> {
        Some(self.data.lock().expect("monitor lock").peak_rss).filter(|&b| b > 0)
    }
}

impl Drop for Monitor {
    fn drop(&mut self) {
        for t in &self.tasks {
            t.abort();
        }
    }
}

/// `VmRSS` of a process.
fn rss_bytes(pid: u32) -> Option<u64> {
    let status = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    let kb = status.lines().find_map(|l| l.strip_prefix("VmRSS:"))?.trim().trim_end_matches("kB").trim();
    kb.parse::<u64>().ok().map(|k| k * 1024)
}

/// Mean of the samples in `[from, to]`; for windows shorter than the sample
/// interval, the samples closest to the window.
fn avg_watts(samples: &[(f64, f64)], from: f64, to: f64) -> Option<f64> {
    let inside: Vec<f64> = samples.iter().filter(|(t, _)| *t >= from && *t <= to).map(|(_, w)| *w).collect();
    if !inside.is_empty() {
        return Some(inside.iter().sum::<f64>() / inside.len() as f64);
    }
    let mid = (from + to) / 2.0;
    samples
        .iter()
        .min_by(|a, b| (a.0 - mid).abs().total_cmp(&(b.0 - mid).abs()))
        .filter(|(t, _)| (t - mid).abs() < 1.0)
        .map(|(_, w)| *w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn averages_windows() {
        let s: Vec<(f64, f64)> = (0..20).map(|i| (f64::from(i) * 0.1, if i < 10 { 10.0 } else { 20.0 })).collect();
        assert_eq!(avg_watts(&s, 0.0, 0.95), Some(10.0));
        assert_eq!(avg_watts(&s, 1.0, 1.9), Some(20.0));
        // narrower than the interval: nearest sample
        assert_eq!(avg_watts(&s, 1.52, 1.54), Some(20.0));
        assert_eq!(avg_watts(&s, 10.0, 11.0), None, "no samples nearby");
        assert_eq!(avg_watts(&[], 0.0, 1.0), None);
    }

    #[test]
    fn reads_own_rss() {
        assert!(rss_bytes(std::process::id()).is_some_and(|b| b > 1 << 20));
        assert_eq!(rss_bytes(u32::MAX), None);
    }
}
