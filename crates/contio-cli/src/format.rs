//! Human-readable output.

use std::fmt::Write;

use contio_core::hw::{HardwareInfo, MemoryPool, MemoryPoolKind, PowerProfile};
use contio_core::model::{ModelState, ModelStatus};
use serde_json::Value;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / GIB)
}

fn pool_name(k: MemoryPoolKind) -> &'static str {
    match k {
        MemoryPoolKind::Gpu => "gpu",
        MemoryPoolKind::System => "system",
    }
}

fn state(s: &ModelState) -> String {
    match s {
        ModelState::Stopped => "stopped".into(),
        ModelState::Loading => "loading".into(),
        ModelState::Ready { port } => format!("ready :{port}"),
        ModelState::Failed { error } => {
            let first = error.lines().next().unwrap_or_default();
            format!("failed: {}", first.chars().take(60).collect::<String>())
        }
    }
}

pub fn models(models: &[ModelStatus]) -> String {
    if models.is_empty() {
        return "no models configured (see docs/config.example.toml)".into();
    }
    let rows: Vec<[String; 6]> = models
        .iter()
        .map(|m| {
            [
                m.id.clone(),
                format!("{:?}", m.device).to_lowercase(),
                m.backend.clone().unwrap_or_else(|| "-".into()),
                m.estimated_bytes.map_or_else(|| "-".into(), gib),
                if m.in_flight > 0 { format!("{} ({} active)", state(&m.state), m.in_flight) } else { state(&m.state) },
                if m.pinned { "pinned".into() } else { String::new() },
            ]
        })
        .collect();
    table(&["MODEL", "DEVICE", "BACKEND", "EST. MEM", "STATE", ""], &rows)
}

pub fn status(url: &str, health: &Value, models: &[ModelStatus]) -> String {
    let loaded: Vec<&str> = models
        .iter()
        .filter(|m| matches!(m.state, ModelState::Ready { .. }))
        .map(|m| m.id.as_str())
        .collect();
    let mut s = format!(
        "contiod {} at {url}\nOpenAI API: {url}/v1\nmodels: {} configured, {} loaded",
        health["version"].as_str().unwrap_or("?"),
        models.len(),
        loaded.len()
    );
    if !loaded.is_empty() {
        let _ = write!(s, " ({})", loaded.join(", "));
    }
    s
}

pub fn hardware(hw: &HardwareInfo) -> String {
    let mut s = String::new();
    let classes = hw
        .cpu
        .core_classes
        .iter()
        .map(|c| format!("{}× {:.1} GHz", c.cores, f64::from(c.max_mhz) / 1000.0))
        .collect::<Vec<_>>()
        .join(" + ");
    let _ = writeln!(s, "CPU   {}", hw.cpu.model);
    let _ = writeln!(
        s,
        "      {} cores / {} threads ({classes}){}",
        hw.cpu.cores,
        hw.cpu.threads,
        if hw.cpu.avx512 { ", AVX-512" } else { "" }
    );
    match &hw.gpu {
        Some(g) => {
            let _ = writeln!(s, "GPU   {} {}  {} VRAM + {} GTT", g.driver, g.pci_id, gib(g.vram_total_bytes), gib(g.gtt_total_bytes));
        }
        None => s.push_str("GPU   not detected\n"),
    }
    match &hw.npu {
        Some(n) => {
            let access = if n.accessible { "" } else { "  (no access: check the `render` group)" };
            let _ = writeln!(s, "NPU   {} {}{access}", n.driver, n.device_path);
        }
        None => s.push_str("NPU   not detected\n"),
    }
    for p in &hw.pools {
        let _ = writeln!(s, "MEM   {:<6} {}", pool_name(p.kind), pool_bar(p));
    }
    let profile = match &hw.power.profile {
        PowerProfile::LowPower => "low-power".to_owned(),
        PowerProfile::Balanced => "balanced".to_owned(),
        PowerProfile::Performance => "performance".to_owned(),
        PowerProfile::Unknown(u) => format!("unknown ({u})"),
    };
    let ac = match hw.power.on_ac {
        Some(true) => ", on AC",
        Some(false) => ", on battery",
        None => "",
    };
    let bat = hw.power.battery_percent.map(|b| format!(", battery {b}%")).unwrap_or_default();
    let _ = write!(s, "POWER {profile}{ac}{bat}");
    s
}

pub fn job_line(job: &Value) -> String {
    let title = job["title"].as_str().unwrap_or_default();
    let step = job["step"].as_str().unwrap_or("");
    let progress = match (job["progress"]["done"].as_u64(), job["progress"]["total"].as_u64()) {
        (Some(d), Some(t)) if t > 0 => format!(" {:>5.1}% ({} / {})", 100.0 * d as f64 / t as f64, gib(d), gib(t)),
        (Some(d), _) => format!(" {}", gib(d)),
        _ => String::new(),
    };
    let state = job["state"].as_str().unwrap_or("?");
    format!("[{state}] {title}: {step}{progress}")
}

pub fn jobs(raw: &Value) -> String {
    let jobs = raw.as_array().cloned().unwrap_or_default();
    if jobs.is_empty() {
        return "no jobs".into();
    }
    jobs.iter()
        .map(|j| {
            let mut line = format!("{:>3}  {}", j["id"], job_line(j));
            if let Some(e) = j["error"].as_str() {
                let _ = write!(line, "\n     error: {}", e.lines().next().unwrap_or_default());
            }
            line
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn backends(raw: &Value) -> String {
    let mut s = String::new();
    let registered: Vec<&str> = raw["registered"].as_array().into_iter().flatten().filter_map(|v| v.as_str()).collect();
    if registered.is_empty() {
        s.push_str("registered: none\n");
    } else {
        s.push_str("registered (first match wins):\n");
        for r in &registered {
            let _ = writeln!(s, "  {r}");
        }
    }
    let installed = raw["installed"].as_array().cloned().unwrap_or_default();
    if installed.is_empty() {
        s.push_str("installed:  none (try `contio install llama.cpp --method prebuilt`)");
    } else {
        s.push_str("installed:");
        for b in installed {
            let at = b["bin_dir"].as_str().or(b["image"].as_str()).or(b["prefix"].as_str()).unwrap_or("?");
            let _ = write!(s, "\n  {} {} ({})  {at}", b["backend"].as_str().unwrap_or("?"), b["version"].as_str().unwrap_or("?"), b["method"].as_str().unwrap_or("?"));
        }
    }
    s
}

pub fn preflight(raw: &Value) -> String {
    if raw["ok"] == true {
        return "all prerequisites present".into();
    }
    let p = &raw["preflight"];
    let mut lines = Vec::new();
    let list = |v: &Value| v.as_array().into_iter().flatten().filter_map(|x| x.as_str().map(str::to_owned)).collect::<Vec<_>>();
    let pkgs = list(&p["missing_packages"]);
    if !pkgs.is_empty() {
        lines.push(format!("missing packages: {}", pkgs.join(", ")));
    }
    let tools = list(&p["missing_tools"]);
    if !tools.is_empty() {
        lines.push(format!("missing tools: {}", tools.join(", ")));
    }
    lines.extend(list(&p["notes"]));
    if let Some(h) = raw["install_hint"].as_str() {
        lines.push(format!("\nrun:  {h}"));
    }
    lines.join("\n")
}

fn pool_bar(p: &MemoryPool) -> String {
    const WIDTH: usize = 24;
    let frac = if p.total_bytes == 0 { 0.0 } else { p.used_bytes as f64 / p.total_bytes as f64 };
    let filled = ((frac * WIDTH as f64).round() as usize).min(WIDTH);
    format!("[{}{}] {} / {}", "█".repeat(filled), "·".repeat(WIDTH - filled), gib(p.used_bytes), gib(p.total_bytes))
}

fn table<const N: usize>(header: &[&str; N], rows: &[[String; N]]) -> String {
    let mut widths = header.map(|h| h.chars().count());
    for r in rows {
        for (w, c) in widths.iter_mut().zip(r) {
            *w = (*w).max(c.chars().count());
        }
    }
    let line = |cells: Vec<&str>| {
        cells
            .iter()
            .zip(widths)
            .map(|(c, w)| format!("{c:<w$}"))
            .collect::<Vec<_>>()
            .join("  ")
            .trim_end()
            .to_owned()
    };
    let mut out = vec![line(header.to_vec())];
    out.extend(rows.iter().map(|r| line(r.iter().map(String::as_str).collect())));
    out.join("\n")
}

#[cfg(test)]
mod tests {
    use contio_core::hw::DeviceKind;

    use super::*;

    fn status(id: &str, state: ModelState) -> ModelStatus {
        ModelStatus {
            id: id.into(),
            device: DeviceKind::Gpu,
            pool: MemoryPoolKind::Gpu,
            backend: Some("llama.cpp".into()),
            pinned: id == "a",
            estimated_bytes: Some(24 * 1024 * 1024 * 1024),
            state,
            in_flight: 0,
        }
    }

    #[test]
    fn models_table_aligns() {
        let t = models(&[
            status("a", ModelState::Ready { port: 4242 }),
            status("longer-name", ModelState::Failed { error: "boom\nmore".into() }),
        ]);
        let lines: Vec<&str> = t.lines().collect();
        assert!(lines[0].starts_with("MODEL        DEVICE"));
        assert!(lines[1].contains("ready :4242") && lines[1].ends_with("pinned"));
        assert!(lines[2].contains("failed: boom") && !lines[2].contains("more"));
        assert!(lines[1].contains("24.0 GiB"));
    }

    #[test]
    fn job_lines() {
        let j = serde_json::json!({"title": "x/y.gguf", "state": "running", "step": "downloading",
                                   "progress": {"done": 512u64 * 1024 * 1024, "total": 1024u64 * 1024 * 1024}});
        assert_eq!(job_line(&j), "[running] x/y.gguf: downloading  50.0% (0.5 GiB / 1.0 GiB)");
        let failed = serde_json::json!([{"id": 3, "title": "t", "state": "failed", "error": "boom\nmore"}]);
        assert!(jobs(&failed).contains("error: boom") && !jobs(&failed).contains("more"));
    }

    #[test]
    fn preflight_hint() {
        let raw = serde_json::json!({"ok": false, "install_hint": "sudo dnf install glslc",
                                     "preflight": {"missing_packages": ["glslc"], "missing_tools": [], "notes": []}});
        let t = preflight(&raw);
        assert!(t.contains("missing packages: glslc") && t.ends_with("run:  sudo dnf install glslc"));
    }

    #[test]
    fn empty_models_hint() {
        assert!(models(&[]).contains("config.example.toml"));
    }

    #[test]
    fn pool_bar_is_bounded() {
        let p = MemoryPool { kind: MemoryPoolKind::Gpu, total_bytes: 100, used_bytes: 250 };
        assert_eq!(pool_bar(&p).chars().filter(|c| *c == '█').count(), 24);
    }
}
