//! The background service: contiod as a systemd user unit, so the API keeps
//! running without the window and starts at login.
//!
//! A daemon can't outlive the app while it runs from inside the AppImage (the
//! image is unmounted when the app exits), so enabling the service copies
//! `contiod` and `contio` to `~/.local/share/contio/bin` and points the unit
//! there. The app refreshes those copies when it is newer.

use std::path::PathBuf;
use std::process::Command;

pub const UNIT: &str = "contiod.service";

fn home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

/// Where the service's binaries live.
pub fn bin_dir() -> Option<PathBuf> {
    let data = std::env::var_os("XDG_DATA_HOME").map(PathBuf::from).or_else(|| home().map(|h| h.join(".local/share")))?;
    Some(data.join("contio/bin"))
}

fn unit_path() -> Option<PathBuf> {
    let config = std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from).or_else(|| home().map(|h| h.join(".config")))?;
    Some(config.join("systemd/user").join(UNIT))
}

fn systemctl(args: &[&str]) -> Result<(), String> {
    let out = Command::new("systemctl")
        .arg("--user")
        .args(args)
        .output()
        .map_err(|e| format!("systemctl: {e}"))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(format!("systemctl --user {}: {}", args.join(" "), String::from_utf8_lossy(&out.stderr).trim()))
    }
}

pub fn available() -> bool {
    Command::new("systemctl").args(["--user", "--version"]).output().is_ok_and(|o| o.status.success())
}

pub fn installed() -> bool {
    unit_path().is_some_and(|p| p.is_file())
}

pub fn active() -> bool {
    Command::new("systemctl").args(["--user", "is-active", "--quiet", UNIT]).status().is_ok_and(|s| s.success())
}

/// `contiod --version` of a binary, e.g. `contiod 0.3.0`.
pub fn version_of(path: &std::path::Path) -> Option<String> {
    let out = Command::new(path).arg("--version").output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

pub fn installed_version() -> Option<String> {
    version_of(&bin_dir()?.join("contiod"))
}

/// Copy the bundled binaries next to each other in `bin_dir` (atomically: a
/// running service keeps its old file until it restarts).
fn copy_binaries(contiod: &std::path::Path) -> Result<(), String> {
    let dir = bin_dir().ok_or("no home directory")?;
    std::fs::create_dir_all(&dir).map_err(|e| format!("creating {}: {e}", dir.display()))?;
    let from_dir = contiod.parent().ok_or("bad path")?;
    for name in ["contiod", "contio"] {
        let src = from_dir.join(name);
        let tmp = dir.join(format!(".{name}.new"));
        std::fs::copy(&src, &tmp).map_err(|e| format!("copying {name}: {e}"))?;
        std::fs::rename(&tmp, dir.join(name)).map_err(|e| format!("installing {name}: {e}"))?;
    }
    Ok(())
}

/// Install and start the service from the bundled `contiod`.
pub fn enable(bundled_contiod: &std::path::Path) -> Result<(), String> {
    copy_binaries(bundled_contiod)?;
    let unit = unit_path().ok_or("no home directory")?;
    std::fs::create_dir_all(unit.parent().expect("has a parent")).map_err(|e| e.to_string())?;
    let exec = bin_dir().ok_or("no home directory")?.join("contiod");
    let text = format!(
        "[Unit]\n\
         Description=contio daemon (local LLMs, OpenAI-compatible API)\n\
         After=network-online.target\n\n\
         [Service]\n\
         ExecStart={}\n\
         Restart=on-failure\n\
         # SIGTERM to contiod first: it stops its model servers in order\n\
         KillMode=mixed\n\
         TimeoutStopSec=30\n\n\
         [Install]\n\
         WantedBy=default.target\n",
        exec.display()
    );
    std::fs::write(&unit, text).map_err(|e| format!("writing {}: {e}", unit.display()))?;
    systemctl(&["daemon-reload"])?;
    systemctl(&["enable", "--now", UNIT])
}

/// Stop and remove the service (the copied binaries are removed too).
pub fn disable() -> Result<(), String> {
    let _ = systemctl(&["disable", "--now", UNIT]);
    if let Some(unit) = unit_path() {
        let _ = std::fs::remove_file(unit);
    }
    systemctl(&["daemon-reload"])?;
    if let Some(dir) = bin_dir() {
        for name in ["contiod", "contio"] {
            let _ = std::fs::remove_file(dir.join(name));
        }
    }
    Ok(())
}

pub fn start() -> Result<(), String> {
    systemctl(&["start", UNIT])
}

pub fn stop() -> Result<(), String> {
    systemctl(&["stop", UNIT])
}

/// After an AppImage update: give the service the new binaries.
pub fn refresh(bundled_contiod: &std::path::Path) -> Result<bool, String> {
    let (Some(new), Some(old)) = (version_of(bundled_contiod), installed_version()) else { return Ok(false) };
    if new == old {
        return Ok(false);
    }
    copy_binaries(bundled_contiod)?;
    systemctl(&["restart", UNIT])?;
    Ok(true)
}
