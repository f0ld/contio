//! `~/.config/contio/config.toml`.
//!
//! Until the catalog and installer exist (M2), models are registered here by
//! hand. Paths may start with `~/`.

use std::collections::HashSet;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use contio_core::model::{ModelProfile, ModelSource};
use serde::Deserialize;

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub server: ServerConfig,
    pub manager: ManagerSection,
    pub backends: BackendsSection,
    pub models: Vec<ModelProfile>,
    pub aliases: Vec<Alias>,
}

/// Another name for a model, e.g. `coder` → `qwen3.6-35b-a3b` (spec §7.2).
/// `upstream_model` overrides the name sent to the backend, which reaches
/// models co-loaded in one process (FLM's embedding model next to an LLM).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Alias {
    pub name: String,
    pub model: String,
    #[serde(default)]
    pub upstream_model: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    pub listen: Option<SocketAddr>,
}

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ManagerSection {
    /// Unload unpinned idle models after this many seconds (0 = never).
    pub idle_ttl_secs: u64,
    pub startup_timeout_secs: u64,
    /// Share of each memory pool kept free for the desktop and OS.
    pub reserve_percent: u8,
}

impl Default for ManagerSection {
    fn default() -> Self {
        Self { idle_ttl_secs: 15 * 60, startup_timeout_secs: 180, reserve_percent: 10 }
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BackendsSection {
    pub llamacpp: Option<LlamaCppSection>,
    pub flm: Option<FlmSection>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LlamaCppSection {
    pub bin_dir: PathBuf,
    pub cpu_threads: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlmSection {
    pub prefix: PathBuf,
    pub model_path: PathBuf,
    #[serde(default = "default_xrt_dir")]
    pub xrt_dir: PathBuf,
    #[serde(default)]
    pub embed: bool,
}

fn default_xrt_dir() -> PathBuf {
    "/opt/xilinx/xrt".into()
}

impl Config {
    /// Load from `path`, or return defaults if the file doesn't exist.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                tracing::info!("no config at {}, using defaults", path.display());
                return Ok(Self::default());
            }
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        };
        Self::parse(&text).with_context(|| format!("in {}", path.display()))
    }

    pub fn parse(text: &str) -> anyhow::Result<Self> {
        let mut c: Self = toml::from_str(text)?;
        let home = std::env::var_os("HOME").map(PathBuf::from);
        let expand = |p: &mut PathBuf| *p = expand_tilde(p, home.as_deref());
        if let Some(l) = &mut c.backends.llamacpp {
            expand(&mut l.bin_dir);
        }
        if let Some(f) = &mut c.backends.flm {
            expand(&mut f.prefix);
            expand(&mut f.model_path);
            expand(&mut f.xrt_dir);
        }
        let mut seen = HashSet::new();
        for m in &mut c.models {
            if !seen.insert(m.id.clone()) {
                bail!("duplicate model id `{}`", m.id);
            }
            if let ModelSource::Gguf { path } = &mut m.source {
                expand(path);
            }
        }
        for a in &c.aliases {
            if !seen.insert(a.name.clone()) {
                bail!("alias `{}` clashes with a model id or another alias", a.name);
            }
            if !c.models.iter().any(|m| m.id == a.model) {
                bail!("alias `{}` points at unknown model `{}`", a.name, a.model);
            }
        }
        if c.manager.reserve_percent > 90 {
            bail!("manager.reserve_percent must be ≤ 90");
        }
        Ok(c)
    }
}

fn expand_tilde(p: &Path, home: Option<&Path>) -> PathBuf {
    match (p.strip_prefix("~"), home) {
        (Ok(rest), Some(home)) => home.join(rest),
        _ => p.to_owned(),
    }
}

/// `$XDG_CONFIG_HOME/contio/config.toml`, defaulting to `~/.config`.
pub fn default_config_path() -> PathBuf {
    xdg_dir("XDG_CONFIG_HOME", ".config").join("contio/config.toml")
}

/// `$XDG_STATE_HOME/contio/logs`, defaulting to `~/.local/state`.
pub fn default_log_dir() -> PathBuf {
    xdg_dir("XDG_STATE_HOME", ".local/state").join("contio/logs")
}

fn xdg_dir(var: &str, fallback: &str) -> PathBuf {
    std::env::var_os(var)
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(fallback)))
        .unwrap_or_else(|| PathBuf::from("/tmp"))
}

#[cfg(test)]
mod tests {
    use contio_core::hw::DeviceKind;

    use super::*;

    const EXAMPLE: &str = r#"
[server]
listen = "127.0.0.1:8080"

[manager]
idle_ttl_secs = 600

[backends.llamacpp]
bin_dir = "~/llama.cpp/build-vulkan/bin"
cpu_threads = 12

[backends.flm]
prefix = "~/flm"
model_path = "/data/flm-models"
embed = true

[[models]]
id = "qwen3.6-35b-a3b"
format = "gguf"
path = "~/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
device = "gpu"
context = 32768
pinned = true

[[models]]
id = "qwen3-4b-npu"
format = "flm"
tag = "qwen3-it:4b"
device = "npu"

[[aliases]]
name = "coder"
model = "qwen3.6-35b-a3b"

[[aliases]]
name = "embed-gemma"
model = "qwen3-4b-npu"
upstream_model = "embed-gemma:300m"
"#;

    #[test]
    fn parses_example() {
        let c = Config::parse(EXAMPLE).unwrap();
        assert_eq!(c.server.listen.unwrap().port(), 8080);
        assert_eq!(c.manager.idle_ttl_secs, 600);
        assert_eq!(c.manager.reserve_percent, 10);
        let home = PathBuf::from(std::env::var_os("HOME").unwrap());
        assert_eq!(c.backends.llamacpp.unwrap().bin_dir, home.join("llama.cpp/build-vulkan/bin"));
        let flm = c.backends.flm.unwrap();
        assert_eq!(flm.xrt_dir, PathBuf::from("/opt/xilinx/xrt"));
        assert!(flm.embed);
        assert_eq!(c.models.len(), 2);
        assert_eq!(
            c.models[0].source,
            ModelSource::Gguf { path: home.join("gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf") }
        );
        assert!(c.models[0].pinned);
        assert_eq!(c.models[1].device, DeviceKind::Npu);
        assert_eq!(c.aliases.len(), 2);
        assert_eq!(c.aliases[1].upstream_model.as_deref(), Some("embed-gemma:300m"));
    }

    #[test]
    fn rejects_bad_aliases() {
        let model = "[[models]]\nid='a'\nformat='flm'\ntag='x'\ndevice='npu'\n";
        let clash = format!("{model}[[aliases]]\nname='a'\nmodel='a'\n");
        assert!(Config::parse(&clash).unwrap_err().to_string().contains("clashes"));
        let dangling = format!("{model}[[aliases]]\nname='b'\nmodel='nope'\n");
        assert!(Config::parse(&dangling).unwrap_err().to_string().contains("unknown model"));
    }

    #[test]
    fn shipped_example_parses() {
        let c = Config::parse(include_str!("../../../docs/config.example.toml")).unwrap();
        assert_eq!(c.models.len(), 3);
    }

    #[test]
    fn empty_config_is_valid() {
        let c = Config::parse("").unwrap();
        assert!(c.models.is_empty() && c.backends.llamacpp.is_none());
    }

    #[test]
    fn rejects_duplicates_and_typos() {
        let dup = "[[models]]\nid='a'\nformat='flm'\ntag='x'\ndevice='npu'\n".repeat(2);
        assert!(Config::parse(&dup).unwrap_err().to_string().contains("duplicate"));
        assert!(Config::parse("[manager]\nidle_ttl = 5\n").is_err(), "unknown keys must fail");
    }
}
