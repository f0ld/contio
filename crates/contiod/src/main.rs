//! `contiod`: the contio daemon (spec §3).
//!
//! Owns all state; the GUI and CLI are thin clients of its management API.

mod aliases;
mod api;
mod catalog_api;
mod config;
mod gateway;
mod install_api;
mod registry;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use clap::Parser;
use contio_backends::{
    Backend, FlmBackend, FlmConfig, LlamaCppBackend, LlamaCppConfig, LlamaRuntime, ManagerConfig, ModelManager,
};
use contio_core::hw::HardwareInfo;
use contio_install::BackendKind;
use contio_hw::HwProbe;
use tracing_subscriber::EnvFilter;

use crate::config::Config;

const DEFAULT_LISTEN: &str = "127.0.0.1:8080";
const REAP_INTERVAL: Duration = Duration::from_secs(15);

#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    /// Config file [default: $XDG_CONFIG_HOME/contio/config.toml]
    #[arg(long, env = "CONTIO_CONFIG")]
    config: Option<PathBuf>,

    /// Address to listen on; overrides the config. Keep it on localhost unless
    /// an API key is configured (spec §7.3).
    #[arg(long, env = "CONTIO_LISTEN")]
    listen: Option<SocketAddr>,

    /// Live metrics sample interval in milliseconds.
    #[arg(long, default_value_t = 1000)]
    metrics_interval_ms: u64,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();
    let args = Args::parse();
    let config = Config::load(&args.config.unwrap_or_else(config::default_config_path))?;
    let listen = args
        .listen
        .or(config.server.listen)
        .unwrap_or_else(|| DEFAULT_LISTEN.parse().expect("valid default"));
    if !listen.ip().is_loopback() {
        tracing::warn!("listening on non-loopback address {listen}; the API has no auth yet");
    }

    let probe = HwProbe::default();
    let hw = probe.probe().context("hardware probe failed")?;
    tracing::info!(
        cpu = %hw.cpu.model,
        devices = ?hw.devices(),
        profile = ?hw.power.profile,
        "hardware detected"
    );

    let data_dir = config::default_data_dir();
    let (installer, interrupted) = install_api::Installer::persistent(data_dir.clone(), hw.cpu.cores);
    let orphans = contio_backends::stop_orphans(&config::default_log_dir());
    if !orphans.is_empty() {
        tracing::warn!(?orphans, "stopped backends a previous contiod left running");
    }
    let manager = build_manager(&config, &hw);
    if let Some(f) = &config.backends.flm {
        installer.set_flm(flm_config(f));
    }
    register_installed_backends(&config, &manager, &installer);
    let registry = registry::Registry::new(&data_dir);
    registry.restore(&manager);
    tracing::info!(
        models = manager.profiles().len(),
        added_via_api = manager.managed_profiles().len(),
        "model registry loaded"
    );
    let aliases = aliases::Aliases::new(config.aliases.clone(), Some(data_dir.join("aliases.json")), &manager);
    let catalog = catalog_api::CatalogService::new(
        contio_catalog::Catalog::builtin(),
        installer.http.clone(),
        contio_catalog::hf::HF_BASE,
        data_dir.join("cache/hf"),
    );
    let reaper = {
        let m = manager.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(REAP_INTERVAL);
            loop {
                tick.tick().await;
                m.reap().await;
            }
        })
    };

    let state = api::AppState::start(
        api::Services { probe, manager: manager.clone(), aliases, installer, catalog, registry },
        Duration::from_millis(args.metrics_interval_ms),
    );
    install_api::resume_interrupted(&state, interrupted);
    let listener = tokio::net::TcpListener::bind(listen)
        .await
        .with_context(|| format!("binding {listen}"))?;
    tracing::info!("listening on http://{listen} (OpenAI API at http://{listen}/v1)");

    axum::serve(listener, api::router(state))
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    reaper.abort();
    manager.shutdown().await;
    Ok(())
}

/// Ctrl-C, SIGTERM (systemd, `kill`) or SIGHUP (terminal closed): all stop
/// the backends in order.
async fn shutdown_signal() {
    use tokio::signal::unix::{SignalKind, signal};
    let mut term = signal(SignalKind::terminate()).expect("installing SIGTERM handler");
    let mut hup = signal(SignalKind::hangup()).expect("installing SIGHUP handler");
    let name = tokio::select! {
        _ = tokio::signal::ctrl_c() => "SIGINT",
        _ = term.recv() => "SIGTERM",
        _ = hup.recv() => "SIGHUP",
    };
    tracing::info!("{name}: shutting down");
}

fn build_manager(config: &Config, hw: &HardwareInfo) -> ModelManager {
    let mut backends: Vec<Arc<dyn Backend>> = Vec::new();
    if let Some(l) = &config.backends.llamacpp {
        let runtime = match (&l.bin_dir, &l.image) {
            (Some(bin_dir), _) => LlamaRuntime::Native { bin_dir: bin_dir.clone() },
            (None, Some(image)) => LlamaRuntime::Podman { image: image.clone() },
            (None, None) => unreachable!("validated in Config::parse"),
        };
        backends.push(Arc::new(LlamaCppBackend::new(LlamaCppConfig { runtime, cpu_threads: l.cpu_threads })));
    }
    if let Some(f) = &config.backends.flm {
        backends.push(Arc::new(FlmBackend::new(flm_config(f))));
    }

    let keep = 100 - u64::from(config.manager.reserve_percent);
    let pool_budgets: HashMap<_, _> = hw.pools.iter().map(|p| (p.kind, p.total_bytes / 100 * keep)).collect();
    let idle = config.manager.idle_ttl_secs;
    ModelManager::new(
        ManagerConfig {
            log_dir: config::default_log_dir(),
            startup_timeout: Duration::from_secs(config.manager.startup_timeout_secs),
            default_idle_ttl: (idle > 0).then(|| Duration::from_secs(idle)),
            pool_budgets,
        },
        backends,
        config.models.clone(),
    )
}

fn flm_config(f: &config::FlmSection) -> FlmConfig {
    FlmConfig { prefix: f.prefix.clone(), model_path: f.model_path.clone(), xrt_dir: f.xrt_dir.clone(), embed: f.embed }
}

/// Backends installed by contio (`contio backends install …`) are used for any kind
/// the config doesn't set explicitly. Newest install wins.
fn register_installed_backends(config: &Config, manager: &ModelManager, installer: &install_api::Installer) {
    let configured = |k: BackendKind| match k {
        BackendKind::LlamaCpp => config.backends.llamacpp.is_some(),
        BackendKind::Flm => config.backends.flm.is_some(),
    };
    let mut done = Vec::new();
    for installed in contio_install::discover(&installer.data_dir) {
        if configured(installed.backend) || done.contains(&installed.backend) {
            continue;
        }
        if let Some(b) = install_api::backend_from_installed(&installed, installer) {
            tracing::info!(
                backend = installed.backend.dir_name(),
                version = %installed.version,
                method = ?installed.method,
                "using installed backend"
            );
            manager.add_backend(b);
            done.push(installed.backend);
        }
    }
    if manager.backend_names().is_empty() && !manager.profiles().is_empty() {
        tracing::warn!("models are configured but no backend is; installing a model sets one up");
    }
}
