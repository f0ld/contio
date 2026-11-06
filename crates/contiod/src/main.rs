//! `contiod`: the contio daemon (spec §3).
//!
//! Owns all state; the GUI and CLI are thin clients of its management API.

mod aliases;
mod api;
mod bench_api;
mod catalog_api;
mod config;
mod gateway;
mod install_api;
mod registry;
mod usage;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use clap::Parser;
use contio_backends::{
    Backend, FlmBackend, FlmConfig, LlamaCppBackend, LlamaCppConfig, LlamaRuntime, ManagerConfig, ModelManager,
    WhisperBackend, WhisperConfig,
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
    let mut config = Config::load(&args.config.unwrap_or_else(config::default_config_path))?;
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
    // saved overrides of config.toml models apply before the manager sees them
    let mut registry = registry::Registry::new(&data_dir);
    registry.apply_overrides(&mut config.models);
    let manager = build_manager(&config, &hw);
    if let Some(f) = &config.backends.flm {
        installer.set_flm(flm_config(f));
    }
    register_installed_backends(&config, &manager, &installer);
    registry.restore(&manager);
    tracing::info!(
        models = manager.profiles().len(),
        added_via_api = manager.managed_profiles().len(),
        "model registry loaded"
    );
    let bench = contio_bench::store::Store::open(&data_dir.join("bench.db")).map_err(anyhow::Error::msg)?;
    match bench.fail_interrupted() {
        Ok(0) => {}
        Ok(n) => tracing::warn!("{n} benchmark run(s) were cut off by the last shutdown"),
        Err(e) => tracing::warn!("{e}"),
    }
    // benchmarks call the gateway like a client would
    let self_url = if listen.ip().is_unspecified() {
        format!("http://127.0.0.1:{}", listen.port())
    } else {
        format!("http://{listen}")
    };
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
        api::Services { probe, manager: manager.clone(), aliases, installer, catalog, registry, bench, self_url },
        Duration::from_millis(args.metrics_interval_ms),
    );
    install_api::resume_interrupted(&state, interrupted);
    let listener = tokio::net::TcpListener::bind(listen)
        .await
        .with_context(|| format!("binding {listen}"))?;
    tracing::info!("listening on http://{listen} (OpenAI API at http://{listen}/v1)");

    let jobs = state.installer.jobs.clone();
    let requested = state.shutdown.clone();
    let stop = Arc::new(tokio::sync::Notify::new());
    let stopped = stop.clone();
    let serve = axum::serve(listener, api::router(state)).with_graceful_shutdown(async move { stopped.notified().await });
    let signal = async {
        tokio::select! {
            () = shutdown_signal() => {}
            () = requested.notified() => {}
        }
        // benchmarks keep requests open against our own endpoint; downloads
        // and builds are left to resume on the next start
        for j in jobs.list() {
            if j.kind == "bench" && j.state == contio_install::jobs::JobState::Running {
                jobs.cancel(j.id);
            }
        }
        stop.notify_one();
        tokio::time::sleep(SHUTDOWN_GRACE).await;
    };
    tokio::select! {
        r = serve.into_future() => r?,
        () = signal => tracing::warn!("requests still open after {}s; closing them", SHUTDOWN_GRACE.as_secs()),
    }
    reaper.abort();
    manager.shutdown().await;
    Ok(())
}

/// How long open requests (streams) may finish after a shutdown signal.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(10);

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
    if let Some(w) = &config.backends.whisper {
        backends.push(Arc::new(WhisperBackend::new(WhisperConfig { bin_dir: w.bin_dir.clone(), cpu_threads: w.cpu_threads })));
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
    FlmConfig {
        prefix: f.prefix.clone(),
        model_path: f.model_path.clone(),
        xrt_dir: f.xrt_dir.clone(),
        embed: f.embed,
        asr: f.asr,
    }
}

/// Backends installed by contio (`contio backends install …`) are used for any kind
/// the config doesn't set explicitly. Newest install wins.
fn register_installed_backends(config: &Config, manager: &ModelManager, installer: &install_api::Installer) {
    let configured = |k: BackendKind| match k {
        BackendKind::LlamaCpp => config.backends.llamacpp.is_some(),
        BackendKind::Flm => config.backends.flm.is_some(),
        BackendKind::Whisper => config.backends.whisper.is_some(),
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
