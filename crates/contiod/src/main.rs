//! `contiod`: the contio daemon (spec §3).
//!
//! Owns all state; the GUI and CLI are thin clients of its management API.

mod api;
mod config;
mod gateway;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use clap::Parser;
use contio_backends::{Backend, FlmBackend, FlmConfig, LlamaCppBackend, LlamaCppConfig, ManagerConfig, ModelManager};
use contio_core::hw::HardwareInfo;
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

    let manager = build_manager(&config, &hw);
    tracing::info!(models = manager.profiles().len(), "model registry loaded");
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

    let state = api::AppState::start(probe, manager.clone(), Duration::from_millis(args.metrics_interval_ms));
    let listener = tokio::net::TcpListener::bind(listen)
        .await
        .with_context(|| format!("binding {listen}"))?;
    tracing::info!("listening on http://{listen} (OpenAI API at http://{listen}/v1)");

    axum::serve(listener, api::router(state))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutting down");
        })
        .await?;
    reaper.abort();
    manager.shutdown().await;
    Ok(())
}

fn build_manager(config: &Config, hw: &HardwareInfo) -> ModelManager {
    let mut backends: Vec<Arc<dyn Backend>> = Vec::new();
    if let Some(l) = &config.backends.llamacpp {
        backends.push(Arc::new(LlamaCppBackend::new(LlamaCppConfig {
            bin_dir: l.bin_dir.clone(),
            cpu_threads: l.cpu_threads,
        })));
    }
    if let Some(f) = &config.backends.flm {
        backends.push(Arc::new(FlmBackend::new(FlmConfig {
            prefix: f.prefix.clone(),
            model_path: f.model_path.clone(),
            xrt_dir: f.xrt_dir.clone(),
            embed: f.embed,
        })));
    }
    if backends.is_empty() && !config.models.is_empty() {
        tracing::warn!("models are configured but no [backends.*] section is; nothing can be loaded");
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
