//! `contiod`: the contio daemon (spec §3).
//!
//! Owns all state; the GUI and CLI are thin clients of its management API.

mod api;

use std::net::SocketAddr;
use std::time::Duration;

use anyhow::Context;
use clap::Parser;
use contio_hw::HwProbe;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    /// Address to listen on. Keep it on localhost unless an API key is configured (spec §7.3).
    #[arg(long, env = "CONTIO_LISTEN", default_value = "127.0.0.1:8080")]
    listen: SocketAddr,

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

    if !args.listen.ip().is_loopback() {
        tracing::warn!("listening on non-loopback address {}; the API has no auth yet", args.listen);
    }

    let probe = HwProbe::default();
    let hw = probe.probe().context("hardware probe failed")?;
    tracing::info!(
        cpu = %hw.cpu.model,
        devices = ?hw.devices(),
        profile = ?hw.power.profile,
        "hardware detected"
    );

    let state = api::AppState::start(probe, Duration::from_millis(args.metrics_interval_ms));
    let listener = tokio::net::TcpListener::bind(args.listen)
        .await
        .with_context(|| format!("binding {}", args.listen))?;
    tracing::info!("listening on http://{}", args.listen);

    axum::serve(listener, api::router(state))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutting down");
        })
        .await?;
    Ok(())
}
