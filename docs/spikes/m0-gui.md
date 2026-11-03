# M0 Spike: Tauri + Svelte shell ↔ daemon

> Status: **Done.** Date: 2026-11-03.
> The spike was built as the real workspace skeleton (spec §3), so it carries into M1.

## What was built

| Piece | Path | Notes |
|---|---|---|
| Workspace | `Cargo.toml` | edition 2024, shared deps and lints, release profile at root |
| `contio-core` | `crates/contio-core` | serde types: `HardwareInfo`, `Metrics`, `MemoryPool{Gpu,System}`, `DeviceKind`, `PowerProfile` |
| `contio-hw` | `crates/contio-hw` | sysfs/procfs probe below a configurable root, no root privileges needed |
| `contiod` | `crates/contiod` | axum: `GET /api/health`, `GET /api/hw`, `WS /api/events` (1 Hz `Metrics`), `GET /v1/models` (stub) |
| GUI | `apps/gui` | Tauri 2 + SvelteKit SPA + TS: dashboard with pool bars, SoC W, GPU busy, profile, devices |

## Verified
- `cargo test`: **9 tests pass** (probe against a Framework 13 fixture tree; degraded systems; inaccessible NPU; API JSON + errors; CORS allow/deny).
- `npm run check`: 0 errors / 0 warnings. `npm run build` OK. `cargo build -p contio-gui` OK (59 s cold).
- **Real hardware** (`GET /api/hw`): HX 370 detected as 4× 5.2 GHz + 8× 3.3 GHz cores, AVX-512, 64 GiB VRAM + 31 GiB GTT, NPU `amdxdna` accessible, `balanced` on AC. The WebSocket streams `soc_watts` (~5.4 W idle) and pool usage every second.
- **End to end**: `npm run tauri dev` → the WebKitGTK webview holds 2 connections to `contiod` (REST + WebSocket).

## Decisions / findings
1. **GUI → daemon directly over HTTP/WS** from the webview (no Tauri-command proxy). The daemon CORS-allows only the GUI origins (`tauri://localhost`, `http://tauri.localhost`, Vite dev `:1420`); the Tauri CSP limits `connect-src` to the daemon. Revisit when API keys arrive (spec §7.3): a proxy via Tauri commands would keep the key out of the webview.
2. **Hardware probe sources (all user-readable)**: `/proc/cpuinfo`, `/proc/meminfo`, `cpuN/cpufreq/cpuinfo_max_freq` (Zen 5 vs 5c split), `drm/cardN/device/mem_info_*`, `gpu_busy_percent`, amdgpu hwmon `PPT`, `/sys/class/accel`, `/sys/firmware/acpi/platform_profile`, `power_supply/*`.
3. **One shared metrics sampler** (`tokio::sync::watch`): N GUI clients cost one sysfs read per second.
4. **Tooling gaps**: Fedora's `rust` package ships without `clippy`/`rustfmt` → `sudo dnf install clippy rustfmt` (lints are configured but were not run).
5. The repo `.gitignore` (Python template) ignored `apps/gui/src/lib/` via `lib/`, so an exception was added.

## Next (M1)
- Model registry + catalog schema (spec §4), `Backend` trait with llama.cpp Vulkan and FLM adapters (spec §5).
- Lifecycle manager: spawn/health-check/stop child processes, per-pool budgets (spec §7.2).
- OpenAI gateway: proxy `/v1/chat/completions` (SSE) to the routed backend.
- `contio` CLI on top of the management API.
