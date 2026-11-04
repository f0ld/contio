# contio — Local LLM Manager · Initial Spec (v0.1, draft)

> Status: **Draft for discussion.** Last updated 2026-11-03. All initial open questions are decided (§12). Items marked **[ASSUMPTION]** are defaults chosen so we can move forward and may be revised.

---

## 1. Vision

contio is a Rust desktop app and background service for running local LLMs on a Framework 13 (AMD Ryzen AI 300). It should make it easy to:

1. **Find** a model from a large, well-organized catalog (family → model → format → quantization).
2. **Choose** where it runs: **CPU**, **iGPU**, or **NPU** (and later, hybrid).
3. **Set it up automatically**: download the backend engine and model weights, verify them, and configure everything in the background.
4. **Serve** it through one **OpenAI-compatible API** endpoint that agentic tools can use (Claude Code-style CLIs, Aider, Continue, opencode, LangChain, …).
5. **Benchmark** models, quants, and devices against each other, covering both speed and quality.

### Non-goals (for v1)
- Training or fine-tuning (LoRA adapters can be *loaded*, but not trained).
- Multi-machine or cluster serving.
- Writing our own inference kernels. contio **orchestrates** existing engines; it does not replace them.
- Windows or macOS support. Linux (Fedora) comes first. Keep the code portable where it costs nothing.

---

## 2. Target Hardware (detected on dev machine)

| Component | Detail | Relevance |
|---|---|---|
| CPU | AMD Ryzen AI 9 HX 370: 12C/24T (4× Zen 5 + 8× Zen 5c), AVX-512 | llama.cpp CPU backend; thread pinning to the Zen 5 cores matters |
| iGPU | Radeon 890M: RDNA 3.5, 16 CU, `gfx1150` | Vulkan (RADV) or ROCm/HIP |
| NPU | XDNA 2 (~50 TOPS), `amdxdna` driver loaded, `/dev/accel/accel0` present | Needs XRT plus an NPU-capable runtime (see §5.3) |
| Memory | **128 GB** LPDDR5x, unified. Currently split **64 GB VRAM carve-out (GPU)** and **64 GB system RAM (CPU/NPU)** | Two memory pools to track separately (§4.3, §7.2). Bandwidth (~120 GB/s) is shared and caps decode speed |
| OS | Fedora 44, kernel 7.2, GNOME, Rust 1.99, podman available | |

**Key performance fact:** token generation is memory-bandwidth-bound. A rough ceiling is `tok/s ≈ bandwidth / bytes-read-per-token`. Example: an 8B Q4 model (~5 GB) tops out around 20–25 tok/s on any device. **MoE models** are the sweet spot for this machine. With 64 GB per pool, large MoE models fit: e.g. a 30B-A3B model at Q8 easily, and 100B+ MoE models at ~4-bit only just (a 120B MoE at MXFP4 is ~63 GB, right at the GPU pool's limit; a split between GPU and CPU pools is an option). The GUI should show this estimate before download.

---

## 3. Architecture

```
┌────────────────────┐   ┌──────────────┐   ┌──────────────────────┐
│ contio-gui         │   │ contio (CLI) │   │ Agentic tools        │
│ (desktop app)      │   │              │   │ (OpenAI SDK clients) │
└────────┬───────────┘   └──────┬───────┘   └──────────┬───────────┘
         │ Management API (REST + WebSocket events)    │ OpenAI API
         └──────────────┬───────────────┘              │
                        ▼                              ▼
              ┌───────────────────────────────────────────────┐
              │ contiod  (daemon, systemd --user service)     │
              │  ├─ Gateway / Router (axum)                   │
              │  ├─ Model Lifecycle Manager (load/unload/TTL)  │
              │  ├─ Job Queue (downloads, installs, benches)  │
              │  ├─ Catalog + Registry (SQLite)               │
              │  ├─ Hardware Probe + Fit Estimator            │
              │  └─ Benchmark Runner                          │
              └───────┬──────────────┬──────────────┬─────────┘
                      ▼              ▼              ▼
               llama-server     llama-server    NPU runtime
               (CPU build)      (Vulkan/ROCm)   (FLM / Lemonade / OGA)
               child processes on 127.0.0.1:<ephemeral port>
```

### Design principles
- **The daemon owns all state.** The GUI and CLI are thin clients, so closing the GUI never stops the API.
- **Backends run as child processes** with their own OpenAI-ish HTTP server. contio proxies to them, which keeps crashes isolated and backends swappable.
- **One public port** (default `127.0.0.1:8080` **[ASSUMPTION]**). Requests are routed by the `model` field.
- **Everything long-running is a job**: it is cancellable, resumable, reports progress, and streams events to the GUI.

### Cargo workspace layout
```
crates/
  contio-core/      # domain types: catalog, artifacts, devices, config, errors
  contio-hw/        # hardware detection (CPU flags, GPU, NPU, memory, power sensors)
  contio-catalog/   # registry, HF Hub client, fit estimation
  contio-backends/  # Backend trait + llama.cpp / NPU adapters, installers
  contio-jobs/      # job queue, downloads (resume, sha256), progress events
  contio-api/       # OpenAI-compatible gateway + management API (axum)
  contio-bench/     # perf + quality benchmark harness
  contiod/          # daemon binary
  contio-cli/       # `contio` CLI binary
  contio-gui/       # desktop GUI binary
```

### Core tech choices **[ASSUMPTION]**
| Concern | Choice | Notes |
|---|---|---|
| Async runtime | tokio | |
| HTTP | axum + reqwest (rustls) | SSE streaming passthrough |
| Storage | SQLite via sqlx (or rusqlite) | catalog cache, installed artifacts, bench results |
| Config | TOML (`~/.config/contio/config.toml`) | |
| Data dir | `~/.local/share/contio/{backends,models,bench}` | Models stored by content hash and deduplicated |
| Logging | tracing + tracing-subscriber, with per-backend log files | |
| GUI | **Tauri 2** (Rust core + TypeScript frontend) | Decided, see §9.2 |

---

## 4. Model Catalog

### 4.1 Hierarchy
```
Family            e.g. Qwen3, Llama 3.x, Gemma 3, Mistral/Devstral, Phi-4, gpt-oss, DeepSeek, GLM
 └─ Model         e.g. Qwen3-30B-A3B-Instruct  (params, active params, ctx length, license, modality,
 │                     capabilities: tools, reasoning, vision, embeddings, FIM/code)
     └─ Format    GGUF | ONNX (OGA) | FLM | safetensors
         └─ Quant Q2_K … Q8_0, IQ2_XXS … IQ4_XS, Q4_K_M, UD-*, MXFP4, F16/BF16, int4-AWQ (NPU)
             └─ Artifact  concrete file(s) from a source repo, size, sha256, uploader
```

Each **Artifact** declares a **compatibility set**: which (backend, device) pairs can run it. The GUI filters the catalog by the device you select. For example, picking *NPU* only shows NPU-ready artifacts.

### 4.2 Sources
1. **Curated registry** (`registry/*.toml`, in the repo and updatable): hand-picked families and models with good defaults, recommended quants, chat-template fixes, and tool-calling notes. This is the "well-structured" core.
2. **Live Hugging Face Hub search**: any GGUF/ONNX repo can be added. Metadata is parsed from GGUF headers (arch, params, ctx, quant type) and repo cards. These appear as "community/unverified".
3. **Local import**: point at an existing `.gguf` file or directory.

### 4.3 Fit estimator
For each artifact, given the device and context length:
- `weights_bytes + kv_cache(ctx, layers, kv_heads, head_dim, kv_quant) + runtime_overhead`
- Compared against the **pool** the device uses: the GPU pool (VRAM carve-out, plus GTT if enabled) or the system pool (CPU/NPU). Pool sizes are detected at runtime (`mem_info_vram_total`, `/proc/meminfo`), never hardcoded, since the BIOS split can change.
- For models too large for one pool, suggest a **GPU+CPU split** (llama.cpp `--n-gpu-layers` / MoE expert offload) and estimate its speed.
- Output: ✅ fits / ⚠️ tight / ❌ won't fit, plus an **estimated tok/s** from the bandwidth model, refined later with real benchmark data.

### 4.4 Browsing UX
- Filters: capability (tools, vision, code, reasoning), size, license, device, "fits on my machine".
- Sort by: popularity, recency, estimated speed, benchmark score (once measured).
- Quant picker shows size, expected quality loss (heuristic tiers), and fit badge side by side.

---

## 5. Backends & Devices

### 5.1 Backend abstraction
```rust
trait Backend {
    fn id(&self) -> BackendId;                       // e.g. "llamacpp-vulkan"
    fn devices(&self) -> &[DeviceKind];              // Cpu | Gpu | Npu | Hybrid
    fn supports(&self, a: &Artifact) -> bool;
    async fn install(&self, ctx: &JobCtx) -> Result<InstalledBackend>;
    async fn spawn(&self, a: &Artifact, opts: &LaunchOpts) -> Result<RunningInstance>; // returns local URL
    fn capabilities(&self) -> BackendCaps;           // tools, embeddings, vision, json-schema, ...
}
```

### 5.2 CPU and GPU via llama.cpp (primary engine, v1)
- **CPU**: llama.cpp built with AVX-512 / Zen 5 flags. Options: thread count and core affinity (prefer the Zen 5 cores), mlock, mmap.
- **GPU**: llama.cpp **Vulkan (RADV)** as the default (verified in M0: builds in ~3 min, fastest device on every model tested). **ROCm/HIP** is an optional second GPU backend that is benchmarked against Vulkan.
- **Default device policy (from M0, [`docs/spikes/m0-gpu.md`](spikes/m0-gpu.md))**: GPU for chat/agentic models (MoE models like Qwen3.6 35B-A3B are the sweet spot). NPU for embeddings, Whisper, and small helper models, so the GPU stays free. CPU for overflow and small models. Speed estimates use measured numbers keyed by power profile. The **power profile is a speed/energy dial**: in M0, balanced (~25 W cap) gave +40% prefill vs power saver (~15 W cap) at +57% J/token on the GPU. contio exposes this as a user choice ("quiet / efficient" vs "fast") instead of hiding it.
- **Build preflight (Vulkan)**: `cmake`, `ninja-build`, `gcc-c++`, `vulkan-headers`, `vulkan-loader-devel`, `glslc`, `spirv-headers-devel`. Missing packages are shown as one `dnf install` line before the build starts.
- **Unified memory**: the iGPU uses the BIOS VRAM carve-out (currently 64 GB) plus GTT, which is borrowed from system RAM. contio detects the current limits and, if needed, **shows** the user how to raise them (e.g. `amdgpu`/`ttm` kernel params). It does **not** silently change system config.
- **Install strategy (decided: user-selectable)**: the first-run setup wizard asks the user to choose, and Settings → Backends lets them change it later, per backend:
  - **Build from source**: compiled in the background with Zen 5 / Vulkan / ROCm flags. Best performance, one-time ~5 min build. *Suggested default in the wizard.*
  - **Prebuilt binaries**: pinned upstream releases. Fastest setup.
  - **Podman container**: isolated and reproducible, with device passthrough (`/dev/dri`, `/dev/accel`).
  - Switching methods installs the new variant side by side, then switches profiles over to it. The old one is kept until the user removes it, so the two can be benchmarked against each other.
  - All three install methods implement the same `BackendInstaller` trait, so each backend declares which methods it supports.
- Launched with `--jinja` and the correct chat template, which tool calling needs.

### 5.2b Speech via whisper.cpp
- **whisper.cpp** (CPU / Vulkan) as a separate backend behind the same `Backend` trait, with the same three install methods. GGML Whisper models (tiny → large-v3-turbo, quantized variants) are part of the catalog.

### 5.3 NPU (XDNA 2) via FastFlowLM
**Decided after the M0 spike** (details: [`docs/spikes/m0-npu.md`](spikes/m0-npu.md)):
- The system stack is verified working: in-tree `amdxdna`, firmware 1.1.2.64, XRT 2.27, `xrt-smi validate` passes (47 TOPS).
- **Runtime: FastFlowLM (FLM)**, spawned as `flm serve <tag> --port N --ctx-len N --pmode …`. It has an OpenAI-compatible API with streaming, tools, JSON schema, embeddings, and audio transcriptions. Lemonade's Linux NPU path is FLM underneath, so it is not used. ONNX Runtime GenAI is not used in v1.
- **Install**: source build via FLM's `home_install.sh` into contio's prefix (no sudo). Models go in `FLM_MODEL_PATH`. Pinned FLM version per contio release.
- **Catalog**: imported from FLM's `model_list.json` (~70 tags, incl. Qwen3/3.5/3.6-MoE-35B, gpt-oss-20b, Gemma 4, embeddings, Whisper). `*-flash` (1k ctx, single-turn) tags are hidden from agentic presets.
- **NPU scheduling**: one `flm serve` process = one LLM + optional co-loaded embedding model (`--embed 1`). Several FLM processes can coexist; the NPU lock is taken per request, so parallel requests share throughput.
- **NPU role (from M0 measurements)**: good decode speed for ≤4B at ~13–15 W, but slow prefill (130–450 tok/s for 4B+, 10–35 s TTFT at 4.5k tokens). Default use: embeddings, Whisper, small helper models, battery-mode chat. **Not the default for agentic coding models** (GPU instead, pending the GPU spike).
- **License**: FLM runtime is MIT; NPU kernels are closed but free for any use and redistributable unmodified. Credit "Powered by FastFlowLM".
- **Hybrid** (NPU prefill + GPU decode) stays a **post-v1 goal**, and the `DeviceKind::Hybrid` variant exists from the start so it can be added without breaking anything.

### 5.4 Later backends (post-v1, behind the same trait)
- **mistral.rs** (Rust-native; useful for an in-process option).
- **Hybrid** NPU+GPU modes (via Lemonade/OGA).
- **Remote passthrough**: proxy a cloud OpenAI-compatible endpoint, useful for benchmarking local against cloud.

---

## 6. Auto-Setup (background jobs)

When the user selects **Model X / Quant Y / Device Z** and clicks **Install & Run**:

1. **Preflight**: hardware probe, check driver/permission requirements (render group, `/dev/accel` access, memlock ulimit for the NPU), fit check, disk space check.
2. **Backend install** if missing: download or build, verify checksum, smoke test (`--version` plus a tiny model load).
3. **Model download**: parallel chunked HTTP from HF with resume, **sha256 verification against the HF `lfs.oid`** (mismatch → delete and retry), optional HF token for gated models. **One writer per artifact**: a download job takes a lock on the target file, and a second request for the same artifact joins the running job (M0: two concurrent writers corrupted a `.part` file).
4. **Config generation**: a launch profile (ctx size, batch, threads, KV cache quant, flash-attn, GPU layers, chat template).
5. **Health check**: spawn the backend, run a test completion, register the model under its API name.
6. **Done**: notification. The model is now served at `/v1`.

Rules:
- Jobs persist in SQLite and **survive daemon restarts** (downloads resume).
- Steps that need root are **never** run automatically in v1. contio explains why each one is needed and shows the exact command with a "copy" button, then re-checks once the user has run it.
- **Post-v1 (decided):** an optional small privileged helper (`contio-helper`) gated by **polkit**, with a fixed allowlist of actions (GTT/TTM limits, memlock limits, `render`/`video` group membership). It never runs arbitrary commands.
- Backends are version-pinned per profile, and one-click "update backend" re-runs benchmarks for comparison.

---

## 7. OpenAI-Compatible API

### 7.1 Endpoints (v1)
| Endpoint | Notes |
|---|---|
| `GET /v1/models` | Installed models, with contio metadata in an extension field |
| `POST /v1/chat/completions` | Streaming (SSE), **tools / tool_choice**, `response_format` (json_schema), `stream_options.include_usage`, reasoning content passthrough |
| `POST /v1/completions` | Legacy plus FIM for code completion |
| `POST /v1/embeddings` | Embedding models (llama.cpp `--embedding`; NPU candidates) |
| `POST /v1/rerank` | Reranker models (llama.cpp `--reranking`; Jina/Cohere-style request shape) |
| `POST /v1/audio/transcriptions` | Speech-to-text via **whisper.cpp** (CPU/Vulkan); `/v1/audio/translations` too |
| Vision | Image inputs (`image_url` content parts) in chat completions via llama.cpp multimodal projector (`mmproj`) |
| `POST /v1/responses` | OpenAI Responses API translation layer (stateless; `previous_response_id` stored in SQLite with a TTL) |
| `POST /v1/messages` | Anthropic Messages API translation layer (tool_use / tool_result blocks, streaming events, `/v1/messages/count_tokens`) |

Every API surface can be toggled in **Settings → API**. **Default: all three enabled** on the same port. They are only reachable from localhost, so enabling them costs nothing, and most agentic tools work without the user touching any settings. Chat Completions cannot be disabled because the others are translated onto it internally.

### 7.2 Routing & lifecycle
- `model` resolves to an installed profile, either by alias (e.g. `coder`, `fast`, `default`) or full id (`qwen3-30b-a3b:q4_k_m@gpu`).
- **On-demand loading**: a request for a model that isn't loaded starts it automatically. The request waits (configurable timeout) or gets 503 with `Retry-After`.
- **Memory-aware eviction**: the manager keeps a **separate budget per memory pool** (GPU pool, system pool) and unloads LRU or idle models in the pool that needs space (idle TTL per model). Split models count against both pools.
- **Pinning**: mark models as "always loaded".
- **Multiple models at once (decided)**: e.g. a coder model on the GPU, an embedding model on the NPU, and a small model on the CPU. Loads that would exceed the budget trigger eviction first, or are refused with a clear error if pinned models block them.
- **Tool-call loop guard**: N identical consecutive tool calls in one response (M0: Qwen3.6 repeated the same call 7×) are collapsed to one and logged. Configurable.
- **Structured-output validation**: `response_format` results are validated against the schema, and violations are reported (engines don't always enforce it).
- Concurrency: per-model request queue, plus a configurable number of parallel slots (`--parallel` in llama.cpp).

### 7.3 Security
- **Decided:** binds to `127.0.0.1` by default. LAN exposure is an opt-in setting that **requires** an API key before it can be enabled, and shows a persistent "exposed on LAN" indicator in the GUI. Optional TLS (self-signed or user-provided cert) for LAN mode.
- Optional API keys, even on localhost, for per-tool usage stats.

### 7.4 Management API (for GUI/CLI, not OpenAI)
`/api/catalog`, `/api/installed`, `/api/jobs`, `/api/instances` (load/unload), `/api/bench`, `/api/hw`, `/api/events` (WebSocket: job progress, instance state, live metrics).

---

## 8. Benchmarking

### 8.1 Performance benchmarks
Per (model, quant, device, backend, settings):
- **Prompt processing** tok/s at several prompt lengths (e.g. 512 / 4k / 16k)
- **Generation** tok/s
- **TTFT** (time to first token)
- **Peak memory** (RSS + GPU GTT/VRAM)
- **Power & energy**: SoC package power via the amdgpu hwmon `PPT` sensor (readable without root; RAPL needs root), battery `current_now × voltage_now` on battery → **joules per token** (important on battery)
- **Concurrency**: throughput with N parallel requests
- Run conditions are recorded (AC vs battery, power profile, temperature, backend version) so results stay comparable. The power profile **matters a lot**: in M0, *power saver* capped the whole SoC at ~15 W, so device comparisons were power-bound. The runner can switch the profile for the run (and restore it), and results from different profiles are never shown as directly comparable.

### 8.2 Quality benchmarks
- **Decided: all four areas**, each as a small, fast subset with a fixed seed so results are reproducible:
  - **Coding**: HumanEval / MBPP style. Generated code runs **only in podman sandboxes** (no network, CPU/memory/time limits).
  - **Tool calling**: BFCL-style single, parallel, and multi-turn function calls, checked via the real API path, so chat-template bugs show up. This is the most important area for agentic use. It gates an **"agent-ready" badge per (model, backend)**: engine labels aren't trusted (M0: an FLM model labelled tool-calling hallucinated a tool result instead).
  - **Reasoning / math**: GSM8K / MATH subsets.
  - **General knowledge**: MMLU-Pro subset.
- Other model types: embedding retrieval (small MTEB-style subset), reranker ranking quality, and Whisper word error rate on a small audio set.
- **Custom prompt sets**: user-defined YAML with expected answers or regex/JSON-schema checks.
- **Side-by-side arena**: the same prompt sent to 2–4 models, with blind voting stored as Elo.
- Optional LLM-as-judge using another local (or remote) model.
- **Repetition**: `temperature: 0` is not deterministic on GPU backends (M0), so each quality item runs N times (default 3) and is reported as a pass rate.
- **Budgets**: tool-call and structured-output tests allow ≥ 2048 tokens so thinking models can finish reasoning. A separate "thinking off" variant is tested where the model supports it.

### 8.3 Results
- Stored in SQLite and exportable as CSV/JSON.
- GUI views: leaderboard table, speed-vs-quality scatter, quant-ladder chart (same model, Q2→Q8), device comparison (CPU vs GPU vs NPU).
- Bench presets: "Quick" (~1 min), "Standard", "Full".

---

## 9. GUI

### 9.1 Screens
1. **Dashboard**: loaded models, one memory bar per pool (GPU / system, showing what each model uses), live tok/s, power draw, API endpoint and copy-able config snippets for common tools.
2. **Catalog**: browse, filter, and search (§4.4), then a model detail page with the quant picker, device picker, fit badge, and Install & Run button.
3. **Installed / Profiles**: manage launch profiles (ctx, threads, layers, KV quant, aliases, pinning, TTL).
4. **Jobs**: download/build progress, logs, retry/cancel.
5. **Playground**: simple chat to test a model, with a tool-calling inspector.
6. **Benchmarks**: run, compare, charts, arena.
7. **Settings**: paths, port, API keys, HF token, backend versions, hardware info.

Also a **tray icon** **[ASSUMPTION: M5]** for quick load/unload and status.

### 9.2 GUI framework (decided: **Tauri 2**)
The Tauri app is a client of `contiod` over the management API. It does **not** embed the daemon, so the API keeps running without the GUI. Frontend stack (decided): **TypeScript + Svelte 5 (SvelteKit in SPA mode) + Vite**. Chart library **[ASSUMPTION]**: something lightweight such as uPlot or LayerChart.

Options considered:
| Option | Pros | Cons |
|---|---|---|
| **Tauri 2** (Rust + web UI) | Fast to build rich UIs and charts, mature ecosystem | Frontend in TS; webview dependency |
| **iced** | Pure Rust, Elm architecture, good GNOME fit (used by COSMIC) | Fewer ready-made widgets (charts, tables) |
| **egui** | Pure Rust, very fast to iterate, plotting via egui_plot | Immediate-mode look, less "native" |
| **Slint** | Declarative, polished | Licensing (GPL/royalty-free tiers) |



---

## 10. CLI (parity with GUI for scripting)
```
contio hw                               # hardware report
contio search qwen3 --device gpu --fits
contio install qwen3-30b-a3b --quant q4_k_m --device gpu
contio run coder                        # load profile/alias
contio ps | stop <model>
contio bench qwen3-30b-a3b:q4_k_m@gpu qwen3-30b-a3b:q4_k_m@cpu --preset quick
contio serve                            # run daemon in foreground
```

---

## 11. Milestones

| # | Milestone | Scope |
|---|---|---|
| **M0** | Spikes (1–2 wk) | NPU runtime viability on Fedora 44; Vulkan vs ROCm on gfx1150; GUI framework prototype |
| **M1** | Headless core | Workspace skeleton, hw probe (both memory pools), llama.cpp CPU+Vulkan install (all three install methods), GGUF download, daemon + OpenAI gateway (chat/completions/embeddings/rerank/models, streaming, tools, vision), Responses + Messages translation, per-pool eviction, CLI |
| **M2** | Catalog & GUI | Curated registry + HF search, fit estimator, GUI: catalog, install, dashboard, jobs, setup wizard (install method), whisper.cpp backend |
| **M3** | Benchmarks | Perf harness + power, quality subsets, tool-calling suite, comparison views |
| **M4** | NPU | Integrate the winning NPU runtime from M0, NPU catalog subset |
| **M5** | Polish | Tray, polkit helper, NPU+GPU hybrid, RPM package + systemd user unit, Responses/Messages edge cases |

---

## 12. Decisions Log

| # | Topic | Decision |
|---|---|---|
| 1 | GUI | **Tauri 2** + **Svelte 5** / TypeScript, a separate client of `contiod` |
| 2 | Backend install | **User-selectable** in the setup wizard (source / prebuilt / podman), can be changed per backend later |
| 3 | API surfaces | **Chat Completions + Responses + Anthropic Messages**, each toggleable, all on by default |
| 4 | Concurrency | **Multiple models at once**, memory-aware eviction **per pool** |
| 5 | Memory | 128 GB unified. Currently 64 GB GPU carve-out and 64 GB system. Pools are detected at runtime |
| 6 | NPU | **FastFlowLM** runtime; catalog imported from FLM's model list; single-slot scheduling; **hybrid NPU+GPU post-v1** |
| 7 | Quality benchmarks | **Coding, tool calling, reasoning/math, general knowledge**. Code runs in podman sandboxes |
| 8 | Network | **Localhost by default**, LAN opt-in that requires an API key |
| 9 | Root actions | **Copy-able commands in v1**, polkit helper with a fixed allowlist post-v1 |
| 10 | Model types in v1 | **Chat LLMs, embeddings, rerankers, vision (VLM), speech (Whisper)** |
| 11 | Packaging | **Dev builds now**, RPM + systemd user unit later |

## 13. Status & Next Steps

**M0: done.** Spikes in `docs/spikes/` (NPU, GPU/CPU, GUI). ROCm and the `performance` profile are still open.

**M1: done** (merged to `main`):
- [x] Cargo workspace (§3): `contio-core`, `contio-hw`, `contio-backends`, `contiod`, `apps/gui`
- [x] `Backend` trait with **llama.cpp** (GPU/CPU) and **FastFlowLM** (NPU) adapters. FLM env is built explicitly.
- [x] `ModelManager`: on-demand load, **per-pool budgets with LRU eviction**, pinned models, idle TTL, request leases (no eviction mid-stream), crash detection, ordered shutdown. Tested against real child processes (`contio-fake-backend`).
- [x] Gateway: `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` with SSE passthrough, `/v1/models`, OpenAI-shaped errors. Management: `/api/models`, `/api/models/{id}/load|unload`.
- [x] Config file registry (`docs/config.example.toml`) until the catalog exists.
- [x] Verified end-to-end on the Framework 13: GPU (Qwen3.6 35B-A3B), CPU (Llama 1B), and NPU (Qwen3 4B + co-loaded embeddings) served at the same time. Tool calls and streaming work through the gateway.
- [x] `contio` CLI: `status`, `hw`, `models`/`ps`, `load`, `unload`, `chat` (streaming, stdin, speed stats), `--json`, `--url`/`CONTIO_URL`
- [x] GUI: model list with load/unload, live state and errors; API card with OpenAI/Anthropic endpoints; shared styles in `app.css`
- [x] Responses API (incl. streaming, function calls, `previous_response_id` via in-memory store) + Anthropic Messages (incl. streaming, tool use/results, thinking blocks, `count_tokens` estimate). Verified with the official OpenAI and Anthropic Python SDKs (`scripts/e2e/sdk_test.py`).
- [x] Gateway guards: identical consecutive tool calls collapsed (non-stream + stream; streamed tool calls are buffered until the end), `response_format` validated (non-stream), reported under a `contio` response field
- [x] Model aliases (`[[aliases]]`), including `upstream_model` to address FLM's co-loaded embedding model under its own name
- [x] Background jobs (`contio-install`): progress, logs, cancellation, one writer per target (duplicate requests join the running job)
- [x] Model downloads: Hugging Face GGUF with resume (HTTP Range) + sha256 (`x-linked-etag`), gated repos via `HF_TOKEN`; FastFlowLM models via `flm pull`
- [x] Backend installers with preflight (`dnf install` hint): llama.cpp **source** (Vulkan, ~2 min), **prebuilt** (GitHub release, sha256 digest, smoke test), **podman** (nearest available `server-vulkan-bN` image); FastFlowLM **source**. Installs are discovered at startup and registered live after install.
- [x] Verified end-to-end in an isolated data dir: all four install paths, download cancel/resume, GPU/CPU chat on each llama.cpp runtime (incl. inside the container), NPU chat + embeddings on a contio-installed FLM.

**M2: done** (branch `m2-catalog`):
- [x] Curated catalog (`registry/catalog.toml`): 47 models in 13 families (chat, vision, code, embeddings, rerankers). Every GGUF quant of each repo is listed live from Hugging Face (cached 24 h on disk, stale cache used offline); draft/MTP heads, imatrix files and incomplete shard sets are filtered out.
- [x] Fit estimator: weights + projector + KV cache (exact per-token geometry from the GGUF headers in `registry/geometry.json`: sliding-window, hybrid linear attention, MLA, shared-KV layers) + compute buffers against each pool's budget → fits / tight / too big. Bits per weight and a quality tier per quant.
- [x] NPU catalog from FastFlowLM's `model_list.json` (XDNA2 entries), linked to curated models where they match; FLM-only models in their own family.
- [x] Hugging Face search for any GGUF repo ("community", conservative KV estimate).
- [x] One-step install (`POST /api/catalog/install`, `contio install`, GUI): installs a missing backend with the method from settings (llama.cpp default: prebuilt), downloads all shards + vision projector with resume and sha256, registers the model, optionally starts it. Re-installing the same files is idempotent.
- [x] Models added through the API persist (`models.json`), can be changed (`PATCH`: context, pinning, device, TTL) and removed (optionally deleting files inside contio's model dir). `config.toml` models are fixed and win on id clashes.
- [x] Model kinds: embedding (`--embedding`) and reranker (`--reranking`, `/v1/rerank`); vision via `--mmproj` (also mounted for Podman). FLM embedding models run standalone on the NPU; FLM serializes several NPU processes itself.
- [x] Jobs persist (`jobs.json`, last 100); jobs cut off by a restart resume automatically (downloads continue from `.part`); failed jobs can be retried.
- [x] GUI: navigation; Catalog (filters, family groups, fit badges, Hugging Face search, detail dialog with quant/device picker and install); Jobs (progress, logs, cancel/retry); Backends (setup wizard: method choice saved as the default, prerequisite check with copyable `dnf` command, install); model removal on the dashboard. Deep link `/catalog?model=<id>`.
- [x] CLI: `search [--hf]`, `show`, `install`, `rm`, `set`, `retry`, `backends [install]`.
- [x] Verified on the Framework 13 in an isolated data dir: catalog/vision/embedding/reranker/NPU installs, chat + vision answer, kill -9 during a download → resumed after restart, fresh machine without backends → llama.cpp installed automatically, SDK end-to-end test (11/11).
- [x] Speech to text: **whisper.cpp** backend (`whisper-server` v1.9.5, Vulkan or CPU, source install, ~2 min; installed automatically like llama.cpp). `/v1/audio/transcriptions` and `/v1/audio/translations` (multipart, up to 200 MB; json/text/srt/vtt/verbose_json). Catalog family with 8 Whisper models, quants listed live from `ggerganov/whisper.cpp`. `contio transcribe`. Verified: WAV/MP3, German, translation (small), subtitles, OpenAI SDK.
- [x] NPU Whisper via FastFlowLM: FLM only serves Whisper co-loaded next to an LLM (`flm serve <llm> --asr 1`; `whisper-v3:turbo` alone is rejected as "non-llm"), so it's `asr = true` under `[backends.flm]` plus an alias with `upstream_model = "whisper-v3:turbo"`. Verified: 3 s per clip after load; FLM doesn't translate.
- [x] Model editor on the dashboard (context, device, pinning, idle TTL, extra args; config models read-only) and aliases through the API (`/api/aliases`, `contio alias`, editor), saved in `aliases.json` and dropped with their model.
- [x] Fit badges for any context: `?context=` on the catalog detail endpoints, `contio show -c`, live in the install dialog (re-suggests the quant unless one was picked).
- [x] Playground: streaming chat with reasoning, tool definitions and a tool-call inspector (send results back), image input, speed stats and the raw request; speech mode with file upload or microphone (recorded to 16 kHz WAV in the browser).
- [x] Daemon stops its backends on SIGTERM/SIGHUP (was Ctrl-C only); per-backend pid files let the next start stop backends orphaned by a crash or `kill -9`.

**M3: in progress** (branch `m3-bench`):
- [x] `contio-bench` performance harness through the daemon's own OpenAI endpoint (the path clients use): prefill and decode tok/s per prompt length (prompts calibrated to the model's tokenizer with two measurements, unique nonce per prompt against prompt caches, llama.cpp `timings` preferred over client timing), TTFT, N parallel requests; SoC power from the `PPT` sensor at 10 Hz → watts and J/token; idle power; load time; VRAM/system pool growth and backend RSS (mapped weights don't show as "used" on CPU); estimator prediction stored next to it.
- [x] Presets quick (512 tokens, 1 run) / standard (512 + 4k, 3 runs, 4 parallel) / full (+16k, 5 runs). Run conditions recorded: power profile, AC/battery, backend + path, other loaded models.
- [x] Runs as a background job (one at a time): unloads other idle models (optional), reloads the model for load time and memory, holds a lease so it isn't evicted.
- [x] Results in SQLite (`bench.db`: runs + headline metrics), list/detail/delete API, CSV/JSON export. CLI `contio bench run|results|show|export|rm`.
- [x] Tool-calling suite (`registry/bench/tool_calls.json`, 23 cases written for contio): simple, choose among 11 tools, parallel, multi-turn (results, chained lookups, error results), no-call when no tool fits, JSON schema output. Type-strict argument checks; pass rate per category; **agent-ready** = ≥ 90% overall and ≥ 75% in every category. Requests use the model's own sampling: greedy decoding sent Qwen thinking models into 4096-token loops.
- [x] Quality subsets (all MIT, fetched with a fixed seed by `scripts/bench/fetch_subsets.py`): reasoning = 50 GSM8K problems, knowledge = 56 MMLU-Pro questions (4 per subject), coding = 40 HumanEval tasks. Quick runs the first 20 items with a 4096-token budget; standard all items, full all items 3×, 8192 tokens. Truncated answers count as failures and are reported.
- [x] Coding sandbox: generated code runs only in Podman (`python:3.12-alpine`): `--network=none`, read-only root, `--cap-drop=all`, no-new-privileges, 64 pids, 256 MB, 1 CPU, user nobody, 10 s timeout. Verified: network blocked, root read-only, infinite loop killed.
- [x] Power profile per run over D-Bus (`net.hadess.PowerProfiles`, tuned-ppd on Fedora): performance/power-saver are *held* (released automatically if contiod dies), balanced is set and restored; restored after cancellation too.
- [x] GUI Benchmarks page: run form (models, suites, preset, power profile), bar charts per metric (latest run per model), speed-vs-tool-calling scatter, pass rate per quality suite, all runs with detail (per prompt length / per case or item with first failure), CSV/JSON export.
- [x] Measured results feed the catalog: installed quants show their latest speed, tool-calling pass rate and agent-ready badge (install dialog, `contio show`).
- [x] Robustness: cancelling a benchmark drops its in-flight request (the backend stops generating); shutdown cancels benchmark jobs and gives open requests 10 s; orphan cleanup skips backends whose parent contiod is still alive.
- [x] Retrieval suite for embedding and reranker models (`registry/bench/retrieval.json`, written for contio): 24 paraphrased queries over 48 documents, half of them same-topic distractors; recall@1 and MRR. nomic-embed-text-v1.5 and Qwen3 Reranker 0.6B: 92% / MRR 0.96 each (an easier first version without distractors gave 100% for both).
- [ ] Whisper (WER) quality; custom prompt sets; arena with Elo; LLM-as-judge
- [ ] Quant-ladder and device-comparison views; "thinking off" variant

Findings so far:
- llama.cpp rejects requests that combine `tools` with a `response_format` schema ("failed to parse grammar"); FastFlowLM accepts them. The gateway could drop the schema grammar or validate afterwards (M5).
- Reasoning (quick, 4096 tokens): Qwen3.5 4B 65%; every failure was a cut-off, every finished answer was right. The model reaches the answer and then keeps second-guessing it (not a loop). Qwen's recommended sampling (temp 0.6, top-k 20, presence penalty 1.5) gave the same score, so the budget, not sampling, decides quick scores; standard uses 8192 tokens.
- Tool calling (quick): Qwen3.5 4B 100% (agent-ready), Qwen3 4B on the NPU 91% (fails chained lookups: invents the user id), Qwen3.6 35B-A3B 87% on the first suite version (before the JSON/no-tools fix; to rerun).

First measurements (Framework 13, performance profile, AC, standard preset):

| Model | Device | Decode tok/s | Prefill 512 / 4k tok/s | TTFT 512 | J/token | Load | Memory (est.) |
|---|---|---|---|---|---|---|---|
| Qwen3.6 35B-A3B Q4_K_M | GPU | 23.0 | 497 / 509 | 1.12 s | 1.55 | 5.1 s | 21.2 GiB (23.2) |
| Qwen3.5 4B UD-Q4_K_XL | GPU | 22.5 | 655 / 641 | 0.84 s | 1.82 | 1.2 s | 4.1 GiB (5.1) |
| Qwen3.5 4B UD-Q4_K_XL | CPU | 18.5 | 194 / 163 | 2.70 s | 2.56 | 1.5 s | 6.9 GiB RSS (4.1) |
| Qwen3 4B (FLM) | NPU | 19.5 | 336 / 552 | 1.53 s | 1.83 | 3.1 s | 8.3 GiB RSS |

Finding: the CPU estimate is too low (llama.cpp's CPU RSS is 1.7× the estimate); GPU estimates are within 10–25% on the safe side.

**Known gaps / M3 candidates**
- whisper.cpp has no prebuilt Linux binaries upstream; Podman images exist but only tagged by commit (`main-vulkan-<sha>`), so only the source install is offered. large-v3-turbo doesn't translate (model limitation).
- NPU models have no memory estimate yet (FLM's `footprint` is shown in the catalog but not used for pool budgeting).
- Shallow source builds of llama.cpp report version `b1`, so the llama-server web UI asset isn't fetched (the API is unaffected).
- llama.cpp Vulkan on gfx1150 runs flash attention over K/V on the CPU ("missing support" warning). Benchmarked: `--flash-attn on` is still +9% prefill at 4k tokens, so it stays the default.
- Podman containers use the daemon's `XDG_DATA_HOME` storage (a custom value isolates them from the user's default storage).
