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
- **GPU**: llama.cpp **Vulkan (RADV)** as the default, because it is the most reliable on gfx1150. **ROCm/HIP** is an optional second GPU backend that is benchmarked against Vulkan.
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

### 5.3 NPU (XDNA 2): highest-risk area ⚠️
NPU LLM support on Linux is young and changes quickly. The candidates are:

| Option | What it is | Pros | Cons |
|---|---|---|---|
| **FastFlowLM (FLM)** | NPU-native LLM runtime with an OpenAI-ish server | Fast, NPU-only, simple CLI/server | Own model format, limited model list, licensing to check |
| **Lemonade Server** (AMD) | OpenAI-compatible server over llama.cpp + OGA + FLM | Supports NPU and Hybrid (NPU prefill + iGPU decode) | Python stack; Linux NPU support still maturing |
| **ONNX Runtime GenAI + Ryzen AI SW** | AMD's official stack | Official, hybrid mode | Heavy install, Linux support/version churn |

Plan: **Milestone 0 spike** to verify which of these works on Fedora 44 / kernel 7.2 with the installed `amdxdna` + `/opt/xilinx` XRT. **Decided:** in v1 the NPU catalog is a **small curated subset** (only models pre-converted for the chosen runtime, e.g. small chat models and embeddings). The GUI must make this limit clear instead of hiding it. **Hybrid** (NPU prefill + GPU decode) is a **post-v1 goal** (M5+), and the `DeviceKind::Hybrid` variant exists from the start so it can be added without breaking anything.

### 5.4 Later backends (post-v1, behind the same trait)
- **mistral.rs** (Rust-native; useful for an in-process option).
- **Hybrid** NPU+GPU modes (via Lemonade/OGA).
- **Remote passthrough**: proxy a cloud OpenAI-compatible endpoint, useful for benchmarking local against cloud.

---

## 6. Auto-Setup (background jobs)

When the user selects **Model X / Quant Y / Device Z** and clicks **Install & Run**:

1. **Preflight**: hardware probe, check driver/permission requirements (render group, `/dev/accel` access, memlock ulimit for the NPU), fit check, disk space check.
2. **Backend install** if missing: download or build, verify checksum, smoke test (`--version` plus a tiny model load).
3. **Model download**: parallel chunked HTTP from HF with resume, sha256 verification, optional HF token for gated models.
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
- **Power & energy**: package power via hwmon / `amd_pmf` / RAPL → **joules per token** (important on battery)
- **Concurrency**: throughput with N parallel requests
- Run conditions are recorded (AC vs battery, power profile, temperature, backend version) so results stay comparable.

### 8.2 Quality benchmarks
- **Decided: all four areas**, each as a small, fast subset with a fixed seed so results are reproducible:
  - **Coding**: HumanEval / MBPP style. Generated code runs **only in podman sandboxes** (no network, CPU/memory/time limits).
  - **Tool calling**: BFCL-style single, parallel, and multi-turn function calls, checked via the real API path, so chat-template bugs show up. This is the most important area for agentic use.
  - **Reasoning / math**: GSM8K / MATH subsets.
  - **General knowledge**: MMLU-Pro subset.
- Other model types: embedding retrieval (small MTEB-style subset), reranker ranking quality, and Whisper word error rate on a small audio set.
- **Custom prompt sets**: user-defined YAML with expected answers or regex/JSON-schema checks.
- **Side-by-side arena**: the same prompt sent to 2–4 models, with blind voting stored as Elo.
- Optional LLM-as-judge using another local (or remote) model.

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
| 6 | NPU | Curated model list in v1, **hybrid NPU+GPU post-v1** |
| 7 | Quality benchmarks | **Coding, tool calling, reasoning/math, general knowledge**. Code runs in podman sandboxes |
| 8 | Network | **Localhost by default**, LAN opt-in that requires an API key |
| 9 | Root actions | **Copy-able commands in v1**, polkit helper with a fixed allowlist post-v1 |
| 10 | Model types in v1 | **Chat LLMs, embeddings, rerankers, vision (VLM), speech (Whisper)** |
| 11 | Packaging | **Dev builds now**, RPM + systemd user unit later |

## 13. Next Steps
1. M0 spikes: NPU runtime on Fedora 44, Vulkan vs ROCm on gfx1150, minimal Tauri + Svelte shell talking to a stub daemon.
2. Detail specs per area: catalog schema + registry TOML format, `Backend`/`BackendInstaller` traits, management API (OpenAPI), benchmark result schema.
3. Scaffold the Cargo workspace (§3).
