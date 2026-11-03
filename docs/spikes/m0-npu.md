# M0 Spike: NPU on Linux

> Status: **Phase 1 + 2 done.** Follow-ups listed in §7.
> Date: 2026-11-03 · Machine: Framework 13, Ryzen AI 9 HX 370, Fedora 44, kernel 7.2.14

## Questions this spike answers
1. Does the NPU work on this machine and OS, without hacks?
2. Which runtime should contio use for the NPU on Linux?
3. Which models can run on it, and how does contio get that list?
4. What does the runtime's API support (streaming, tools, JSON schema, embeddings, usage)?
5. How fast and how efficient is it compared with CPU/GPU? *(phase 2)*
6. Can the NPU host more than one model or process at a time? *(phase 2)*

---

## 1. System readiness ✅ (verified)

| Check | Result | Required |
|---|---|---|
| Kernel driver | `amdxdna` 0.10.0 in-tree (kernel 7.2.14) | kernel 7.0+ or DKMS |
| NPU firmware | **1.1.2.64** (`amdnpu/17f0_10`) | ≥ 1.1.0.0 |
| XRT | **2.27.0** + `xrt_plugin-amdxdna` (RPMs, `/opt/xilinx/xrt`) | present |
| Device | `/dev/accel/accel0`, mode `0666`, user in `render` group | accessible |
| memlock | `unlimited` | high/unlimited |
| `xrt-smi examine` | NPU Strix, `aie2p`, 6×8 topology | detected |
| `xrt-smi validate` | **PASSED**: 47.4 TOPS, 65 µs latency, ~94k ops/s | passes |
| Build deps | cmake 4.3, ninja, gcc, boost-devel, libcurl-devel, libedit-devel | present |

**Conclusion:** the NPU hardware/driver stack is fully working. Nothing needs root to set up.

## 2. Runtime choice → **FastFlowLM (FLM)** ✅

| Option | Linux NPU status (Nov 2026) | Verdict |
|---|---|---|
| **FastFlowLM** (`ROCm/FastFlowLM`) | Native Linux since 2026-03; XDNA2 only; v1.0.x; now under the ROCm org | **Use directly** |
| Lemonade Server | Its Linux NPU support (since 10.0) **is FLM under the hood** | Redundant layer; skip for NPU |
| ONNX Runtime GenAI / Ryzen AI SW | No mature Linux LLM path found | Skip; revisit for hybrid post-v1 |

FLM facts that matter for contio:
- **License**: orchestration/CLI is MIT. The NPU kernels are closed-source binaries, free for any use (including commercial), and may only be redistributed **unmodified** (FastFlowLM NPU Binary License). They ask for "Powered by FastFlowLM" attribution → add it to contio's About/credits.
- **Build without sudo**: `src/home_install.sh` installs into a user prefix (`FLM_PREFIX`) and writes `flm_env.sh`. This fits contio's "build from source" install method directly. There is also a `linux-portable` preset that bundles XRT (candidate for the "prebuilt" method).
- **Model storage**: `FLM_MODEL_PATH` overrides the location → contio points it at `~/.local/share/contio/models/flm`.
- **CLI**: `flm validate --json`, `flm list --json`, `flm pull <tag>`, `flm check <tag>`, `flm remove <tag>`, `flm serve <tag> --port N --ctx-len N --pmode {powersaver,balanced,performance,turbo}`.
- **Two engines**: `--backend flm` (NPU2 format, from the `FastFlowLM/*-NPU2` HF repos) and `--backend core` (ryzenai-corelib, which runs **standard GGUF** from unsloth/bartowski; off by default in source builds).

## 3. Model catalog for the NPU ✅

FLM ships a **machine-readable catalog**: `src/model_list.json` (~70 tags). Each entry has the HF URL, parameter size, quant level (Q4_0/Q4_1/Q4_K), default context, `footprint` (GB), `vlm`, `think`, and labels (`tool-calling`, `vision`, `audio`, `embeddings`, `reasoning`, `speculative`, …).

→ **contio imports this file directly** into the NPU slice of the catalog and maps it onto Family → Model → Format(`FLM-NPU2`) → Quant. No hand curation is needed beyond picking defaults.

Highlights at the pinned commit:

| Use | Tags (footprint) |
|---|---|
| Agentic / tool-calling | `qwen3-it:4b` (3.1 GB), `qwen3:8b` (5.6), `qwen3.5:4b/9b` (3.9/8.7), `gemma4-it:e4b/12b` (5.4/7.5) |
| Large (MoE / 27B) | `qwen3.6-moe:35b-a3b` (22.1 GB), `qwen3.8-mtp:27b` (18.1), `gpt-oss:20b` (14.0) |
| Embeddings | `embed-gemma:300m` (0.6) |
| Speech | `whisper-v3:turbo` (0.6) |
| Vision | `qwen3vl-it:4b`, `gemma3:4b`, `minicpm-v-4.7:1b` |
| Tiny / fast | `llama3.2:1b`, `qwen3:0.6b`, `lfm2.5-it:1.2b` |

This is much broader than the spec assumed ("small curated subset"). It now includes 35B MoE, tool calling, embeddings, and Whisper on the NPU.

⚠️ `*-flash` variants have a 1024-token context and are `single-turn` only → hide them from agentic presets.

## 4. API surface (from source, `src/server/`) ✅

| Endpoint | Present | contio use |
|---|---|---|
| `/v1/chat/completions` (+ SSE, `stream_options.include_usage`) | ✅ | main path |
| `tools` / `tool_calls` | ✅ (xgrammar-constrained) | agentic |
| `response_format` JSON schema | ✅ (xgrammar) | structured output |
| `/v1/completions` | ✅ | |
| `/v1/embeddings` | ✅ | embeddings on NPU |
| `/v1/audio/transcriptions` | ✅ | Whisper on NPU |
| `/v1/models`, `/api/ps`, `/api/npu/status`, `/api/version` | ✅ | health / status for contio |
| per-request `"backend"` field | ✅ | ignore; contio pins the backend |

**Server semantics:** one `flm serve` process holds **one model**. A request for a different model makes it **swap** (unload → load). Community reports say FLM holds an **exclusive NPU lock** while a model is loaded → **not confirmed**, see §5 (lock is per request).

## 5. Phase 2: measurements ✅

The build step was not run from the assistant session (third-party code builds need your go-ahead). To run it:

```bash
scripts/spikes/npu-flm.sh                         # build + llama3.2:1b, qwen3-it:4b, qwen3:8b, qwen3.6-moe:35b-a3b
SKIP_BUILD=1 scripts/spikes/npu-flm.sh gpt-oss:20b   # more models later
```

- Builds FLM at a **pinned commit** into `~/.local/share/contio-spike/flm` (no sudo), with models in `~/.local/share/contio-spike/models`.
- Downloads about 32 GB for the default model set.
- For each model, `scripts/spikes/npu_probe.py` measures TTFT, prefill tok/s (~2k and ~8k prompts), decode tok/s, **average SoC power (amdgpu `PPT` sensor, no root) and J/token**, plus tool-call, JSON-schema, and embedding-swap latency.
- Concurrency check: does a 2nd `flm serve` work while the first holds a model?
- Results: `~/.local/share/contio-spike/results/<timestamp>/` (`*.jsonl`, logs).

Run on **AC power** with the GNOME power profile noted. For comparison, the same prompts should later run through llama.cpp Vulkan/CPU in the GPU spike.

### Results (run `20261103-091127`, AC power, GNOME **power saver**, FLM `--pmode performance`)

FLM build + install (`home_install.sh`): **131 s**. Downloads ran at ~40 MB/s (22 GB MoE in 660 s).

| Model (NPU2, quant) | Load | Decode tok/s | Prefill tok/s (1.1k / 4.5k prompt) | TTFT 4.5k | Avg SoC W | J/tok* | Tools | JSON schema |
|---|---|---|---|---|---|---|---|---|
| llama3.2:1b (Q4_1) | 3.9 s | **57–61** | 865 / 1283 | 3.4 s | 13–14 | 0.26 | ❌ (not a tool model) | ✅ |
| qwen3-it:4b (Q4_1) | 4.4 s | 18–19 | 397 / 454 | 9.9 s | ~15 | 0.89 | ✅ | ❌ ¹ |
| qwen3:8b (Q4_1) | 5.9 s | 11 | 278 / 321 | 14.0 s | 15–16 | 1.58 | ✅ | ❌ ¹ |
| qwen3.6-moe:35b-a3b (Q4_K) | 22.5 s | 16–17 | 78 / 131 | **34.4 s** | ~11 | 0.90 | ❌ ² | ✅ |

\* J/token on the 256-token story generation (`short_prompt`). Measured on the amdgpu `PPT` sensor, so it is whole-SoC power including idle baseline.
¹ The model reasoned in free text ("Okay, the user wants…") and ran out of the 128-token budget before emitting JSON. **Corrected in the balanced re-run: with a 1024 budget, JSON passes on all NPU models** (see [m0-gpu.md §4](m0-gpu.md)). This was a probe-budget issue, not an FLM bug.
² **Hallucinated the tool result** ("partly cloudy, 18 °C") instead of emitting a tool call, despite the `tool-calling` label. FLM chat format was `qwen_xml with reasoning`. **Reproduced in balanced with a 2048 budget.** The same model calls the tool correctly via llama.cpp, so this is an FLM issue to report upstream.

**Concurrency:** a 2nd `flm serve` process loaded a model (2.98 s) **while the first held Llama on the NPU**. The NPU lock is **per request** (`NPU Locked!` … `Lock Released!`), not per loaded model. The community report of an exclusive lock does not apply to this FLM version. *(The embedding call in that check returned 200 with no data, see below. True parallel two-LLM load is a follow-up.)*

**Embeddings:** `/v1/embeddings` with `embed-gemma:300m` returns *"No embedding model loaded"* (HTTP 200, `data: null`). Per FLM docs the embedding model is **co-loaded next to an LLM** via `flm serve <llm> --embed 1`. It is not swapped in on request.

### Takeaways
1. **The NPU is a low-power device for small models, not the main agentic engine.** Decode speed is fine for ≤4B (18–60 tok/s at ~13–15 W), but **prefill is slow** (130–450 tok/s for 4B+). Agentic tools resend large contexts every turn, so TTFT of 10–35 s at 4.5k tokens is too slow for coding agents.
2. **Best NPU roles for contio:** embeddings (co-loaded), Whisper, small always-on helper models (classification, titles, autocomplete-ish), and battery-mode chat. The GPU spike must confirm the GPU is better for the 8B–35B agentic models.
3. **Engine "tool-calling" labels can't be trusted.** contio's tool-calling benchmark must gate the "agent-ready" badge per (model, backend), and **contio's gateway should validate `response_format` output** against the schema and report violations.
4. Dense-model decode scales worse than size alone suggests (1B→4B: 60→18 tok/s), so the NPU path appears to use less memory bandwidth than the GPU would. MoE helps (35B-A3B decodes faster than dense 8B).
5. **Embeddings work co-loaded** (`--embed 1`): 768-dim in 0.17 s next to any LLM (balanced re-run).

## 6. Consequences for the spec

1. **NPU backend = FLM**, wrapped as a `Backend` that spawns `flm serve` as a child process (same as llama-server). Lemonade is not needed.
2. **NPU catalog source = FLM's `model_list.json`**, refreshed when the FLM version changes. The NPU offer is much larger than planned.
3. **NPU scheduling model**: an FLM process = **one LLM (+ optional co-loaded embedding model via `--embed 1`)**. Multiple FLM processes can coexist; the NPU is time-shared per request. contio can therefore run several NPU models in separate processes, with the system-memory pool budget (§7.2 of the spec) as the limit. Parallel requests to two NPU processes will queue on the NPU lock, so throughput is shared, not added.
4. **Install methods for FLM**: *source* = `home_install.sh` into contio's prefix (verified build deps present); *prebuilt* = upstream `.deb`/snap aren't usable on Fedora, so "prebuilt" for FLM means the `linux-portable` build artifact cached by contio; *podman* = needs `/dev/accel` + memlock passthrough (validate later).
5. **Power telemetry**: use the amdgpu hwmon `PPT` sensor (SoC package power, user-readable) for J/token. RAPL `energy_uj` needs root. On battery, `BAT1` `current_now × voltage_now` is a second source.
6. **Don't source upstream env scripts**: at the pinned commit, `home_install.sh` writes a developer's hardcoded `FLM_MODEL_PATH=/scratch/alfxu` into `flm_env.sh`, which broke every `flm pull` on the first run. contio must build the child-process environment itself (`FLM_CONFIG_PATH`, `FLM_XCLBIN_PATH`, `FLM_MODEL_PATH`, `LD_LIBRARY_PATH`, XRT vars) and never source `flm_env.sh`. Worth reporting upstream.
7. **Power profile**: the first run was in GNOME *power saver* (quiet environment). FLM sets the NPU mode separately (`--pmode`, default `performance`), but host-side work (tokenization, sampling, HTTP) is slower. **Every device must be benchmarked in the same profile**, and contio's benchmark runner must record the active profile.
8. **Attribution**: "Powered by FastFlowLM" in About/credits. FLM binaries are redistributed only unmodified.

## 7. Follow-ups
- [ ] Probe fixes: start the server with `--embed 1` for the embedding test; raise the JSON-schema `max_tokens` and disable thinking for that test; check response bodies, not just HTTP status, in the concurrency check.
- [ ] Two LLM processes under **parallel** load: per-process throughput and latency.
- [ ] qwen3.6-moe tool calling: reproduce with `tool_choice: "required"` and a system prompt. Check FLM's `qwen_xml` parser. Report upstream if confirmed.
- [ ] `--pmode turbo` vs `performance` vs `powersaver`: speed and power trade-off.
- [ ] Re-run in the *balanced* profile alongside the GPU spike for a fair cross-device comparison.
- [ ] Report the hardcoded `FLM_MODEL_PATH=/scratch/alfxu` in `home_install.sh` upstream.

## Sources
- [FastFlowLM GitHub (ROCm/FastFlowLM)](https://github.com/ROCm/FastFlowLM)
- [FastFlowLM Linux install docs](https://fastflowlm.com/docs/install_lin/)
- [FastFlowLM model list](https://fastflowlm.com/docs/models/)
- [Lemonade: LLMs on Linux with FastFlowLM](https://lemonade-server.ai/flm_npu_linux.html)
- [Phoronix: AMD Ryzen AI NPUs are finally useful under Linux for LLMs](https://www.phoronix.com/news/AMD-Ryzen-AI-NPUs-Linux-LLMs)
- [Framework Community: NPU (XDNA2) with Arch Linux and FastFlowLM](https://community.frame.work/t/guide-use-npu-xdna2-with-arch-linux-and-fastflowlm/80879)
