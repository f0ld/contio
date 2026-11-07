# M0 Spike: GPU (Vulkan) & CPU via llama.cpp, plus a cross-device comparison

> Status: **Done.** Power saver and balanced (§2–4); ROCm vs Vulkan and `performance` added 2026-11-07 (§8).
> Date: 2026-11-03 · Run `gpu-20261103-114636` · llama.cpp `b12068` (4402e65) · Mesa 26.2.5 (RADV)
> Same machine, models, and probe as [m0-npu.md](m0-npu.md).

## 1. Setup

| Item | Value |
|---|---|
| Build | `scripts/spikes/gpu-llamacpp.sh`, Vulkan backend, `GGML_NATIVE=ON`, no sudo, **164 s** |
| Build deps | `vulkan-headers`, `vulkan-loader-devel`, `glslc`, **`spirv-headers-devel`** (was missing → first build failed) |
| ROCm | see §8 (ROCm 7.1.1, Fedora packages) |
| GPU memory | 64 GiB VRAM carve-out + 31 GiB GTT |
| Models | Q4_K_M GGUF (bartowski/unsloth), sha256-verified against HF |
| GPU flags | `-ngl 999 -fa on -c 16384 --jinja` |
| CPU flags | `-ngl 0 --device none -t 12` |
| Conditions | AC power, GNOME **power saver** (same as the NPU run) |

## 2. Results

### llama-bench (tok/s, flash attention, 3 reps)

| Model | GPU pp512 | GPU pp4096 | GPU tg128 | CPU pp512 | CPU pp4096 | CPU tg128 |
|---|---|---|---|---|---|---|
| Llama 3.2 1B | 2557 | 1753 | **79.3** | 364 | 289 | 63.1 |
| Qwen3 4B Instruct | 479 | 402 | **26.7** | 101 | 77 | 22.0 |
| Qwen3 8B | 292 | 257 | **14.2** | 57 | 46 | 12.7 |
| Qwen3.6 35B-A3B (MoE) | 372 | 359 | **22.0** | 80 | 73 | 15.0 |

### Cross-device comparison (API probe, ~4.5k-token prompt, power saver)

Decode values come from `llama-bench` tg128 (GPU/CPU) and from the FLM probe (NPU).

| Model | Device | Load | Decode tok/s | TTFT ~4.5k | Avg SoC W | J/tok* | Tools | JSON |
|---|---|---|---|---|---|---|---|---|
| **Llama 3.2 1B** | NPU | 3.9 s | 57–61 | 3.4 s | 13–14 | 0.26 | ❌ | ✅ |
| | GPU | 2.0 s | **79** | **2.1 s** | 18.5 | 0.27 | ✅ | ✅ |
| | CPU | 2.0 s | 63 | 12.6 s | 15 | **0.24** | ✅ | ✅ |
| **Qwen3 4B** | NPU | 4.4 s | 18–19 | 9.9 s | 15 | 0.89 | ✅ | ❌¹ |
| | GPU | 2.0 s | **26.7** | **9.4 s** | 15 | **0.57** | ✅ | ✅ |
| | CPU | 3.1 s | 22 | 48.5 s | 15 | 0.71 | ✅ | ✅ |
| **Qwen3 8B** | NPU | 5.9 s | 11 | **14.0 s** | 15–16 | 1.58 | ✅ | ❌¹ |
| | GPU | 3.1 s | **14.2** | 15.1 s | 15 | **1.07** | ✅ | ✅ |
| | CPU | 4.1 s | 12.7 | 76.0 s | 15 | 1.20 | ✅ | ✅ |
| **Qwen3.6 35B-A3B** | NPU | 22.5 s | 16–17 | 34.4 s | 11 | 0.90 | ❌² | ✅ |
| | GPU | 6.1 s | **22.0** | **10.5 s** | 15 | **0.70** | ✅³ | ✅ |
| | CPU | 8.1 s | 15 | 54.2 s | 15 | 1.04 | ✅ | ✅ |

\* J/token measured on the 256-token story generation, using whole-SoC power (amdgpu `PPT`) including baseline.
¹ NPU run used a 128-token budget for JSON (thinking ran out of room). GPU/CPU runs used 1024, so these aren't directly comparable.
² FLM: **hallucinated a tool result** (`finish_reason: stop`). This is a real failure.
³ Failed in the main run with a 512-token budget (`finish_reason: length`). The recheck (below) passed 3/3 with 2048.

### Tool-call recheck: Qwen3.6 35B-A3B on GPU
| Budget | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| 512 | ⚠️ **same call repeated 7×** until `length` | ✅ (131 tok) | ❌ `length`, still reasoning |
| 2048 | ✅ (133 tok) | ✅ (111 tok) | ✅ (404 tok) |

- `temperature: 0` is **not deterministic** on Vulkan: three runs gave three different reasoning lengths (40–254 words).
- **Repetition loop**: identical tool calls emitted back-to-back until the token limit.

## 3. Key finding: power saver caps the SoC at ~15 W

Average SoC power sits at **~15.0 W in almost every GPU and CPU test**, whatever the device or model. The power saver profile pins the package power limit, so in this run **all devices share the same ~15 W budget**. Consequences:
- The GPU's compute advantage is mostly cancelled for dense models: Qwen3 8B prefill is GPU 257–299 vs NPU 278–321 tok/s.
- The NPU's one clear advantage here is *drawing less* than the cap on some models (11 W on the MoE), not more speed per watt.
- These numbers show **"what you get on a quiet, cool laptop"**, which is a valid use case, but **not peak capability**. A balanced/performance re-run is needed before deciding device defaults for plugged-in use.

## 4. Balanced re-run (runs `20261103-123941` NPU, `gpu-20261103-124502` GPU/CPU)

Same models, binaries, and probe (tool budget now 2048, JSON 1024, embeddings co-loaded on NPU). **Balanced caps the SoC at ~25 W**: GPU and CPU tests sit at a flat 25.0 W. The NPU stays at 15–19 W because it doesn't need the full budget.

### Cross-device, balanced (~4.5k-token prompt)

| Model | Device | Load | Decode tok/s | Prefill tok/s | TTFT ~4.5k | SoC W | J/tok | Tools | JSON |
|---|---|---|---|---|---|---|---|---|---|
| **Llama 3.2 1B** | NPU | 4.5 s | 53–64 | 1627 | 2.7 s | 16–29 | 0.52 | ❌ | ✅ |
| | GPU | 1.0 s | **90.5** | **2191** | **1.7 s** | 25 | **0.28** | ✅ | ✅ |
| | CPU | 1.0 s | 76 | 411 | 8.8 s | 25 | 0.34 | ✅ | ✅ |
| **Qwen3 4B** | NPU | 4.8 s | 20 | 538 | 8.4 s | 18 | 0.96 | ✅ | ✅ |
| | GPU | 2.0 s | **29.5** | **598** | **6.2 s** | 25 | **0.86** | ✅ | ✅ |
| | CPU | 2.0 s | 24.7 | 108 | 35.2 s | 25 | 1.08 | ✅ | ✅ |
| **Qwen3 8B** | NPU | 6.2 s | 11 | **381** | 11.8 s | 18–19 | 1.77 | ✅ | ✅ |
| | GPU | 2.0 s | **16.2** | 355 | **10.1 s** | 25 | **1.55** | ✅ | ✅ |
| | CPU | 3.0 s | 13.3 | 68 | 54.0 s | 25 | 1.97 | ✅ | ✅ |
| **Qwen3.6 35B-A3B** | NPU | 20.9 s | 15–17 | 170 | 26.4 s | 15 | 1.14 | ❌ | ✅ |
| | GPU | 3.0 s | **23.6** | **509** | **7.5 s** | 25 | **1.10** | ✅ (802 tok) | ✅ |
| | CPU | 5.1 s | 18.2 | 106 | 36.8 s | 25 | 1.41 | ✅ | ✅ |

Prefill values are llama-bench pp4096 for GPU/CPU and the API probe for the NPU. The NPU 1B J/tok is inflated by a 28.7 W spike during the short test, likely other system activity.

### Power saver → balanced (GPU, Qwen3.6 35B-A3B)

| | Power saver (15 W) | Balanced (25 W) | Δ |
|---|---|---|---|
| Prefill pp4096 | 359 tok/s | 509 tok/s | **+42%** |
| Decode | 22.0 tok/s | 23.6 tok/s | +7% |
| TTFT ~4.5k | 10.5 s | 7.5 s | −29% |
| J/token | **0.70** | 1.10 | **+57% energy** |

- **Prefill scales with power; decode barely does** (bandwidth-bound). The CPU gains most (+40–55% prefill) and the NPU least (+18–30%).
- **Power saver is the most energy-efficient mode**, and balanced buys speed with energy. This is true for every device.
- **NPU tool calling with Qwen3.6 still hallucinates with a 2048 budget** → a confirmed FLM issue (the same model works via llama.cpp on GPU and CPU).
- **Embeddings co-loaded on the NPU** (`--embed 1`): 768-dim, **0.17–0.18 s**, alongside every LLM size.
- The NPU 2nd-process check is still HTTP-status only (false positive, "No embedding model loaded"), so it is not evidence for or against concurrency.

## 5. Conclusions

1. **GPU (Vulkan) is the best default device** in both profiles (confirmed in balanced, where it is faster than the NPU on every model and metric except Qwen3 8B prefill). It has the fastest decode on every model (+25–45% vs NPU), the fastest TTFT on 3 of 4 models, the lowest J/token on 4B+, the fastest load times, and working tool calls through llama.cpp's `--jinja` templates.
2. **The MoE on the GPU is the agentic sweet spot**: Qwen3.6 35B-A3B gives 22–24 tok/s decode and a 7.5–10.5 s TTFT at 4.5k tokens, faster than the dense 8B at much higher model quality. NPU TTFT for the same model is 34 s.
3. **CPU decodes almost as fast as the GPU** (bandwidth-bound), but its prefill is 4–6× slower → it suits small models, overflow, and running *alongside* the GPU, not long agent contexts.
4. **Power profile = speed/energy dial**: balanced gives +40% prefill at +57% J/token. contio should expose this as a user choice ("quiet/efficient" vs "fast") rather than hide it.
5. **NPU**: no speed advantage over the GPU here. It is useful to keep the GPU free (e.g. embeddings/Whisper/helper models on the NPU while a coding model runs on the GPU) and for lower *peak* draw. Its tool-calling path (FLM) is less reliable.
6. **Engine matters as much as device for agent-readiness**: Llama 1B and Qwen3.6 call tools correctly via llama.cpp but not via FLM.

## 6. Consequences for contio

1. **Default device policy**: GPU for chat/agentic models. NPU for embeddings, Whisper, and small helper models (keeps the GPU free). CPU as overflow and for small models when the GPU is busy. The fit estimator's speed prediction should use **measured** numbers like these, keyed by power profile.
2. **The benchmark runner must record and control the power profile** and report SoC W. The GUI should show a "results from power-saver" badge and suggest re-running when the profile differs.
3. **Benchmarks repeat each quality test N times** (temperature 0 is not deterministic), and report pass rate, not pass/fail.
4. **Tool-call benchmark budget ≥ 2048 tokens** for thinking models (or a "thinking off" variant tested separately).
5. **The gateway detects repetition loops**: N identical consecutive tool calls → truncate to one and log a warning (configurable).
6. **The downloader verifies sha256** (HF `lfs.oid`) and holds **one writer per artifact** (M0: two runs writing one `.part` corrupted it).
7. **Build preflight** for llama.cpp Vulkan: `vulkan-headers`, `vulkan-loader-devel`, `glslc`, `spirv-headers-devel`, `cmake`, `ninja`, `gcc-c++`. Show the `dnf install` line before starting.

## 7. Open follow-ups
- [x] Re-run all three devices in `balanced` (§4).
- [x] `performance` profile on AC: is there headroom above 25 W? (§8: +6% prefill over balanced, decode unchanged)
- [ ] NPU concurrency check: verify response bodies, and test two LLM processes under parallel load.
- [x] ROCm/HIP build for gfx1150 vs Vulkan (§8: no clear win, Vulkan stays the only GPU backend).
- [ ] Parallel load: GPU model + NPU model serving simultaneously. How much does each slow the other (shared bandwidth)?
- [ ] KV-cache quantization (`-ctk q8_0 -ctv q8_0`) and longer contexts (16k–32k) for agentic workloads.
- [ ] Larger models that only the GPU can hold: e.g. gpt-oss-120b (~63 GB), checking the 64 GB VRAM limit.

## 8. ROCm/HIP vs Vulkan (2026-11-07)

Same llama.cpp commit (4402e65) built twice: `build-vulkan` and `build-rocm`
(`-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1150 -DGGML_NATIVE=ON`, 3.5 min). ROCm 7.1.1 from the Fedora 44
packages (`rocm-hip-devel hipblas-devel rocblas-devel rocwmma-devel`); rocBLAS ships gfx1150 kernels and the
890M shows up as `ROCm0` with the 64 GiB carve-out (Vulkan also counts GTT: 97 GiB). The
`GGML_HIP_ROCWMMA_FATTN` option no longer exists in this llama.cpp; HIP uses its own flash-attention kernels.

`llama-bench -ngl 99 -fa 0,1 -p 512,4096,16384 -n 128 -r 3`, AC power, **performance** profile, backends run
one after the other with nothing else on the GPU. Flash attention on (the contio default); tok/s, ROCm relative to Vulkan:

| Model | pp512 Vulkan / ROCm | pp4096 Vulkan / ROCm | pp16384 Vulkan / ROCm | tg128 Vulkan / ROCm |
|---|---|---|---|---|
| Qwen3 4B Instruct Q4_K_M | 773 / 763 (−1%) | 644 / 661 (+3%) | 396 / 420 (+6%) | **29.5** / 27.1 (−8%) |
| Qwen3.5 4B UD-Q4_K_XL | 663 / 698 (+5%) | 646 / 660 (+2%) | **556** / 516 (−7%) | **23.0** / 22.0 (−5%) |
| Qwen3 8B Q4_K_M | 431 / 447 (+4%) | 390 / 410 (+5%) | 283 / 300 (+6%) | **16.3** / 15.5 (−5%) |
| Qwen3.6 35B-A3B UD-Q4_K_M | 562 / 578 (+3%) | 540 / 563 (+4%) | 462 / **504 (+9%)** | **23.5** / 21.6 (−8%) |

Without flash attention the two are within ±5% on prefill (ROCm +10% only for Qwen3.5 4B at 512); decode is the same picture.
Standard deviations are ≤ 1% except a few pp512 runs.

Findings:

1. **No clear winner.** ROCm prefills 3–9% faster on long prompts (not for the hybrid Qwen3.5), Vulkan decodes 5–8% faster
   everywhere. For agentic coding with the 35B at 16k context, ROCm saves ~3 s of a ~35 s prefill and costs ~1.9 tok/s of decode.
2. **Not worth a second backend.** A ROCm backend would add ~1.5 GB of system packages, a slower build and a per-model choice
   for single-digit gains in either direction. Vulkan stays the only GPU backend; worth re-checking with new ROCm or llama.cpp releases.
3. **Flash attention matters more than the backend:** at 16k it doubles prefill for dense models (Qwen3 4B 200 → 396 tok/s) on both backends; for the MoE it is +3–11%.
4. **`performance` profile:** 35B pp4096 is 540 tok/s vs 509 in balanced and 359 in power saver (§4), decode unchanged (23.5 vs 23.6).
   Most of the gain is power saver → balanced; performance adds ~6% on top.
