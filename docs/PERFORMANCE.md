# Performance roadmap

Goal: make contio faster step by step on this machine (Ryzen AI 9 HX 370, Radeon 890M, 64 GB VRAM
carve-out). Every item is **measured** before it ships and lands as a setting in an **Expert ·
Performance** section (per model in the model editor, daemon-wide under Settings), with a short
explanation of what it does and its upsides and downsides. Defaults change only when a setting is a
clear win with no real downside. Measured negative results stay here so they aren't retried blindly.

Reference model: Qwen3.6 35B-A3B UD-Q4_K_M (MoE, 3B active), `performance` power profile.

| # | Item | Speeds up | Status |
|---|---|---|---|
| 1a | Speculative decoding: n-gram | writing (edits) | **shipped** v0.3.7 |
| 1b | Speculative decoding: MTP | writing | **shipped** v0.3.7 |
| 1c | Speculative decoding: draft model | writing (dense models) | **shipped** v0.3.8 |
| 2a | Prompt cache across turns | time to first token | measured: works; agent sessions are limited by 2d |
| 2b | Larger prompt micro-batch | reading | measured: **not shipped** (slower at depth, crashed the GPU) |
| 2c | KV cache precision | memory, writing on Vulkan | **shipped** v0.3.8 |
| 2d | ROCm for long contexts | reading deep in the context | shipped v0.3.6; **re-measured at depth: +30–59%** |
| 2e | Power while serving | reading | **shipped** v0.3.8 |
| 3 | Mixed backends (ROCm reads, Vulkan writes) | – | measured: **not built** (ROCm alone gets the gain) |
| 4a | MoE offload to system memory | bigger models fit | **shipped** v0.3.8 |
| 4b | Preload pinned models | first request | **shipped** v0.3.8 |
| 4c | Benchmarks show these effects | – | **shipped** v0.3.8 (agentic workloads, follow-up turn) |
| 4d | Parallel slots | several clients | already on (llama.cpp default: 4 slots, unified KV) |
| 4e | NPU for side work | GPU stays free | available (put embeddings/Whisper on the NPU) |

## What coding agents need (OpenCode)

A day of OpenCode through contio (2026-11-07): 202 requests, **10.7 M prompt tokens, 89 k generated**.
Agents resend the whole conversation every turn; the prompt cache serves most of it, but the new part
(tool output, file contents) is read **deep in a 50–100k context**, where prompt reading on Vulkan falls
from ~580 to ~125 tok/s. So for agents: **2d (ROCm) matters most, then 1a/1b.**

## 1. Speculative decoding

A cheap guesser proposes the next few tokens; the model checks them all in one pass and keeps the ones
it agrees with. Output is unchanged, only faster when guesses are right. Three workloads, 700 tokens
each, thinking off: *edit* returns a 140-line file with a rename, *code* writes a new program, *prose*
explains a topic.

**35B-A3B (MoE), Vulkan, tok/s:**

| Method | edit | code | prose | Notes |
|---|---|---|---|---|
| none | 22.8 | 22.7 | 22.7 | |
| n-gram (`ngram-simple`) | **106.7 (4.7×)** | ±0 | ±0 | copies spans from the context; idle otherwise |
| MTP (`draft-mtp`) | 39.4 | 34.1 (+50%) | 24.9 | needs a `-MTP` GGUF (+0.5 GB) |
| **n-gram + MTP** | **111 (4.8×)** | **32.8 (+44%)** | 24.6 | best for this model |
| draft Qwen3.5 0.8B | 26.9 | 20.4 (−10%) | 13.4 (−41%) | **loses**: checking drafts wakes many experts |

**Qwen3.5 4B (dense), Vulkan, tok/s:**

| Method | edit | code | prose |
|---|---|---|---|
| none | 21.1 | 15.3 | 14.8 |
| **draft Qwen3.5 0.8B** | **30.9 (+46%)** | **22.2 (+45%)** | 16.3 (+10%) |
| draft + n-gram | 150–189 (8×) | 23.6 | 16.3 |

**Not allowed: n-gram together with a draft model.** With llama.cpp b12068 the pair hangs on some
prompts: reproducible with the perf suite's third 512-token prompt (temperature 0, slot reuse), one
thread spins and the request never finishes. Each method alone and n-gram + MTP (14 requests, same
prompts) are fine. The API refuses the pair; re-test with newer llama.cpp releases.

**Cost on prompt reading** (30k-token prompt of real code, 35B): none 374, n-gram 377 (±0), n-gram +
MTP 350 (−6%) tok/s on Vulkan; ROCm 453 → 431 (−5%) with both. A draft model costs ~20% on short prompts.

Settings: per model `speculative = { ngram, mtp, draft }`; MTP only when the GGUF has
`<arch>.nextn_predict_layers` > 0; a draft must be another installed GGUF with the same vocabulary and
less than half the size. The catalog's Qwen3.6 entries point to the `-MTP-GGUF` repos.

## 2. Faster long prompts

**2a Prompt cache across turns.** Works: llama-server picks the slot with the longest common prefix
(`LCP similarity` in the log) and reads only the new tokens; 8 GiB of host-RAM prompt cache (default)
keeps idle slots. Hybrid models (Qwen3.5/3.6) can only resume from context checkpoints (default 32
per slot, every ≥ 8k tokens). A full re-read seen in a real session was a new conversation (an agent
subagent with its own system prompt), not a cache miss. The benchmark's *follow-up* turn now shows the
effect (`cached_tokens`). Nothing to tune; what's left is the speed of reading the *new* tokens (2d).

**2b Prompt micro-batch (`-ub`).** Measured and **not shipped**:

| tok/s reading 2k tokens at depth | 0 | 32k | 64k | 96k |
|---|---|---|---|---|
| Vulkan ub 512 (default) | 584 | 261 | 189 | 125 |
| Vulkan ub 2048 | 642 | – | 134 | **GPU crash** (device lost, kernel reset) |
| ROCm ub 512 (default) | 573 | 352 | 253 | 197 |
| ROCm ub 2048 | – | 262 | 134 | stopped |

Larger batches only help an empty context and hurt where agents work.

**2c KV cache precision** (at 64k depth, 35B):

| | reading | writing |
|---|---|---|
| Vulkan f16 → q8_0 | 195 → 182 (−7%) | 16.8 → **18.9 (+12%)** |
| ROCm f16 → q8_0 | 254 → 250 (−2%) | 15.8 → **10.5 (−34%)** |

Halves KV memory. Worth it on Vulkan and for dense models with large KV caches (Qwen3.6 27B: 64 KiB per
token, 8 GiB at 128k); never with ROCm. The 35B's KV is small (20 KiB per token).

**2d ROCm for long contexts.** v0.3.6 measured up to 16k (+3–9%). At agent depths the gap is large,
because Vulkan's flash attention falls back to the CPU on this GPU ("Flash Attention over K/V rows not
supported"):

| 35B, tok/s | depth 0 | 32k | 64k | 96k | writing at 64k |
|---|---|---|---|---|---|
| Vulkan | 584 | 261 | 195 | 125 | **16.8** |
| ROCm | 573 | **352 (+35%)** | **254 (+30%)** | **197 (+59%)** | 15.8 (−6%) |

**Recommendation for coding agents: ROCm.** Vulkan stays the default (it writes faster on short
conversations and needs no extra install).

**2e Power while serving.** Daemon setting `serve_power` = off / on AC / always: holds the `performance`
profile while any model works and releases it 10 s after the last request (power-profiles-daemon drops
the hold if contiod dies). Pauses during benchmarks, which pick their own profile. Earlier measurement:
power saver → balanced +42% reading, balanced → performance +6%.

## 3. Mixed backends

ROCm reads prompts 30–59% faster at depth and writes only 6% slower. Running both (two instances of the
model, KV cache moved between them after reading) could win back those 6% of the writing time, at **twice
the model memory** (44 of 64 GB for the 35B) and a cache transfer every turn. Not built: ROCm alone takes
the gain. Revisit if Vulkan's flash attention gets fixed and the two trade places.

## 4. More

- **4a MoE offload** (`--n-cpu-moe N`): the experts of the first N layers stay in system memory.
  Exact sizes come from the GGUF tensor table (35B: 18.2 of 20.6 GiB are experts, 40 layers + MTP),
  both pools are budgeted. Cost on the 35B: 10 layers (4.5 GiB moved) −13% reading and writing,
  20 layers ~−20%. Use: MoE models larger than the GPU pool (~120B class at 4 bits).
- **4b Preload pinned models** at daemon start, one after the other.
- **4c Benchmarks**: the standard/full perf suite adds the three workloads (with draft acceptance) and a
  follow-up turn on an 8k conversation.
- **4d Parallel slots**: llama.cpp's default (4 slots, unified KV) already serves several clients; no setting.
- **4e NPU**: embeddings and Whisper on the NPU keep the GPU for the agent model (FastFlowLM, M2).
