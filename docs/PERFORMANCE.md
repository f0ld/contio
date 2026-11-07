# Performance roadmap

Goal: make contio faster step by step on this machine (Ryzen AI 9 HX 370, Radeon 890M, 64 GB VRAM
carve-out). Every item is **measured** before it ships and lands as a setting in the **Performance
(expert)** section, with a short explanation of what it does and its upsides and downsides. Defaults
change only when a setting is a clear win with no real downside.

Baseline: Qwen3.6 35B-A3B UD-Q4_K_M on Vulkan, `performance` power profile: **~23 tok/s** writing,
**~540 tok/s** reading a 4k prompt, **~460 tok/s** at 16k (m0-gpu.md §8).

| # | Item | Speeds up | Status |
|---|---|---|---|
| 1 | Speculative decoding | writing answers | **n-gram + MTP shipped** (v0.3.7); draft models for dense models open |
| 2 | Faster long prompts | reading prompts, time to first token | planned |
| 3 | Mixed backends (ROCm reads, Vulkan writes) | both, a little | planned (experiment) |
| 4 | More: power while serving, parallel slots, MoE offload, … | various | ideas |

## 1. Speculative decoding

A cheap guesser proposes the next few tokens; the model checks them all in one pass (as fast as
reading a prompt) and keeps the ones it agrees with. Output is unchanged, only faster when guesses
are right. Measured with a small harness (llama-server directly, `timings` from the response): three
workloads, 700 tokens each, thinking off, 2 runs each. *edit* returns a 140-line Rust file with a rename (what coding agents do all day),
*code* writes a new program, *prose* explains a topic.

| Method (35B-A3B, Vulkan) | edit | code | prose | Extra memory | Notes |
|---|---|---|---|---|---|
| none (baseline) | 22.8 tok/s | 22.7 | 22.7 | – | |
| **n-gram** (`ngram-simple`) | **106.7 (4.7×)** | 22.7 (±0) | 22.5 (±0) | ~0 | copies spans already in the context; idle otherwise. Prompt reading −6% |
| n-gram (`ngram-mod`) | 97–118 (4.7×) | 22.6 | 22.7 | ~16 MB | same idea, pool shared across requests; more variance |
| draft model Qwen3.5 0.8B, 3 tokens | 26.9 (+18%) | 20.4 (−10%) | 13.4 (−41%) | +0.8 GB | **loses**: checking drafts on a MoE wakes many experts |
| draft model, 8 tokens | 30.9 (+35%) | 19.8 (−13%) | 9.5 (−58%) | +0.8 GB | worse |
| MTP (`draft-mtp`, `unsloth/…-MTP-GGUF`) | 39.4 (1.7×) | 34.1 (+50%) | 24.9 (+7%) | +0.5 GB file | the model's own prediction layers; prompt reading −10% |
| **n-gram + MTP** | **111 (4.8×)** | **32.8 (+44%)** | **24.6 (+6%)** | +0.5 GB file | best overall; n-gram wins where it has a guess, MTP elsewhere |

Baselines: 22.8 / 22.7 / 22.7 tok/s for the plain file, 23.1 / 23.3 / 23.2 for the MTP file with speculation off
(the extra layers cost nothing when unused).

Findings:
- **n-gram is the big, free win for agentic coding**: edits are mostly copying, and copies are
  guessed almost perfectly (659 of 758 drafts accepted). Nothing to download, no cost when it can't help.
- **MTP helps everywhere a little and new code a lot** (+44–50%), but needs a model file that kept the
  MTP layers; most GGUF repos strip them, unsloth publishes `-MTP-GGUF` variants.
- **Draft models don't pay off for MoE models** here (checking drafts wakes many experts); they may for
  dense models (to measure: Qwen3.5 4B / Qwen3 8B with the 0.8B draft).
- Through the gateway (contio, v0.3.7): 126 tok/s on the edit task with n-gram + MTP.

Shipped (v0.3.7): per-model `speculative = { ngram, mtp }` (off by default), Expert → Performance in the
model editor, `contio set <model> --spec ngram,mtp|off`; MTP is offered only when the GGUF header has
`<arch>.nextn_predict_layers` > 0.

Next: catalog offers the `-MTP` variant where one exists; `contio-bench` gets the *edit* workload so the
perf suite shows the effect; draft models for dense models.

## 2. Faster long prompts

Agents resend a growing context every turn, so time to first token is dominated by prompt reading.

- **2a Prompt-cache reuse across turns.** llama-server keeps a slot's KV cache, so a follow-up turn only
  reads the new tokens. Qwen3.5/3.6 are hybrid models (recurrent layers) whose cache can only be reused
  from *context checkpoints* (`--ctx-checkpoints`, default 32). To measure: second-turn TTFT for a real
  agent conversation, checkpoint count/spacing, `--cache-reuse` (KV shifting when the middle changes).
  Likely the biggest single win for agents.
- **2b Micro-batch size** (`-ub`, default 512): larger batches use the GPU better on long prompts;
  costs compute-buffer memory. Sweep 512/1024/2048.
- **2c KV cache quantization** (`-ctk/-ctv q8_0`): half the KV memory → longer contexts fit; speed
  effect with flash attention on Vulkan to measure; tiny quality cost.
- **2d ROCm per model** (done in v0.3.6): +3–9% prompt reading at 4k–16k, −5–8% writing.
- **2e Power profile while serving**: hold `performance` while requests run (power saver → balanced
  was +42% prompt reading); more heat and fan noise, battery drain.

## 3. Mixed backends

ROCm reads the prompt, Vulkan writes the answer: two llama-server instances of the same model, KV cache
moved with slot save/restore between them. Estimate for 35B, 16k prompt + 500 tokens: ~54 s vs ~56–57 s
(3–5%), at **twice the model memory** and extra latency per turn. Experimental option at best; worth it
only if the ROCm/Vulkan gap grows. Prerequisite check: is a saved slot portable across backends?

## 4. More ideas

- **Parallel slots** (`-np`): several agents at once get served together (higher total throughput),
  each a bit slower; KV memory per slot.
- **MoE expert offload** (`--n-cpu-moe`): keep some experts in the system pool to run models larger
  than the 64 GB GPU pool (e.g. 120B MoE); slower writing.
- **Preload / keep warm**: start pinned models with the daemon; load time is 5–20 s otherwise.
- **NPU for side work**: embeddings, Whisper, small helpers on the NPU so the GPU stays free.
- **Batch size for embeddings/rerankers**, **CPU thread count** for CPU models.
- **Draft models for dense models** (see 1).
