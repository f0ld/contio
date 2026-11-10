/**
 * Expert performance settings: what each one does and what it costs, shown
 * next to the control. Numbers are from docs/PERFORMANCE.md (Qwen3.6 35B-A3B,
 * Radeon 890M); keep them in sync when the roadmap is re-measured.
 */
export interface PerfSetting {
  key: "ngram" | "mtp" | "draft" | "gpu_api" | "serve_power" | "kv_cache" | "cpu_moe";
  title: string;
  /** One or two sentences, plain words. */
  what: string;
  upsides: string[];
  downsides: string[];
}

export const PERF_SETTINGS: Record<PerfSetting["key"], PerfSetting> = {
  ngram: {
    key: "ngram",
    title: "Speculative decoding: n-gram",
    what:
      "Guesses the next tokens by finding the same text earlier in the conversation; the model checks the guesses in one pass. Answers stay identical.",
    upsides: [
      "Up to ~4.7× faster writing when the answer repeats the input (editing code, rewriting a file)",
      "No extra download or memory; does nothing when it has no guess",
    ],
    downsides: ["No gain for new code or prose (no measurable cost either: prompt reading unchanged at 30k tokens)"],
  },
  mtp: {
    key: "mtp",
    title: "Speculative decoding: MTP",
    what:
      "Uses the model's own multi-token-prediction layers to guess the next tokens. Needs a model file that keeps these layers (usually a “…-MTP-GGUF” repo).",
    upsides: ["~40% faster writing new code, ~70% on edits, a little on prose", "Combines with n-gram (best: both on)"],
    downsides: [
      "Reading prompts gets ~6% slower (30k-token prompt)",
      "Only for model files with MTP layers (+0.5 GB for the 35B)",
    ],
  },
  draft: {
    key: "draft",
    title: "Speculative decoding: draft model",
    what:
      "A much smaller model with the same tokenizer guesses ahead; this model checks the guesses. Pays off for dense models, not for MoE models.",
    upsides: ["Dense Qwen3.5 4B with the 0.8B draft: +46% on edits and new code, +10% prose"],
    downsides: [
      "Can't be combined with n-gram (llama.cpp hangs on some prompts with both)",
      "MoE models (e.g. 35B-A3B) get slower: −10% code, −40% prose",
      "Prompt reading ~20% slower; the draft's memory (~1 GB) counts toward the model",
    ],
  },
  serve_power: {
    key: "serve_power",
    title: "Power while serving",
    what:
      "Switches the laptop to the performance power profile while a model works and back to your profile about 10 s after the last request.",
    upsides: [
      "Faster prompt reading: power saver → balanced alone was +42% for the 35B",
      "Your own profile stays active the rest of the time",
    ],
    downsides: ["More heat and fan noise while serving", "Drains the battery faster (use “only on AC”)"],
  },
  kv_cache: {
    key: "kv_cache",
    title: "KV cache precision",
    what:
      "The model's memory of the conversation, stored at lower precision. Halves (q8_0) or quarters (q4_0) its size, so longer contexts fit.",
    upsides: [
      "q8_0: about half the context memory, practically no quality loss",
      "Vulkan + q8_0: writing +12% deep in a long conversation (64k)",
    ],
    downsides: [
      "ROCm + q8_0: writing −34% deep in a long conversation; keep f16 there",
      "Prompt reading a few % slower; q4_0 costs noticeable quality on long contexts",
    ],
  },
  cpu_moe: {
    key: "cpu_moe",
    title: "MoE offload",
    what:
      "Keeps the expert weights of the first N layers in system memory instead of the GPU's. Lets MoE models larger than the 64 GB GPU pool run.",
    upsides: ["Frees GPU memory: bigger models or longer contexts fit", "Both pools are budgeted, nothing is overcommitted"],
    downsides: [
      "The CPU computes the offloaded experts: 35B-A3B with 10 of 40 layers −13% reading and writing, 20 layers ~−20%",
      "Uses system memory that CPU models and the NPU share",
    ],
  },
  gpu_api: {
    key: "gpu_api",
    title: "GPU backend",
    what:
      "Which llama.cpp build runs this model on the GPU. Coding agents (OpenCode, …) resend long conversations, so prompt reading deep in the context usually decides how fast they feel.",
    upsides: [
      "ROCm: reads prompts +30% faster at 64k context, +59% at 96k (Vulkan's attention kernel is slow at depth)",
      "Vulkan: writes answers 6–8% faster (the default)",
    ],
    downsides: ["ROCm needs its own install (Backends)", "Switching restarts the model"],
  },
};
