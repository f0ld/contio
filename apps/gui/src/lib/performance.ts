/**
 * Expert performance settings: what each one does and what it costs, shown
 * next to the control. Numbers are from docs/PERFORMANCE.md (Qwen3.6 35B-A3B,
 * Radeon 890M); keep them in sync when the roadmap is re-measured.
 */
export interface PerfSetting {
  key: "ngram" | "mtp" | "gpu_api";
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
    downsides: ["Reading the prompt gets ~6% slower", "No gain for new code or prose"],
  },
  mtp: {
    key: "mtp",
    title: "Speculative decoding: MTP",
    what:
      "Uses the model's own multi-token-prediction layers to guess the next tokens. Needs a model file that keeps these layers (usually a “…-MTP-GGUF” repo).",
    upsides: ["~40% faster writing new code, ~70% on edits, a little on prose", "Combines with n-gram (best: both on)"],
    downsides: [
      "Reading the prompt gets ~10% slower",
      "Only for model files with MTP layers (+0.5 GB for the 35B)",
    ],
  },
  gpu_api: {
    key: "gpu_api",
    title: "GPU backend",
    what: "Which llama.cpp build runs this model on the GPU.",
    upsides: ["ROCm: reads long prompts 3–9% faster", "Vulkan: writes answers 5–8% faster (the default)"],
    downsides: ["ROCm needs its own install (Backends)", "Switching restarts the model"],
  },
};
