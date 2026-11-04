// Mirrors crates/contio-core/src/hw.rs (serde JSON shape).

export type DeviceKind = "cpu" | "gpu" | "npu" | "hybrid";
export type MemoryPoolKind = "gpu" | "system";
export type PowerProfile = "low-power" | "balanced" | "performance" | { unknown: string };

export interface MemoryPool {
  kind: MemoryPoolKind;
  total_bytes: number;
  used_bytes: number;
}

export interface CoreClass {
  max_mhz: number;
  cores: number;
  threads: number;
}

export interface CpuInfo {
  model: string;
  cores: number;
  threads: number;
  core_classes: CoreClass[];
  avx512: boolean;
}

export interface GpuInfo {
  pci_id: string;
  driver: string;
  vram_total_bytes: number;
  gtt_total_bytes: number;
}

export interface NpuInfo {
  device_path: string;
  driver: string;
  accessible: boolean;
}

export interface PowerInfo {
  profile: PowerProfile;
  on_ac: boolean | null;
  battery_percent: number | null;
}

export interface HardwareInfo {
  cpu: CpuInfo;
  gpu: GpuInfo | null;
  npu: NpuInfo | null;
  pools: MemoryPool[];
  power: PowerInfo;
}

export interface Metrics {
  timestamp_ms: number;
  soc_watts: number | null;
  gpu_busy_percent: number | null;
  pools: MemoryPool[];
  power: PowerInfo;
}

// Mirrors crates/contio-core/src/model.rs (ModelStatus, state flattened).
export type ModelKind = "chat" | "embedding" | "reranker" | "transcription";

export interface ModelOrigin {
  catalog?: string;
  repo?: string;
  quant?: string;
}

export type ModelState =
  | { state: "stopped" }
  | { state: "loading" }
  | { state: "ready"; port: number }
  | { state: "failed"; error: string };

export type ModelSource =
  | { format: "gguf"; path: string; mmproj?: string }
  | { format: "flm"; tag: string }
  | { format: "whisper"; path: string };

export type ModelStatus = {
  id: string;
  device: DeviceKind;
  kind: ModelKind;
  context: number | null;
  /** Installed through contio (removable); false for config.toml models. */
  managed: boolean;
  origin: ModelOrigin | null;
  source: ModelSource;
  pool: MemoryPoolKind;
  backend: string | null;
  pinned: boolean;
  idle_ttl_secs: number | null;
  extra_args: string[];
  estimated_bytes: number | null;
  in_flight: number;
} & ModelState;

// Mirrors crates/contiod/src/aliases.rs.
export interface Alias {
  name: string;
  model: string;
  upstream_model: string | null;
  /** Added through contio; false for config.toml aliases (read-only). */
  managed: boolean;
}

export interface ModelPatch {
  context?: number;
  pinned?: boolean;
  /** 0 = daemon default. */
  idle_ttl_secs?: number;
  device?: "gpu" | "cpu";
  extra_args?: string[];
}

// Mirrors crates/contiod/src/catalog_api.rs.
export type Verdict = "fits" | "tight" | "too_big";

export interface Fit {
  verdict: Verdict;
  needed_bytes: number;
  budget_bytes: number;
}

export interface QuantView {
  label: string;
  size: number;
  files: number;
  bits_per_weight: number | null;
  tier: string | null;
  fit: { gpu: Fit | null; cpu: Fit | null };
  recommended: boolean;
  downloaded: boolean;
  installed: string[];
  measured: Measured[];
}

/** Latest benchmark results of an installed model. */
export interface Measured {
  model: string;
  tg_tps: number | null;
  pp512_tps: number | null;
  j_per_token: number | null;
  tools_pass_rate: number | null;
  agent_ready: boolean | null;
}

export interface NpuView {
  tag: string;
  name: string;
  footprint_bytes: number | null;
  default_context: number | null;
  capabilities: string[];
  single_turn: boolean;
  unsupported: boolean;
  fit: Fit | null;
  downloaded: boolean;
  installed: string[];
  measured: Measured[];
}

export interface RepoFile {
  path: string;
  size: number;
  sha256: string | null;
}

export interface ModelDetail {
  id: string | null;
  name: string;
  family: string;
  repo: string | null;
  kind: ModelKind;
  params: number | null;
  active_b: number | null;
  context_length: number | null;
  default_context: number;
  /** Context the quant fits were computed for. */
  fit_context: number;
  capabilities: string[];
  notes: string | null;
  recommended: string | null;
  quants: QuantView[];
  mmproj: RepoFile | null;
  npu: NpuView[];
  error: string | null;
}

export interface ModelSummary {
  id: string;
  name: string;
  family: string;
  repo: string | null;
  kind: ModelKind;
  params: number | null;
  active_b: number | null;
  capabilities: string[];
  default_context: number;
  quant_count: number;
  min_size: number | null;
  max_size: number | null;
  suggested: QuantView | null;
  npu: string[];
  installed: string[];
  error: string | null;
}

export interface Family {
  id: string;
  name: string;
  vendor: string;
  license: string;
  summary: string;
}

export interface CatalogOverview {
  families: Family[];
  models: ModelSummary[];
  npu_available: boolean;
}

export interface RepoInfo {
  repo: string;
  downloads: number;
  likes: number;
  pipeline: string | null;
  arch: string | null;
  params: number | null;
  context_length: number | null;
  tools: boolean;
  reasoning: boolean;
  gated: boolean;
}

export interface InstallModelRequest {
  model?: string;
  repo?: string;
  quant?: string;
  flm?: string;
  device?: DeviceKind;
  context?: number;
  id?: string;
  no_vision?: boolean;
  load?: boolean;
}

// Mirrors crates/contio-install/src/jobs.rs.
export type JobState = "running" | "done" | "failed" | "cancelled";

export interface Job {
  id: number;
  kind: string;
  title: string;
  state: JobState;
  step: string | null;
  progress: { done: number; total: number | null } | null;
  error: string | null;
  result: Record<string, unknown> | null;
  started_at: number;
  finished_at: number | null;
  log: string[];
  spec: unknown | null;
  retried_as: number | null;
}

// Backends and settings (install_api.rs, registry.rs).
export type InstallMethod = "source" | "prebuilt" | "podman";
export type BackendId = "llama.cpp" | "flm" | "whisper.cpp";

export interface InstalledBackend {
  backend: BackendId;
  method: InstallMethod;
  version: string;
  dir: string;
  bin_dir: string | null;
  image: string | null;
  prefix: string | null;
  installed_at: number;
}

export interface Settings {
  llamacpp_method: InstallMethod;
  flm_method: InstallMethod;
  whisper_method: InstallMethod;
}

export interface BackendsInfo {
  registered: string[];
  installed: InstalledBackend[];
  defaults: Record<BackendId, string>;
  settings: Settings;
}

export interface PreflightResult {
  ok: boolean;
  install_hint: string | null;
  preflight: { missing_packages: string[]; missing_tools: string[]; notes: string[] };
}

// Mirrors crates/contio-bench (store::Run) and contiod/src/bench_api.rs.
export type BenchSuite = "perf" | "tools" | "reasoning" | "knowledge" | "coding" | "retrieval";
export type BenchPreset = "quick" | "standard" | "full";

export interface BenchRun {
  id: number;
  suite: BenchSuite;
  model: string;
  preset: BenchPreset;
  started_at: number;
  finished_at: number | null;
  status: "running" | "done" | "failed" | "cancelled";
  error: string | null;
  target: {
    device?: DeviceKind;
    backend?: string;
    context?: number | null;
    origin?: ModelOrigin | null;
  };
  conditions: {
    power_profile?: string | null;
    on_ac?: boolean | null;
    battery_percent?: number | null;
    backend?: string | null;
    other_models?: string[];
  };
  /** Full report (detail endpoint only). */
  report: unknown;
  metrics: Record<string, number>;
  job_id: number | null;
}

export interface BenchRequest {
  models: string[];
  preset: BenchPreset;
  suites: BenchSuite[];
  exclusive: boolean;
  reload: boolean;
  /** Switch for the run, then restore. */
  power_profile?: "power-saver" | "balanced" | "performance";
}

export interface PerfSummary {
  prompt_tokens: number;
  ttft_s: number | null;
  prefill_tps: number | null;
  decode_tps: number | null;
  avg_watts: number | null;
  joules_per_token: number | null;
}

export interface PerfReportView {
  load: { load_s: number | null; estimated_bytes: number | null; peak_rss_bytes: number | null };
  perf: {
    idle_watts: number | null;
    prompts: { target_tokens: number; summary: PerfSummary; samples: unknown[] }[];
    skipped: number[];
    concurrency: { requests: number; aggregate_tps: number; per_request_decode_tps: number | null } | null;
  };
}

export interface QualityReportView {
  suite: string;
  pass_rate: number;
  reps: number;
  truncated: number;
  categories: Record<string, number>;
  items: { id: string; category?: string; runs: number; passed: number; failures: string[]; mean_s: number }[];
}

export interface RetrievalReportView {
  mode: "embedding" | "reranker";
  recall_at_1: number;
  mrr: number;
  seconds: number;
  queries: { query: string; expected: string; rank: number; top: string }[];
}

export interface ToolsReportView {
  pass_rate: number;
  agent_ready: boolean;
  categories: Record<string, number>;
  reps: number;
  cases: { id: string; category: string; runs: number; passed: number; failures: string[]; mean_s: number }[];
}
