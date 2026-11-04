// Thin client for the contiod management API (spec §7.4).

import type {
  Alias,
  ArenaRating,
  ArenaWinner,
  BackendId,
  BackendsInfo,
  BenchRequest,
  BenchRun,
  CatalogOverview,
  HardwareInfo,
  InstallMethod,
  InstallModelRequest,
  Job,
  Metrics,
  ModelDetail,
  ModelPatch,
  ModelStatus,
  PreflightResult,
  RepoInfo,
  Settings,
} from "./types";

export const DAEMON_URL: string = import.meta.env.VITE_CONTIOD_URL ?? "http://127.0.0.1:8080";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${DAEMON_URL}${path}`, init);
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.error?.message ?? `HTTP ${res.status}`);
  }
  return res.json();
}

async function send<T>(path: string, method: string, body?: unknown): Promise<T> {
  return request<T>(path, {
    method,
    headers: body === undefined ? undefined : { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

const enc = encodeURIComponent;

export const fetchHardware = (signal?: AbortSignal) => request<HardwareInfo>("/api/hw", { signal });

export const fetchModels = () => request<ModelStatus[]>("/api/models");

export const loadModel = (id: string) =>
  request<ModelStatus>(`/api/models/${encodeURIComponent(id)}/load`, { method: "POST" });

export const unloadModel = (id: string) =>
  request<{ unloaded: boolean }>(`/api/models/${encodeURIComponent(id)}/unload`, { method: "POST" });

export const removeModel = (id: string, deleteFiles: boolean) =>
  send<{ removed: string; deleted_files: string[] }>(`/api/models/${enc(id)}?delete_files=${deleteFiles}`, "DELETE");

export const updateModel = (id: string, patch: ModelPatch) => send<ModelStatus>(`/api/models/${enc(id)}`, "PATCH", patch);

export const fetchAliases = () => request<Alias[]>("/api/aliases");

export const setAlias = (name: string, model: string) => send<Alias>(`/api/aliases/${enc(name)}`, "PUT", { model });

export const deleteAlias = (name: string) => send<Alias>(`/api/aliases/${enc(name)}`, "DELETE");

// --- catalog ---

export const fetchCatalog = (refresh = false) => request<CatalogOverview>(`/api/catalog?refresh=${refresh}`);

export const fetchCatalogModel = (id: string, context?: number) =>
  request<ModelDetail>(`/api/catalog/models/${enc(id)}${context ? `?context=${context}` : ""}`);

export const fetchRepo = (repo: string, context?: number) =>
  request<ModelDetail>(`/api/catalog/repo?repo=${enc(repo)}${context ? `&context=${context}` : ""}`);

export const searchHf = (q: string) => request<RepoInfo[]>(`/api/catalog/search?q=${enc(q)}`);

export const installModel = (req: InstallModelRequest) => send<{ job_id: number }>("/api/catalog/install", "POST", req);

// --- jobs ---

export const fetchJobs = () => request<Job[]>("/api/jobs");

export const cancelJob = (id: number) => send<{ cancelled: boolean }>(`/api/jobs/${id}/cancel`, "POST");

export const retryJob = (id: number) => send<{ job_id: number }>(`/api/jobs/${id}/retry`, "POST");

// --- backends & settings ---

export const fetchBackends = () => request<BackendsInfo>("/api/backends");

export const checkPreflight = (backend: BackendId, method: InstallMethod) =>
  request<PreflightResult>(`/api/backends/preflight?backend=${enc(backend)}&method=${method}`);

export const installBackend = (backend: BackendId, method: InstallMethod) =>
  send<{ job_id: number }>("/api/backends/install", "POST", { backend, method });

export const saveSettings = (s: Settings) => send<Settings>("/api/settings", "PUT", s);

export type ConnectionState = "connecting" | "open" | "closed";

/**
 * Subscribe to live metrics. Reconnects with backoff (1s → 10s) until the
 * returned function is called.
 */
export function subscribeMetrics(
  onMetrics: (m: Metrics) => void,
  onState: (s: ConnectionState) => void,
): () => void {
  const url = DAEMON_URL.replace(/^http/, "ws") + "/api/events";
  let ws: WebSocket | null = null;
  let retry: ReturnType<typeof setTimeout> | undefined;
  let delay = 1000;
  let stopped = false;

  const connect = () => {
    onState("connecting");
    ws = new WebSocket(url);
    ws.onopen = () => {
      delay = 1000;
      onState("open");
    };
    ws.onmessage = (ev) => onMetrics(JSON.parse(ev.data));
    ws.onclose = () => {
      onState("closed");
      if (!stopped) {
        retry = setTimeout(connect, delay);
        delay = Math.min(delay * 2, 10_000);
      }
    };
  };
  connect();

  return () => {
    stopped = true;
    clearTimeout(retry);
    ws?.close();
  };
}

export function formatBytes(bytes: number): string {
  const gib = bytes / 1024 ** 3;
  return gib >= 1 ? `${gib.toFixed(1)} GiB` : `${(bytes / 1024 ** 2).toFixed(0)} MiB`;
}

export function formatParams(params: number | null, activeB: number | null = null): string {
  if (params == null) return "–";
  const b = params / 1e9;
  if (activeB != null) return `${b.toFixed(0)}B · ${activeB}B active`;
  return b < 1 ? `${(params / 1e6).toFixed(0)}M` : `${b < 10 ? b.toFixed(1) : b.toFixed(0)}B`;
}

export function formatContext(tokens: number | null): string {
  return tokens == null ? "–" : tokens >= 1024 ? `${Math.round(tokens / 1024)}k` : `${tokens}`;
}

export const errorText = (e: unknown) => (e instanceof Error ? e.message : String(e));

export function profileLabel(p: HardwareInfo["power"]["profile"]): string {
  return typeof p === "string" ? p : `unknown (${p.unknown})`;
}

// --- benchmarks ---

export const fetchBenchRuns = (model?: string) =>
  request<BenchRun[]>(`/api/bench/runs?limit=500${model ? `&model=${enc(model)}` : ""}`);

export const fetchBenchRun = (id: number) => request<BenchRun>(`/api/bench/runs/${id}`);

export const startBench = (req: BenchRequest) => send<{ job_id: number }>("/api/bench", "POST", req);

export const deleteBenchRun = (id: number) => send<{ deleted: number }>(`/api/bench/runs/${id}`, "DELETE");

export const benchExportUrl = (format: "csv" | "json") => `${DAEMON_URL}/api/bench/export?format=${format}`;

export const fetchArena = () =>
  request<{ ratings: ArenaRating[]; votes: number; judge_ratings: ArenaRating[]; judge_votes: number }>("/api/arena");

export const askJudge = (r: { prompt: string; model_a: string; model_b: string; answer_a: string; answer_b: string; judge: string }) =>
  send<{ winner: ArenaWinner; reasons: string[]; consistent: boolean }>("/api/arena/judge", "POST", r);

export const castVote = (vote: { prompt: string; model_a: string; model_b: string; winner: ArenaWinner }) =>
  send<{ id: number }>("/api/arena/votes", "POST", vote);
