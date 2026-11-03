// Thin client for the contiod management API (spec §7.4).

import type { HardwareInfo, Metrics, ModelStatus } from "./types";

export const DAEMON_URL: string = import.meta.env.VITE_CONTIOD_URL ?? "http://127.0.0.1:8080";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${DAEMON_URL}${path}`, init);
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.error?.message ?? `HTTP ${res.status}`);
  }
  return res.json();
}

export const fetchHardware = (signal?: AbortSignal) => request<HardwareInfo>("/api/hw", { signal });

export const fetchModels = () => request<ModelStatus[]>("/api/models");

export const loadModel = (id: string) =>
  request<ModelStatus>(`/api/models/${encodeURIComponent(id)}/load`, { method: "POST" });

export const unloadModel = (id: string) =>
  request<{ unloaded: boolean }>(`/api/models/${encodeURIComponent(id)}/unload`, { method: "POST" });

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

export function profileLabel(p: HardwareInfo["power"]["profile"]): string {
  return typeof p === "string" ? p : `unknown (${p.unknown})`;
}
