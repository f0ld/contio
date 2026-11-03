// Daemon connection and live metrics, shared by all pages. Started once by
// the layout.

import { subscribeMetrics, type ConnectionState } from "./daemon";
import type { Metrics } from "./types";

class Live {
  conn = $state<ConnectionState>("connecting");
  metrics = $state<Metrics | null>(null);
  get online() {
    return this.conn === "open";
  }
}

export const live = new Live();

export function startLive(): () => void {
  return subscribeMetrics(
    (m) => (live.metrics = m),
    (s) => (live.conn = s),
  );
}
