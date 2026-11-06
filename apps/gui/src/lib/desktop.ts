// Controls that only exist in the desktop app (Tauri): starting and stopping
// the daemon, and the background service. In a plain browser they're absent.

import { invoke } from "@tauri-apps/api/core";

export interface DaemonInfo {
  running: boolean;
  /** Who runs it: this app, the systemd background service, something else, or nobody. */
  owner: "app" | "service" | "external" | "none";
  /** This build ships contiod (the AppImage). */
  bundled: boolean;
  service_available: boolean;
  service_installed: boolean;
  service_version: string | null;
}

export const isDesktop = () => typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

export const daemonInfo = () => invoke<DaemonInfo>("daemon_info");
export const daemonStart = () => invoke<DaemonInfo>("daemon_start");
export const daemonStop = () => invoke<DaemonInfo>("daemon_stop");
export const serviceSet = (enabled: boolean) => invoke<DaemonInfo>("service_set", { enabled });
