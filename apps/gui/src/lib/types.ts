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
export type ModelState =
  | { state: "stopped" }
  | { state: "loading" }
  | { state: "ready"; port: number }
  | { state: "failed"; error: string };

export type ModelStatus = {
  id: string;
  device: DeviceKind;
  pool: MemoryPoolKind;
  backend: string | null;
  pinned: boolean;
  estimated_bytes: number | null;
  in_flight: number;
} & ModelState;
