//! Hardware description and live metrics (spec §2, §4.3, §8.1).

use serde::{Deserialize, Serialize};

/// Where a model runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceKind {
    Cpu,
    Gpu,
    Npu,
    /// NPU prefill + GPU decode. Reserved for post-v1 (spec §5.3).
    Hybrid,
}

/// Unified memory is split into pools that are budgeted separately (spec §7.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryPoolKind {
    /// BIOS VRAM carve-out used by the iGPU.
    Gpu,
    /// System RAM used by CPU and NPU (and GTT for the iGPU).
    System,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryPool {
    pub kind: MemoryPoolKind,
    pub total_bytes: u64,
    pub used_bytes: u64,
}

/// Cores that share a maximum frequency, e.g. Zen 5 vs Zen 5c.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoreClass {
    pub max_mhz: u32,
    pub cores: u32,
    pub threads: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuInfo {
    pub model: String,
    pub cores: u32,
    pub threads: u32,
    /// Fastest class first.
    pub core_classes: Vec<CoreClass>,
    pub avx512: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuInfo {
    /// PCI vendor:device, e.g. `1002:150e`.
    pub pci_id: String,
    pub driver: String,
    pub vram_total_bytes: u64,
    pub gtt_total_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NpuInfo {
    /// e.g. `/dev/accel/accel0`.
    pub device_path: String,
    pub driver: String,
    /// Whether the current user can open the device node.
    pub accessible: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PowerProfile {
    LowPower,
    Balanced,
    Performance,
    Unknown(String),
}

impl PowerProfile {
    /// Parse a value of `/sys/firmware/acpi/platform_profile`.
    pub fn from_platform_profile(s: &str) -> Self {
        match s.trim() {
            "low-power" | "quiet" | "cool" => Self::LowPower,
            "balanced" => Self::Balanced,
            "performance" => Self::Performance,
            other => Self::Unknown(other.to_owned()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PowerInfo {
    pub profile: PowerProfile,
    pub on_ac: Option<bool>,
    pub battery_percent: Option<u8>,
}

/// Static-ish description of the machine; probed once at startup.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub cpu: CpuInfo,
    pub gpu: Option<GpuInfo>,
    pub npu: Option<NpuInfo>,
    pub pools: Vec<MemoryPool>,
    pub power: PowerInfo,
}

impl HardwareInfo {
    pub fn devices(&self) -> Vec<DeviceKind> {
        let mut d = vec![DeviceKind::Cpu];
        if self.gpu.is_some() {
            d.push(DeviceKind::Gpu);
        }
        if self.npu.as_ref().is_some_and(|n| n.accessible) {
            d.push(DeviceKind::Npu);
        }
        d
    }
}

/// Frequently sampled values for the dashboard and benchmark runner.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Metrics {
    /// Unix time in milliseconds.
    pub timestamp_ms: u64,
    /// Whole-SoC package power (amdgpu `PPT`), CPU+GPU+NPU.
    pub soc_watts: Option<f64>,
    pub gpu_busy_percent: Option<u8>,
    pub pools: Vec<MemoryPool>,
    pub power: PowerInfo,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_platform_profiles() {
        assert_eq!(PowerProfile::from_platform_profile("balanced\n"), PowerProfile::Balanced);
        assert_eq!(PowerProfile::from_platform_profile("low-power"), PowerProfile::LowPower);
        assert_eq!(
            PowerProfile::from_platform_profile("balanced-performance"),
            PowerProfile::Unknown("balanced-performance".into())
        );
    }
}
