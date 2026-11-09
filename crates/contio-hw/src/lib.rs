//! Hardware detection and live metrics from Linux sysfs/procfs.
//!
//! Everything is read relative to a configurable root (`/` in production) so the
//! probe can be tested against fixture directory trees. Nothing here needs root
//! privileges; values that need them (e.g. RAPL energy) are deliberately not used
//! (see `docs/spikes/m0-gpu.md`).

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use contio_core::hw::{
    CoreClass, CpuInfo, GpuInfo, HardwareInfo, MemoryPool, MemoryPoolKind, Metrics, NpuInfo,
    PowerInfo, PowerProfile,
};

const AMD_PCI_VENDOR: &str = "0x1002";

#[derive(Debug, thiserror::Error)]
pub enum HwError {
    #[error("reading {path}: {source}")]
    Io { path: PathBuf, source: io::Error },
    #[error("unexpected value in {path}: {value:?}")]
    Parse { path: PathBuf, value: String },
}

pub type Result<T> = std::result::Result<T, HwError>;

/// Watts from a hwmon power file (µW).
pub fn read_microwatts(path: &Path) -> Option<f64> {
    read_u64(path).ok().map(|uw| uw as f64 / 1e6)
}

/// Reads hardware state below `root`.
#[derive(Debug, Clone)]
pub struct HwProbe {
    root: PathBuf,
}

impl Default for HwProbe {
    fn default() -> Self {
        Self::new("/")
    }
}

impl HwProbe {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Full hardware description. Fails only if the CPU cannot be described;
    /// GPU, NPU and power details degrade to `None`/unknown.
    pub fn probe(&self) -> Result<HardwareInfo> {
        let gpu_dir = self.amd_gpu_dir();
        Ok(HardwareInfo {
            cpu: self.cpu()?,
            gpu: gpu_dir.as_deref().and_then(|d| self.gpu(d)),
            npu: self.npu(),
            pools: self.pools(gpu_dir.as_deref())?,
            power: self.power(),
        })
    }

    /// Cheap snapshot for the live dashboard (sampled ~1 Hz).
    pub fn sample(&self) -> Result<Metrics> {
        let gpu_dir = self.amd_gpu_dir();
        let split = gpu_dir.as_deref().and_then(apu_power_split);
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or_default();
        Ok(Metrics {
            timestamp_ms,
            soc_watts: gpu_dir.as_deref().and_then(|d| self.soc_watts(d)),
            gpu_watts: split.map(|(gpu, _)| gpu),
            cpu_watts: split.map(|(_, cpu)| cpu),
            gpu_busy_percent: gpu_dir
                .as_deref()
                .and_then(|d| read_u64(&d.join("gpu_busy_percent")).ok())
                .map(|v| v.min(100) as u8),
            pools: self.pools(gpu_dir.as_deref())?,
            power: self.power(),
        })
    }

    fn path(&self, rel: &str) -> PathBuf {
        self.root.join(rel.trim_start_matches('/'))
    }

    // --- CPU ----------------------------------------------------------------

    fn cpu(&self) -> Result<CpuInfo> {
        let cpuinfo_path = self.path("/proc/cpuinfo");
        let cpuinfo = read_string(&cpuinfo_path)?;
        let model = field(&cpuinfo, "model name").unwrap_or("unknown").to_owned();
        let avx512 = field(&cpuinfo, "flags").is_some_and(|f| f.split_whitespace().any(|x| x == "avx512f"));

        // (package, core) -> max MHz, plus thread count per class
        let mut cores: BTreeMap<(u64, u64), u32> = BTreeMap::new();
        let mut threads_per_mhz: BTreeMap<u32, u32> = BTreeMap::new();
        let mut threads = 0;
        for cpu_dir in numbered_entries(&self.path("/sys/devices/system/cpu"), "cpu") {
            let topo = cpu_dir.join("topology");
            let (Ok(core), Ok(pkg)) = (
                read_u64(&topo.join("core_id")),
                read_u64(&topo.join("physical_package_id")),
            ) else {
                continue; // offline CPU
            };
            let mhz = read_u64(&cpu_dir.join("cpufreq/cpuinfo_max_freq"))
                .map(|khz| (khz / 1000) as u32)
                .unwrap_or(0);
            threads += 1;
            *threads_per_mhz.entry(mhz).or_default() += 1;
            cores.insert((pkg, core), mhz);
        }
        if threads == 0 {
            // No sysfs topology (containers): fall back to /proc/cpuinfo.
            threads = cpuinfo.lines().filter(|l| l.starts_with("processor")).count() as u32;
            if threads == 0 {
                return Err(HwError::Parse { path: cpuinfo_path, value: "no processors".into() });
            }
        }

        let mut cores_per_mhz: BTreeMap<u32, u32> = BTreeMap::new();
        for mhz in cores.values() {
            *cores_per_mhz.entry(*mhz).or_default() += 1;
        }
        let core_classes = cores_per_mhz
            .iter()
            .rev()
            .map(|(&max_mhz, &n)| CoreClass {
                max_mhz,
                cores: n,
                threads: threads_per_mhz.get(&max_mhz).copied().unwrap_or(0),
            })
            .collect();

        Ok(CpuInfo {
            model,
            cores: if cores.is_empty() { threads } else { cores.len() as u32 },
            threads,
            core_classes,
            avx512,
        })
    }

    // --- GPU ----------------------------------------------------------------

    /// `/sys/class/drm/cardN/device` of the first AMD GPU with VRAM info.
    fn amd_gpu_dir(&self) -> Option<PathBuf> {
        numbered_entries(&self.path("/sys/class/drm"), "card")
            .into_iter()
            .map(|card| card.join("device"))
            .find(|dev| {
                read_string(&dev.join("vendor")).is_ok_and(|v| v.trim() == AMD_PCI_VENDOR)
                    && dev.join("mem_info_vram_total").exists()
            })
    }

    fn gpu(&self, dev: &Path) -> Option<GpuInfo> {
        let vendor = read_string(&dev.join("vendor")).ok()?;
        let device = read_string(&dev.join("device")).ok()?;
        Some(GpuInfo {
            pci_id: format!(
                "{}:{}",
                vendor.trim().trim_start_matches("0x"),
                device.trim().trim_start_matches("0x")
            ),
            driver: link_name(&dev.join("driver")).unwrap_or_else(|| "unknown".into()),
            vram_total_bytes: read_u64(&dev.join("mem_info_vram_total")).ok()?,
            gtt_total_bytes: read_u64(&dev.join("mem_info_gtt_total")).unwrap_or(0),
        })
    }

    /// amdgpu hwmon `PPT` = SoC package power, readable without root.
    fn soc_watts(&self, dev: &Path) -> Option<f64> {
        read_microwatts(&self.soc_power_file(dev)?)
    }

    /// The file holding SoC package power in µW, for fast repeated sampling
    /// (benchmarks read it at 10 Hz). Read it with [`read_microwatts`].
    pub fn soc_power_sensor(&self) -> Option<PathBuf> {
        self.soc_power_file(&self.amd_gpu_dir()?)
    }

    fn soc_power_file(&self, dev: &Path) -> Option<PathBuf> {
        let hwmons = numbered_entries(&dev.join("hwmon"), "hwmon");
        let pick = hwmons
            .iter()
            .find(|h| read_string(&h.join("power1_label")).is_ok_and(|l| l.trim() == "PPT"))
            .or(hwmons.first())?;
        ["power1_input", "power1_average"].iter().map(|f| pick.join(f)).find(|p| p.is_file())
    }

    // --- NPU ----------------------------------------------------------------

    fn npu(&self) -> Option<NpuInfo> {
        numbered_entries(&self.path("/sys/class/accel"), "accel")
            .into_iter()
            .find_map(|dir| {
                let name = dir.file_name()?.to_str()?.to_owned();
                let device_path = format!("/dev/accel/{name}");
                let accessible = fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(self.path(&device_path))
                    .is_ok();
                Some(NpuInfo {
                    device_path,
                    driver: link_name(&dir.join("device/driver")).unwrap_or_else(|| "unknown".into()),
                    accessible,
                })
            })
    }

    // --- memory & power -----------------------------------------------------

    fn pools(&self, gpu_dir: Option<&Path>) -> Result<Vec<MemoryPool>> {
        let mut pools = Vec::with_capacity(2);
        if let Some(dev) = gpu_dir
            && let Ok(total) = read_u64(&dev.join("mem_info_vram_total"))
        {
            pools.push(MemoryPool {
                kind: MemoryPoolKind::Gpu,
                total_bytes: total,
                used_bytes: read_u64(&dev.join("mem_info_vram_used")).unwrap_or(0),
            });
        }

        let meminfo_path = self.path("/proc/meminfo");
        let meminfo = read_string(&meminfo_path)?;
        let kb = |key: &str| {
            field(&meminfo, key)
                .and_then(|v| v.trim_end_matches("kB").trim().parse::<u64>().ok())
                .ok_or_else(|| HwError::Parse { path: meminfo_path.clone(), value: key.into() })
        };
        let total = kb("MemTotal")? * 1024;
        let available = kb("MemAvailable")? * 1024;
        pools.push(MemoryPool {
            kind: MemoryPoolKind::System,
            total_bytes: total,
            used_bytes: total.saturating_sub(available),
        });
        Ok(pools)
    }

    fn power(&self) -> PowerInfo {
        let profile = read_string(&self.path("/sys/firmware/acpi/platform_profile"))
            .map(|s| PowerProfile::from_platform_profile(&s))
            .unwrap_or_else(|_| PowerProfile::Unknown("unavailable".into()));

        let mut on_ac = None;
        let mut battery_percent = None;
        for supply in dir_entries(&self.path("/sys/class/power_supply")) {
            match read_string(&supply.join("type")).as_deref().map(str::trim) {
                Ok("Mains") => {
                    let online = read_u64(&supply.join("online")).is_ok_and(|v| v == 1);
                    on_ac = Some(on_ac.unwrap_or(false) || online);
                }
                Ok("Battery") if battery_percent.is_none() => {
                    battery_percent = read_u64(&supply.join("capacity")).ok().map(|v| v.min(100) as u8);
                }
                _ => {}
            }
        }
        PowerInfo { profile, on_ac, battery_percent }
    }
}

/// (graphics, all CPU cores) in watts from amdgpu's binary `gpu_metrics`
/// table. Only the APU layout v3.0 (Strix Point and later) is decoded; its
/// field offsets come from `struct gpu_metrics_v3_0` in the kernel's
/// `kgd_pp_interface.h`. Other layouts give `None` instead of guessing.
fn apu_power_split(dev: &Path) -> Option<(f64, f64)> {
    const GFX_POWER: usize = 124; // average_gfx_power, mW
    const CORE_POWER: usize = 132; // average_all_core_power, mW
    let b = fs::read(dev.join("gpu_metrics")).ok()?;
    if b.len() < CORE_POWER + 4 || (b[2], b[3]) != (3, 0) {
        return None;
    }
    let mw = |at: usize| u32::from_le_bytes(b[at..at + 4].try_into().unwrap());
    let (gfx, core) = (mw(GFX_POWER), mw(CORE_POWER));
    // the firmware fills fields it doesn't support with all ones
    (gfx != u32::MAX && core != u32::MAX).then(|| (gfx as f64 / 1e3, core as f64 / 1e3))
}

// --- helpers ------------------------------------------------------------------

fn read_string(path: &Path) -> Result<String> {
    fs::read_to_string(path).map_err(|source| HwError::Io { path: path.to_owned(), source })
}

fn read_u64(path: &Path) -> Result<u64> {
    let s = read_string(path)?;
    s.trim()
        .parse()
        .map_err(|_| HwError::Parse { path: path.to_owned(), value: s.trim().to_owned() })
}

/// Value of the first `key : value` line in procfs-style text.
fn field<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    text.lines().find_map(|line| {
        let (k, v) = line.split_once(':')?;
        (k.trim() == key).then(|| v.trim())
    })
}

/// File name of a symlink target, e.g. the driver behind `device/driver`.
fn link_name(path: &Path) -> Option<String> {
    fs::read_link(path).ok()?.file_name()?.to_str().map(str::to_owned)
}

fn dir_entries(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .collect();
    v.sort();
    v
}

/// Entries named `<prefix><digits>` (e.g. `card1`, not `card1-eDP-1`), in numeric order.
fn numbered_entries(dir: &Path, prefix: &str) -> Vec<PathBuf> {
    let mut v: Vec<(u64, PathBuf)> = dir_entries(dir)
        .into_iter()
        .filter_map(|p| {
            let n = p.file_name()?.to_str()?.strip_prefix(prefix)?.parse().ok()?;
            Some((n, p))
        })
        .collect();
    v.sort_by_key(|(n, _)| *n);
    v.into_iter().map(|(_, p)| p).collect()
}

#[cfg(test)]
mod tests;
