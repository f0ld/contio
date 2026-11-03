use std::os::unix::fs::symlink;
use std::path::Path;

use contio_core::hw::{DeviceKind, MemoryPoolKind, PowerProfile};

use super::*;

const GIB: u64 = 1024 * 1024 * 1024;

fn write(root: &Path, rel: &str, content: &str) {
    let p = root.join(rel);
    fs::create_dir_all(p.parent().unwrap()).unwrap();
    fs::write(p, content).unwrap();
}

/// Fixture modelled on the Framework 13 (Ryzen AI 9 HX 370) from the M0 spikes:
/// 4 Zen 5 cores @ 5157 MHz + 8 Zen 5c @ 3289 MHz, SMT2, 64 GiB VRAM carve-out.
fn framework13(root: &Path) {
    write(
        root,
        "proc/cpuinfo",
        "processor\t: 0\nmodel name\t: AMD Ryzen AI 9 HX 370 w/ Radeon 890M\nflags\t\t: fpu sse avx2 avx512f avx512bw\n\n",
    );
    write(root, "proc/meminfo", "MemTotal:       65607936 kB\nMemFree:  1 kB\nMemAvailable:   55607936 kB\n");
    for cpu in 0..24u64 {
        let core = cpu % 12;
        let khz = if core < 4 { 5_157_895 } else { 3_289_474 };
        let d = format!("sys/devices/system/cpu/cpu{cpu}");
        write(root, &format!("{d}/topology/core_id"), &format!("{core}\n"));
        write(root, &format!("{d}/topology/physical_package_id"), "0\n");
        write(root, &format!("{d}/cpufreq/cpuinfo_max_freq"), &format!("{khz}\n"));
    }
    // noise that must be ignored
    fs::create_dir_all(root.join("sys/devices/system/cpu/cpufreq")).unwrap();
    fs::create_dir_all(root.join("sys/class/drm/card1-eDP-1")).unwrap();

    let gpu = "sys/class/drm/card1/device";
    write(root, &format!("{gpu}/vendor"), "0x1002\n");
    write(root, &format!("{gpu}/device"), "0x150e\n");
    write(root, &format!("{gpu}/mem_info_vram_total"), &format!("{}\n", 64 * GIB));
    write(root, &format!("{gpu}/mem_info_vram_used"), &format!("{}\n", 3 * GIB));
    write(root, &format!("{gpu}/mem_info_gtt_total"), &format!("{}\n", 31 * GIB));
    write(root, &format!("{gpu}/gpu_busy_percent"), "42\n");
    write(root, &format!("{gpu}/hwmon/hwmon9/power1_label"), "PPT\n");
    write(root, &format!("{gpu}/hwmon/hwmon9/power1_input"), "15020000\n");
    symlink("../../../bus/pci/drivers/amdgpu", root.join(gpu).join("driver")).unwrap();

    write(root, "sys/class/accel/accel0/device/.keep", "");
    symlink("../../../../bus/pci/drivers/amdxdna", root.join("sys/class/accel/accel0/device/driver")).unwrap();
    write(root, "dev/accel/accel0", "");

    write(root, "sys/firmware/acpi/platform_profile", "balanced\n");
    write(root, "sys/class/power_supply/ACAD/type", "Mains\n");
    write(root, "sys/class/power_supply/ACAD/online", "1\n");
    write(root, "sys/class/power_supply/BAT1/type", "Battery\n");
    write(root, "sys/class/power_supply/BAT1/capacity", "87\n");
}

#[test]
fn probes_framework13() {
    let dir = tempfile::tempdir().unwrap();
    framework13(dir.path());
    let hw = HwProbe::new(dir.path()).probe().unwrap();

    assert_eq!(hw.cpu.model, "AMD Ryzen AI 9 HX 370 w/ Radeon 890M");
    assert_eq!((hw.cpu.cores, hw.cpu.threads), (12, 24));
    assert!(hw.cpu.avx512);
    assert_eq!(hw.cpu.core_classes.len(), 2);
    assert_eq!(
        (hw.cpu.core_classes[0].max_mhz, hw.cpu.core_classes[0].cores, hw.cpu.core_classes[0].threads),
        (5157, 4, 8)
    );
    assert_eq!((hw.cpu.core_classes[1].cores, hw.cpu.core_classes[1].threads), (8, 16));

    let gpu = hw.gpu.as_ref().unwrap();
    assert_eq!(gpu.pci_id, "1002:150e");
    assert_eq!(gpu.driver, "amdgpu");
    assert_eq!(gpu.vram_total_bytes, 64 * GIB);
    assert_eq!(gpu.gtt_total_bytes, 31 * GIB);

    let npu = hw.npu.as_ref().unwrap();
    assert_eq!(npu.device_path, "/dev/accel/accel0");
    assert_eq!(npu.driver, "amdxdna");
    assert!(npu.accessible);

    assert_eq!(hw.pools.len(), 2);
    assert_eq!(hw.pools[0].kind, MemoryPoolKind::Gpu);
    assert_eq!(hw.pools[0].used_bytes, 3 * GIB);
    assert_eq!(hw.pools[1].kind, MemoryPoolKind::System);
    assert_eq!(hw.pools[1].total_bytes, 65_607_936 * 1024);
    assert_eq!(hw.pools[1].used_bytes, 10_000_000 * 1024);

    assert_eq!(hw.power.profile, PowerProfile::Balanced);
    assert_eq!(hw.power.on_ac, Some(true));
    assert_eq!(hw.power.battery_percent, Some(87));
    assert_eq!(hw.devices(), vec![DeviceKind::Cpu, DeviceKind::Gpu, DeviceKind::Npu]);
}

#[test]
fn samples_power_and_gpu_busy() {
    let dir = tempfile::tempdir().unwrap();
    framework13(dir.path());
    let m = HwProbe::new(dir.path()).sample().unwrap();
    assert_eq!(m.soc_watts, Some(15.02));
    assert_eq!(m.gpu_busy_percent, Some(42));
    assert!(m.timestamp_ms > 0);
}

#[test]
fn degrades_without_gpu_npu_or_topology() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write(root, "proc/cpuinfo", "processor\t: 0\nmodel name\t: Generic\n\nprocessor\t: 1\n");
    write(root, "proc/meminfo", "MemTotal: 1024 kB\nMemAvailable: 512 kB\n");

    let hw = HwProbe::new(root).probe().unwrap();
    assert_eq!((hw.cpu.cores, hw.cpu.threads), (2, 2));
    assert!(hw.gpu.is_none() && hw.npu.is_none());
    assert_eq!(hw.pools.len(), 1);
    assert_eq!(hw.power.profile, PowerProfile::Unknown("unavailable".into()));
    assert_eq!(hw.power.on_ac, None);
    assert_eq!(hw.devices(), vec![DeviceKind::Cpu]);

    let m = HwProbe::new(root).sample().unwrap();
    assert_eq!(m.soc_watts, None);
}

#[test]
fn inaccessible_npu_is_not_a_device() {
    let dir = tempfile::tempdir().unwrap();
    framework13(dir.path());
    fs::remove_file(dir.path().join("dev/accel/accel0")).unwrap();
    let hw = HwProbe::new(dir.path()).probe().unwrap();
    assert!(!hw.npu.as_ref().unwrap().accessible);
    assert!(!hw.devices().contains(&DeviceKind::Npu));
}

#[test]
fn missing_meminfo_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "proc/cpuinfo", "processor\t: 0\n");
    assert!(matches!(HwProbe::new(dir.path()).probe(), Err(HwError::Io { .. })));
}
