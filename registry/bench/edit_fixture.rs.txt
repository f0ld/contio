//! Build preflight: report missing packages/tools *before* starting a build,
//! as one `dnf install` line (spec §5.2, m0-gpu.md). Nothing is installed
//! automatically: root actions stay with the user (spec §12, decision 9).

use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Serialize;

/// llama.cpp Vulkan build (verified in M0; `spirv-headers-devel` was the one missing).
pub const LLAMACPP_SOURCE_PACKAGES: &[&str] = &[
    "gcc-c++",
    "cmake",
    "ninja-build",
    "git",
    "vulkan-headers",
    "vulkan-loader-devel",
    "glslc",
    "spirv-headers-devel",
];

/// llama.cpp ROCm/HIP build (verified on Fedora 44 with ROCm 7.1, m0-gpu.md §8).
pub const LLAMACPP_ROCM_SOURCE_PACKAGES: &[&str] =
    &["gcc-c++", "cmake", "ninja-build", "git", "rocm-hip-devel", "hipblas-devel", "rocblas-devel"];

/// FastFlowLM source build (verified in M0).
pub const FLM_SOURCE_PACKAGES: &[&str] =
    &["gcc-c++", "cmake", "ninja-build", "git", "boost-devel", "libcurl-devel", "libedit-devel"];

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Preflight {
    pub missing_packages: Vec<String>,
    pub missing_tools: Vec<String>,
    /// Other blockers or caveats, in plain words.
    pub notes: Vec<String>,
}

impl Preflight {
    pub fn ok(&self) -> bool {
        self.missing_packages.is_empty() && self.missing_tools.is_empty()
    }

    /// The command the user should run, if anything is missing.
    pub fn install_hint(&self) -> Option<String> {
        (!self.missing_packages.is_empty()).then(|| format!("sudo dnf install {}", self.missing_packages.join(" ")))
    }

    /// A one-paragraph error for a job that can't start.
    pub fn error(&self) -> String {
        let mut parts = Vec::new();
        if let Some(h) = self.install_hint() {
            parts.push(format!("missing packages: {} (run: {h})", self.missing_packages.join(", ")));
        }
        if !self.missing_tools.is_empty() {
            parts.push(format!("missing tools: {}", self.missing_tools.join(", ")));
        }
        parts.extend(self.notes.iter().cloned());
        parts.join("; ")
    }
}

/// Check RPM packages and tools on `$PATH`. Blocking (runs `rpm`).
pub fn check(packages: &[&str], tools: &[&str]) -> Preflight {
    let mut p = Preflight::default();
    if find_in_path("rpm").is_some() {
        let installed = |pkg: &str| Command::new("rpm").args(["-q", "--quiet", pkg]).status().is_ok_and(|s| s.success());
        p.missing_packages = packages.iter().filter(|pkg| !installed(pkg)).map(|s| (*s).to_owned()).collect();
    } else if !packages.is_empty() {
        p.notes.push(format!("can't check packages without rpm; make sure these are installed: {}", packages.join(", ")));
    }
    p.missing_tools = tools.iter().filter(|t| find_in_path(t).is_none()).map(|s| (*s).to_owned()).collect();
    p
}

/// FLM needs the XRT runtime with the NPU plugin (Fedora: xrt RPMs in /opt/xilinx/xrt).
pub fn check_xrt(xrt_dir: &Path, p: &mut Preflight) {
    if !xrt_dir.join("version.json").is_file() {
        p.notes.push(format!(
            "XRT not found in {} (install the xrt and xrt_plugin-amdxdna packages)",
            xrt_dir.display()
        ));
    }
}

pub fn find_in_path(name: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|path| std::env::split_paths(&path).map(|d| d.join(name)).find(|p| p.is_file()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hint_and_error() {
        let p = Preflight {
            missing_packages: vec!["glslc".into(), "spirv-headers-devel".into()],
            missing_tools: vec![],
            notes: vec![],
        };
        assert!(!p.ok());
        assert_eq!(p.install_hint().unwrap(), "sudo dnf install glslc spirv-headers-devel");
        assert!(p.error().contains("run: sudo dnf install"));
        assert!(Preflight::default().ok());
    }

    #[test]
    fn finds_tools_on_path() {
        let p = check(&[], &["sh", "definitely-not-a-tool-xyz"]);
        assert_eq!(p.missing_tools, ["definitely-not-a-tool-xyz"]);
    }

    #[test]
    fn missing_xrt_is_a_note() {
        let mut p = Preflight::default();
        check_xrt(Path::new("/nonexistent/xrt"), &mut p);
        assert!(p.notes[0].contains("XRT not found"));
    }
}
