//! FastFlowLM installer: source build only. FLM ships no Fedora packages and
//! NPU access from containers is untested (docs/spikes/m0-npu.md §6.4).

use std::path::Path;

use tokio::process::Command;

use crate::jobs::JobCtx;
use crate::preflight::{self, FLM_SOURCE_PACKAGES, Preflight};
use crate::{BackendKind, InstalledBackend, Method, git_checkout, install_dir, now, run_logged};

/// Commit tested in M0 (FLM 1.0.12).
pub const FLM_DEFAULT_VERSION: &str = "d4a4553c9b12c64f16d913423d68d0fddd3536b3";
const REPO: &str = "https://github.com/ROCm/FastFlowLM.git";
const XRT_DIR: &str = "/opt/xilinx/xrt";

/// What `method` needs that's missing. Blocking.
pub fn flm_preflight(method: Method) -> Preflight {
    let mut p = preflight::check(FLM_SOURCE_PACKAGES, &[]);
    preflight::check_xrt(Path::new(XRT_DIR), &mut p);
    if method != Method::Source {
        p.notes.push("FastFlowLM can only be installed from source on Fedora".into());
    }
    p
}

pub async fn install_flm(data_dir: &Path, method: Method, version: &str, ctx: &JobCtx) -> Result<InstalledBackend, String> {
    if method != Method::Source {
        return Err("FastFlowLM can only be installed from source on Fedora (no RPMs; NPU in containers untested)".into());
    }
    ctx.step("checking prerequisites");
    let pf = tokio::task::spawn_blocking(move || flm_preflight(method)).await.map_err(|e| e.to_string())?;
    if !pf.ok() || !pf.notes.is_empty() {
        return Err(pf.error());
    }

    let short = &version[..version.len().min(12)];
    let dir = install_dir(data_dir, BackendKind::Flm, short, method);
    let src = dir.join("src");
    let prefix = dir.join("prefix");
    git_checkout(ctx, REPO, version, &src).await?;
    ctx.step("fetching submodules");
    run_logged(
        ctx,
        Command::new("git").arg("-C").arg(&src).args(["submodule", "update", "-q", "--init", "--recursive", "--depth", "1"]),
    )
    .await?;

    // FLM's own user-prefix installer (no sudo). Its generated flm_env.sh is
    // never sourced: contio builds the runtime env itself (m0-npu.md §6.6).
    ctx.step("building (a few minutes)");
    run_logged(
        ctx,
        Command::new("./home_install.sh")
            .current_dir(src.join("src"))
            .env("FLM_PREFIX", &prefix)
            .env("XRT_DIR", XRT_DIR),
    )
    .await?;
    if !prefix.join("bin/flm").is_file() {
        return Err(format!("build finished but {} is missing", prefix.join("bin/flm").display()));
    }

    let installed = InstalledBackend {
        backend: BackendKind::Flm,
        method,
        version: version.to_owned(),
        dir,
        bin_dir: None,
        image: None,
        prefix: Some(prefix),
        installed_at: now(),
    };
    installed.write_marker()?;
    Ok(installed)
}
