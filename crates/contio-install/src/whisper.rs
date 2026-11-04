//! whisper.cpp installer: source build (Vulkan). Upstream publishes no Linux
//! release binaries, and its container images are tagged by commit only.

use std::path::{Path, PathBuf};

use tokio::process::Command;

use crate::jobs::JobCtx;
use crate::preflight::{self, LLAMACPP_SOURCE_PACKAGES, Preflight};
use crate::{BackendKind, InstalledBackend, Method, git_checkout, install_dir, now, run_logged};

/// Release tested in M2.
pub const WHISPER_DEFAULT_VERSION: &str = "v1.9.5";
const REPO: &str = "https://github.com/ggml-org/whisper.cpp.git";

/// What `method` needs that's missing. Blocking. Same toolchain as llama.cpp (both are ggml).
pub fn whisper_preflight(method: Method) -> Preflight {
    let mut p = preflight::check(LLAMACPP_SOURCE_PACKAGES, &[]);
    if method != Method::Source {
        p.notes.push("whisper.cpp can only be installed from source (upstream has no Linux binaries)".into());
    }
    p
}

pub async fn install_whisper(
    data_dir: &Path,
    method: Method,
    version: &str,
    ctx: &JobCtx,
) -> Result<InstalledBackend, String> {
    if method != Method::Source {
        return Err("whisper.cpp can only be installed from source (upstream has no Linux binaries)".into());
    }
    ctx.step("checking prerequisites");
    let pf = tokio::task::spawn_blocking(move || whisper_preflight(method)).await.map_err(|e| e.to_string())?;
    if !pf.ok() || !pf.notes.is_empty() {
        return Err(pf.error());
    }
    let dir = install_dir(data_dir, BackendKind::Whisper, version, method);
    tokio::fs::create_dir_all(&dir).await.map_err(|e| e.to_string())?;
    let bin_dir = build_from_source(version, &dir, ctx).await?;
    let installed = InstalledBackend {
        backend: BackendKind::Whisper,
        method,
        version: version.to_owned(),
        dir,
        bin_dir: Some(bin_dir),
        image: None,
        prefix: None,
        installed_at: now(),
    };
    installed.write_marker()?;
    Ok(installed)
}

async fn build_from_source(version: &str, dir: &Path, ctx: &JobCtx) -> Result<PathBuf, String> {
    let src = dir.join("src");
    let build = dir.join("build");
    git_checkout(ctx, REPO, version, &src).await?;
    ctx.step("configuring (Vulkan, native CPU)");
    run_logged(
        ctx,
        Command::new("cmake").arg("-S").arg(&src).arg("-B").arg(&build).args([
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_NATIVE=ON",
            "-DGGML_VULKAN=ON",
            "-DBUILD_SHARED_LIBS=OFF",
            "-DWHISPER_BUILD_TESTS=OFF",
            "-DWHISPER_BUILD_SERVER=ON",
        ]),
    )
    .await?;
    ctx.step("building (a few minutes)");
    let jobs = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).to_string();
    run_logged(ctx, Command::new("cmake").arg("--build").arg(&build).args(["--target", "whisper-server", "-j", &jobs]))
        .await?;
    let bin = build.join("bin");
    ctx.step("checking whisper-server runs");
    let server = bin.join("whisper-server");
    run_logged(ctx, Command::new(&server).arg("--help"))
        .await
        .map_err(|e| format!("{} doesn't run: {e}", server.display()))?;
    Ok(bin)
}
