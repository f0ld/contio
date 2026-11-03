//! llama.cpp installers: source build (Vulkan), prebuilt release, Podman image.

use std::path::{Path, PathBuf};

use serde_json::Value;
use tokio::process::Command;

use crate::download::{Remote, fetch};
use crate::jobs::JobCtx;
use crate::preflight::{self, LLAMACPP_SOURCE_PACKAGES, Preflight};
use crate::{BackendKind, InstalledBackend, Method, git_checkout, install_dir, now, run_logged};

/// Release tested in M0 (docs/spikes/m0-gpu.md).
pub const LLAMACPP_DEFAULT_VERSION: &str = "b12068";
const REPO: &str = "https://github.com/ggml-org/llama.cpp.git";
const RELEASES_API: &str = "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags";
const IMAGE: &str = "ghcr.io/ggml-org/llama.cpp";

/// What `method` needs that's missing. Blocking.
pub fn llamacpp_preflight(method: Method) -> Preflight {
    match method {
        Method::Source => preflight::check(LLAMACPP_SOURCE_PACKAGES, &[]),
        Method::Prebuilt => preflight::check(&[], &["tar"]),
        Method::Podman => preflight::check(&[], &["podman"]),
    }
}

pub fn image_ref(version: &str) -> String {
    if version == "latest" { format!("{IMAGE}:server-vulkan") } else { format!("{IMAGE}:server-vulkan-{version}") }
}

pub async fn install_llamacpp(
    http: &reqwest::Client,
    data_dir: &Path,
    method: Method,
    version: &str,
    ctx: &JobCtx,
) -> Result<InstalledBackend, String> {
    ctx.step("checking prerequisites");
    let pf = tokio::task::spawn_blocking(move || llamacpp_preflight(method)).await.map_err(|e| e.to_string())?;
    if !pf.ok() {
        return Err(pf.error());
    }
    let dir = install_dir(data_dir, BackendKind::LlamaCpp, version, method);
    tokio::fs::create_dir_all(&dir).await.map_err(|e| e.to_string())?;
    let mut installed = InstalledBackend {
        backend: BackendKind::LlamaCpp,
        method,
        version: version.to_owned(),
        dir: dir.clone(),
        bin_dir: None,
        image: None,
        prefix: None,
        installed_at: 0,
    };

    match method {
        Method::Source => installed.bin_dir = Some(build_from_source(version, &dir, ctx).await?),
        Method::Prebuilt => installed.bin_dir = Some(install_prebuilt(http, data_dir, version, &dir, ctx).await?),
        Method::Podman => {
            // Not every release gets an image (M1: b12068 had none, b12063 did).
            ctx.step("resolving image tag");
            let resolved = match resolve_image_version(http, version).await {
                Ok(v) => v,
                Err(e) => {
                    ctx.log(&format!("tag lookup failed ({e}); trying {version} as given"));
                    version.to_owned()
                }
            };
            if resolved != version {
                ctx.log(&format!("no image for {version}; using the closest older one: {resolved}"));
            }
            let image = image_ref(&resolved);
            ctx.step(&format!("pulling {image}"));
            run_logged(ctx, Command::new("podman").args(["pull", &image])).await?;
            installed.version = resolved;
            installed.image = Some(image);
        }
    }
    installed.installed_at = now();
    installed.write_marker()?;
    Ok(installed)
}

/// The newest `server-vulkan-bN` tag with N ≤ the requested build number.
async fn resolve_image_version(http: &reqwest::Client, version: &str) -> Result<String, String> {
    let Some(want) = build_number(version) else { return Ok(version.to_owned()) }; // "latest" etc.
    let token: Value = http
        .get("https://ghcr.io/token?scope=repository:ggml-org/llama.cpp:pull")
        .send()
        .await
        .map_err(|e| e.to_string())?
        .json()
        .await
        .map_err(|e| e.to_string())?;
    let token = token["token"].as_str().ok_or("no registry token")?;
    let mut url = Some("https://ghcr.io/v2/ggml-org/llama.cpp/tags/list?n=1000".to_owned());
    let mut best: Option<u64> = None;
    while let Some(u) = url.take() {
        let res = http.get(&u).bearer_auth(token).send().await.map_err(|e| e.to_string())?;
        url = next_link(res.headers().get("link").and_then(|v| v.to_str().ok()));
        let page: Value = res.json().await.map_err(|e| e.to_string())?;
        for t in page["tags"].as_array().into_iter().flatten().filter_map(Value::as_str) {
            if let Some(n) = t.strip_prefix("server-vulkan-").and_then(build_number)
                && n <= want
                && best.is_none_or(|b| n > b)
            {
                best = Some(n);
            }
        }
    }
    best.map(|n| format!("b{n}")).ok_or_else(|| format!("no Vulkan image at or below {version}"))
}

fn build_number(version: &str) -> Option<u64> {
    version.strip_prefix('b')?.parse().ok()
}

/// `</v2/...?last=x&n=1000>; rel="next"` → absolute URL.
fn next_link(link: Option<&str>) -> Option<String> {
    let l = link?;
    let start = l.find('<')? + 1;
    let end = l.find('>')?;
    l.contains(r#"rel="next""#).then(|| format!("https://ghcr.io{}", &l[start..end]))
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
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=OFF",
        ]),
    )
    .await?;
    ctx.step("building (a few minutes)");
    let jobs = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).to_string();
    run_logged(
        ctx,
        Command::new("cmake").arg("--build").arg(&build).args(["--target", "llama-server", "llama-bench", "-j", &jobs]),
    )
    .await?;
    let bin = build.join("bin");
    smoke_test(&bin, ctx).await?;
    Ok(bin)
}

async fn install_prebuilt(
    http: &reqwest::Client,
    data_dir: &Path,
    version: &str,
    dir: &Path,
    ctx: &JobCtx,
) -> Result<PathBuf, String> {
    ctx.step("looking up release");
    let asset_name = format!("llama-{version}-bin-ubuntu-vulkan-x64.tar.gz");
    let release: Value = http
        .get(format!("{RELEASES_API}/{version}"))
        .header("User-Agent", "contio")
        .header("Accept", "application/vnd.github+json")
        .send()
        .await
        .map_err(|e| format!("GitHub: {e}"))?
        .error_for_status()
        .map_err(|e| format!("release {version} not found: {e}"))?
        .json()
        .await
        .map_err(|e| e.to_string())?;
    let asset = release["assets"]
        .as_array()
        .and_then(|a| a.iter().find(|x| x["name"] == asset_name.as_str()))
        .ok_or_else(|| format!("release {version} has no {asset_name}"))?;
    let remote = Remote {
        url: asset["browser_download_url"].as_str().unwrap_or_default().to_owned(),
        sha256: asset["digest"].as_str().and_then(|d| d.strip_prefix("sha256:")).map(str::to_owned),
        size: asset["size"].as_u64(),
        token: None,
    };
    let archive = data_dir.join("cache").join(&asset_name);
    fetch(http, &remote, &archive, ctx).await?;

    ctx.step("extracting");
    let out = dir.join("dist");
    tokio::fs::create_dir_all(&out).await.map_err(|e| e.to_string())?;
    run_logged(ctx, Command::new("tar").arg("-xzf").arg(&archive).arg("-C").arg(&out)).await?;
    let server = find_file(&out, "llama-server").ok_or("archive contains no llama-server")?;
    let bin = server.parent().expect("file has a parent").to_owned();
    smoke_test(&bin, ctx).await?;
    Ok(bin)
}

/// Fails early if the binary can't run here (e.g. missing shared libraries
/// in an Ubuntu build on Fedora).
async fn smoke_test(bin_dir: &Path, ctx: &JobCtx) -> Result<(), String> {
    ctx.step("checking llama-server runs");
    let server = bin_dir.join("llama-server");
    run_logged(ctx, Command::new(&server).arg("--version").env("LD_LIBRARY_PATH", bin_dir))
        .await
        .map_err(|e| format!("{} doesn't run on this system: {e}", server.display()))
}

fn find_file(dir: &Path, name: &str) -> Option<PathBuf> {
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let p = entry.path();
        if p.is_dir() {
            if let Some(f) = find_file(&p, name) {
                return Some(f);
            }
        } else if p.file_name().is_some_and(|f| f == name) {
            return Some(p);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_refs() {
        assert_eq!(image_ref("b12068"), "ghcr.io/ggml-org/llama.cpp:server-vulkan-b12068");
        assert_eq!(image_ref("latest"), "ghcr.io/ggml-org/llama.cpp:server-vulkan");
    }

    #[test]
    fn tag_helpers() {
        assert_eq!(build_number("b12063"), Some(12063));
        assert_eq!(build_number("latest"), None);
        assert_eq!(
            next_link(Some(r#"</v2/ggml-org/llama.cpp/tags/list?last=x&n=1000>; rel="next""#)).as_deref(),
            Some("https://ghcr.io/v2/ggml-org/llama.cpp/tags/list?last=x&n=1000")
        );
        assert_eq!(next_link(None), None);
    }

    #[test]
    fn finds_nested_files() {
        let d = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(d.path().join("build/bin")).unwrap();
        std::fs::write(d.path().join("build/bin/llama-server"), "").unwrap();
        assert_eq!(find_file(d.path(), "llama-server").unwrap(), d.path().join("build/bin/llama-server"));
        assert!(find_file(d.path(), "nope").is_none());
    }
}
