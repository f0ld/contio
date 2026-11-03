//! Resumable, verified downloads (spec §6 step 3).
//!
//! Data goes to `<dest>.part` and is renamed only after the size and SHA-256
//! match. A `.part` file left by an interrupted run is resumed with an HTTP
//! Range request.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use reqwest::StatusCode;
use reqwest::header::{AUTHORIZATION, CONTENT_LENGTH, RANGE};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

use crate::jobs::JobCtx;

/// What to fetch and how to check it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Remote {
    pub url: String,
    /// Lowercase hex SHA-256, when the source publishes one.
    pub sha256: Option<String>,
    pub size: Option<u64>,
    /// Bearer token (gated Hugging Face repos).
    pub token: Option<String>,
}

pub fn hf_url(repo: &str, file: &str, revision: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/{revision}/{file}")
}

/// Look up size and SHA-256 of a Hugging Face file. LFS files report both via
/// `x-linked-size` / `x-linked-etag` on the (redirecting) resolve URL.
pub async fn hf_remote(
    http: &reqwest::Client,
    base: &str,
    repo: &str,
    file: &str,
    revision: &str,
    token: Option<String>,
) -> Result<Remote, String> {
    let url = format!("{base}/{repo}/resolve/{revision}/{file}");
    let mut req = http.head(&url);
    if let Some(t) = &token {
        req = req.header(AUTHORIZATION, format!("Bearer {t}"));
    }
    let res = req.send().await.map_err(|e| format!("contacting Hugging Face: {e}"))?;
    match res.status() {
        s if s.is_success() || s.is_redirection() => {}
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
            return Err(format!("{repo} is gated or private: accept its license on huggingface.co and set HF_TOKEN"));
        }
        StatusCode::NOT_FOUND => return Err(format!("{file} not found in {repo}@{revision}")),
        s => return Err(format!("Hugging Face answered HTTP {s} for {url}")),
    }
    let header = |name: &str| res.headers().get(name).and_then(|v| v.to_str().ok()).map(str::to_owned);
    let sha256 = header("x-linked-etag")
        .map(|e| e.trim_matches('"').to_ascii_lowercase())
        .filter(|e| e.len() == 64 && e.bytes().all(|b| b.is_ascii_hexdigit()));
    let size = header("x-linked-size")
        .or_else(|| header(CONTENT_LENGTH.as_str()))
        .and_then(|s| s.parse().ok());
    Ok(Remote { url, sha256, size, token })
}

fn part_path(dest: &Path) -> PathBuf {
    let mut name = dest.file_name().unwrap_or_default().to_os_string();
    name.push(".part");
    dest.with_file_name(name)
}

/// Download `remote` to `dest`, resuming a previous `.part` file.
pub async fn fetch(http: &reqwest::Client, remote: &Remote, dest: &Path, ctx: &JobCtx) -> Result<(), String> {
    if let Some(dir) = dest.parent() {
        tokio::fs::create_dir_all(dir).await.map_err(|e| format!("creating {}: {e}", dir.display()))?;
    }
    if dest.is_file() && verify(dest, remote, ctx).await.is_ok() {
        ctx.step("already downloaded");
        return Ok(());
    }

    let part = part_path(dest);
    let mut have = tokio::fs::metadata(&part).await.map(|m| m.len()).unwrap_or(0);
    if remote.size.is_some_and(|s| have > s) {
        have = 0; // stale/foreign file: start over
    }

    if remote.size != Some(have) {
        ctx.step(if have > 0 { "resuming download" } else { "downloading" });
        let mut req = http.get(&remote.url);
        if let Some(t) = &remote.token {
            req = req.header(AUTHORIZATION, format!("Bearer {t}"));
        }
        if have > 0 {
            req = req.header(RANGE, format!("bytes={have}-"));
        }
        let res = req.send().await.map_err(|e| format!("download failed: {e}"))?;
        let append = match res.status() {
            StatusCode::PARTIAL_CONTENT => true,
            s if s.is_success() => false, // server ignored the range
            s => return Err(format!("download failed: HTTP {s}")),
        };
        if !append {
            have = 0;
        }
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .append(append)
            .truncate(!append)
            .open(&part)
            .await
            .map_err(|e| format!("opening {}: {e}", part.display()))?;

        let mut stream = res.bytes_stream();
        let mut last_report = Instant::now();
        ctx.progress(have, remote.size);
        while let Some(chunk) = stream.next().await {
            if ctx.is_cancelled() {
                file.flush().await.ok();
                return Err("cancelled (partial download kept for resuming)".into());
            }
            let chunk = chunk.map_err(|e| format!("download interrupted: {e} (will resume on retry)"))?;
            file.write_all(&chunk).await.map_err(|e| format!("writing {}: {e}", part.display()))?;
            have += chunk.len() as u64;
            if last_report.elapsed() > Duration::from_millis(250) {
                ctx.progress(have, remote.size);
                last_report = Instant::now();
            }
        }
        file.flush().await.map_err(|e| e.to_string())?;
        ctx.progress(have, remote.size);
    }

    if let Err(e) = verify(&part, remote, ctx).await {
        let _ = tokio::fs::remove_file(&part).await;
        return Err(e);
    }
    tokio::fs::rename(&part, dest).await.map_err(|e| format!("moving into place: {e}"))?;
    Ok(())
}

async fn verify(path: &Path, remote: &Remote, ctx: &JobCtx) -> Result<(), String> {
    let len = tokio::fs::metadata(path).await.map_err(|e| e.to_string())?.len();
    if let Some(size) = remote.size
        && len != size
    {
        return Err(format!("size mismatch: got {len} bytes, expected {size}"));
    }
    let Some(expected) = &remote.sha256 else { return Ok(()) };
    ctx.step("verifying sha256");
    let actual = sha256_file(path.to_owned(), ctx.clone()).await?;
    if &actual != expected {
        return Err(format!("sha256 mismatch: got {actual}, expected {expected} (file deleted)"));
    }
    Ok(())
}

/// Hash a file on the blocking pool, reporting progress.
pub async fn sha256_file(path: PathBuf, ctx: JobCtx) -> Result<String, String> {
    tokio::task::spawn_blocking(move || {
        use std::io::Read;
        let mut f = std::fs::File::open(&path).map_err(|e| e.to_string())?;
        let total = f.metadata().map(|m| m.len()).ok();
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 4 << 20];
        let mut done = 0u64;
        loop {
            ctx.check_cancelled()?;
            let n = f.read(&mut buf).map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            done += n as u64;
            ctx.progress(done, total);
        }
        Ok(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use axum::Router;
    use axum::http::{HeaderMap, StatusCode as S};
    use axum::response::IntoResponse;
    use axum::routing::get;

    use super::*;
    use crate::jobs::{JobState, Jobs};

    const BODY: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";

    fn sha(data: &[u8]) -> String {
        Sha256::digest(data).iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Serves BODY with Range support, plus HF-style HEAD metadata.
    async fn server(range_requests: Arc<AtomicUsize>) -> String {
        let hits = range_requests.clone();
        let app = Router::new()
            .route(
                "/file",
                get(move |headers: HeaderMap| {
                    let hits = hits.clone();
                    async move {
                        if let Some(r) = headers.get("range").and_then(|v| v.to_str().ok()) {
                            hits.fetch_add(1, Ordering::SeqCst);
                            let from: usize = r.trim_start_matches("bytes=").trim_end_matches('-').parse().unwrap();
                            return (S::PARTIAL_CONTENT, BODY[from..].to_vec()).into_response();
                        }
                        BODY.to_vec().into_response()
                    }
                }),
            )
            .route(
                "/org/repo/resolve/main/m.gguf",
                axum::routing::head(|| async {
                    let mut h = HeaderMap::new();
                    h.insert("x-linked-etag", format!("\"{}\"", sha(BODY).to_uppercase()).parse().unwrap());
                    h.insert("x-linked-size", BODY.len().to_string().parse().unwrap());
                    (S::FOUND, h)
                }),
            )
            .route("/org/gated/resolve/main/m.gguf", axum::routing::head(|| async { S::UNAUTHORIZED }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}")
    }

    async fn run(jobs: &Jobs, remote: Remote, dest: PathBuf) -> crate::jobs::JobInfo {
        let id = jobs.spawn("download", "t", None, move |ctx| async move {
            fetch(&reqwest::Client::new(), &remote, &dest, &ctx).await.map(|_| serde_json::Value::Null)
        });
        loop {
            let j = jobs.get(id).unwrap();
            if j.state != JobState::Running {
                return j;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    #[tokio::test]
    async fn hf_metadata_from_head() {
        let base = server(Arc::default()).await;
        let http = reqwest::Client::builder().redirect(reqwest::redirect::Policy::none()).build().unwrap();
        let r = hf_remote(&http, &base, "org/repo", "m.gguf", "main", None).await.unwrap();
        assert_eq!(r.sha256.as_deref(), Some(sha(BODY).as_str()), "lowercased, quotes stripped");
        assert_eq!(r.size, Some(BODY.len() as u64));
        let err = hf_remote(&http, &base, "org/gated", "m.gguf", "main", None).await.unwrap_err();
        assert!(err.contains("HF_TOKEN"), "{err}");
    }

    #[tokio::test]
    async fn downloads_verifies_and_resumes() {
        let ranges = Arc::new(AtomicUsize::new(0));
        let base = server(ranges.clone()).await;
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("sub/m.gguf");
        let remote = Remote {
            url: format!("{base}/file"),
            sha256: Some(sha(BODY)),
            size: Some(BODY.len() as u64),
            token: None,
        };
        let jobs = Jobs::default();

        // simulate an interrupted earlier run
        std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
        std::fs::write(part_path(&dest), &BODY[..10]).unwrap();
        let j = run(&jobs, remote.clone(), dest.clone()).await;
        assert_eq!(j.state, JobState::Done, "{:?}", j.error);
        assert_eq!(std::fs::read(&dest).unwrap(), BODY);
        assert!(!part_path(&dest).exists());
        assert_eq!(ranges.load(Ordering::SeqCst), 1, "resumed with a Range request");

        // second run: nothing to do
        let j = run(&jobs, remote, dest.clone()).await;
        assert_eq!(j.step.as_deref(), Some("already downloaded"));
    }

    #[tokio::test]
    async fn checksum_mismatch_deletes_partial_file() {
        let base = server(Arc::default()).await;
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("m.gguf");
        let remote = Remote { url: format!("{base}/file"), sha256: Some("0".repeat(64)), size: None, token: None };
        let j = run(&Jobs::default(), remote, dest.clone()).await;
        assert_eq!(j.state, JobState::Failed);
        assert!(j.error.unwrap().contains("sha256 mismatch"));
        assert!(!dest.exists() && !part_path(&dest).exists());
    }
}
