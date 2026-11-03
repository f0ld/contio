//! Hugging Face Hub: repo file listings (cached on disk) and GGUF search.

use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::quant::{RepoFile, RepoListing, parse_tree};

pub const HF_BASE: &str = "https://huggingface.co";
/// Listings change rarely; a day keeps browsing fast and offline-friendly.
pub const LISTING_TTL: Duration = Duration::from_secs(24 * 3600);

#[derive(Clone)]
pub struct HfClient {
    http: reqwest::Client,
    base: String,
    cache_dir: PathBuf,
    token: Option<String>,
}

/// A GGUF repo from the Hub search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepoInfo {
    pub repo: String,
    pub downloads: u64,
    pub likes: u64,
    pub pipeline: Option<String>,
    pub arch: Option<String>,
    pub params: Option<u64>,
    pub context_length: Option<u32>,
    pub tools: bool,
    pub reasoning: bool,
    pub gated: bool,
}

#[derive(Serialize, Deserialize)]
struct CachedListing {
    fetched_at: u64,
    files: Vec<RepoFile>,
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or_default()
}

/// `owner/name` with characters that are safe in URLs and file names.
pub fn valid_repo(repo: &str) -> bool {
    let parts: Vec<&str> = repo.split('/').collect();
    parts.len() == 2
        && parts.iter().all(|p| {
            !p.is_empty() && !p.starts_with('.') && p.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c))
        })
}

impl HfClient {
    pub fn new(http: reqwest::Client, base: &str, cache_dir: PathBuf) -> Self {
        let token = std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty());
        Self { http, base: base.trim_end_matches('/').to_owned(), cache_dir, token }
    }

    fn get(&self, url: &str) -> reqwest::RequestBuilder {
        let req = self.http.get(url).timeout(Duration::from_secs(20));
        match &self.token {
            Some(t) => req.bearer_auth(t),
            None => req,
        }
    }

    fn cache_path(&self, repo: &str) -> PathBuf {
        self.cache_dir.join(format!("{}.json", repo.replace('/', "__")))
    }

    /// Quants and projectors of `repo`, from the cache if younger than `ttl`.
    /// A stale cache is still used when the Hub can't be reached.
    pub async fn listing(&self, repo: &str, ttl: Duration) -> Result<RepoListing, String> {
        Ok(parse_tree(&self.files(repo, ttl).await?))
    }

    pub async fn files(&self, repo: &str, ttl: Duration) -> Result<Vec<RepoFile>, String> {
        if !valid_repo(repo) {
            return Err(format!("invalid repo `{repo}`"));
        }
        let path = self.cache_path(repo);
        let cached: Option<CachedListing> =
            tokio::fs::read(&path).await.ok().and_then(|b| serde_json::from_slice(&b).ok());
        if let Some(c) = &cached
            && now().saturating_sub(c.fetched_at) < ttl.as_secs()
        {
            return Ok(c.files.clone());
        }
        match self.fetch_files(repo).await {
            Ok(files) => {
                let entry = CachedListing { fetched_at: now(), files };
                if let Err(e) = write_atomic(&path, &serde_json::to_vec(&entry).unwrap_or_default()).await {
                    tracing::warn!("caching listing of {repo}: {e}");
                }
                Ok(entry.files)
            }
            Err(e) => match cached {
                Some(c) => {
                    tracing::warn!("{repo}: {e}; using cached listing");
                    Ok(c.files)
                }
                None => Err(e),
            },
        }
    }

    async fn fetch_files(&self, repo: &str) -> Result<Vec<RepoFile>, String> {
        let url = format!("{}/api/models/{repo}/tree/main?recursive=true", self.base);
        let res = self.get(&url).send().await.map_err(|e| format!("contacting Hugging Face: {e}"))?;
        match res.status().as_u16() {
            200 => {}
            401 | 403 => return Err(format!("{repo} is gated or private: set HF_TOKEN")),
            404 => return Err(format!("{repo} not found on Hugging Face")),
            s => return Err(format!("Hugging Face answered HTTP {s} for {repo}")),
        }
        let entries: Vec<Value> = res.json().await.map_err(|e| format!("listing {repo}: {e}"))?;
        Ok(entries
            .iter()
            .filter(|e| e["type"] == "file")
            .filter_map(|e| {
                Some(RepoFile {
                    path: e["path"].as_str()?.to_owned(),
                    size: e["size"].as_u64()?,
                    sha256: e["lfs"]["oid"].as_str().map(str::to_owned),
                })
            })
            .collect())
    }

    /// GGUF repos matching `query`, most downloaded first.
    pub async fn search(&self, query: &str, limit: u32) -> Result<Vec<RepoInfo>, String> {
        let limit = limit.clamp(1, 100).to_string();
        let params = [("search", query), ("filter", "gguf"), ("sort", "downloads"), ("limit", &limit)];
        let url = with_params(&format!("{}/api/models", self.base), params.into_iter().chain(EXPAND))?;
        let res = self
            .get(url.as_str())
            .send()
            .await
            .map_err(|e| format!("contacting Hugging Face: {e}"))?;
        if !res.status().is_success() {
            return Err(format!("Hugging Face search failed: HTTP {}", res.status()));
        }
        let hits: Vec<Value> = res.json().await.map_err(|e| format!("search results: {e}"))?;
        Ok(hits.iter().filter_map(repo_info).collect())
    }

    /// Metadata of one repo (same shape as a search hit).
    pub async fn info(&self, repo: &str) -> Result<RepoInfo, String> {
        if !valid_repo(repo) {
            return Err(format!("invalid repo `{repo}`"));
        }
        let url = with_params(&format!("{}/api/models/{repo}", self.base), EXPAND)?;
        let res = self
            .get(url.as_str())
            .send()
            .await
            .map_err(|e| format!("contacting Hugging Face: {e}"))?;
        if res.status().as_u16() == 404 {
            return Err(format!("{repo} not found on Hugging Face"));
        }
        let v: Value = res.json().await.map_err(|e| format!("{repo}: {e}"))?;
        repo_info(&v).ok_or_else(|| format!("{repo}: unexpected response"))
    }
}

/// Metadata fields requested from the Hub API.
const EXPAND: [(&str, &str); 5] = [
    ("expand[]", "gguf"),
    ("expand[]", "downloads"),
    ("expand[]", "likes"),
    ("expand[]", "pipeline_tag"),
    ("expand[]", "gated"),
];

fn with_params<'a>(url: &str, params: impl IntoIterator<Item = (&'a str, &'a str)>) -> Result<reqwest::Url, String> {
    reqwest::Url::parse_with_params(url, params).map_err(|e| e.to_string())
}

fn repo_info(v: &Value) -> Option<RepoInfo> {
    let g = &v["gguf"];
    let template = g["chat_template"].as_str().unwrap_or_default();
    Some(RepoInfo {
        repo: v["id"].as_str()?.to_owned(),
        downloads: v["downloads"].as_u64().unwrap_or(0),
        likes: v["likes"].as_u64().unwrap_or(0),
        pipeline: v["pipeline_tag"].as_str().map(str::to_owned),
        arch: g["architecture"].as_str().map(str::to_owned),
        params: g["total"].as_u64(),
        context_length: g["context_length"].as_u64().map(|c| c as u32),
        tools: template.contains("tools"),
        reasoning: template.contains("think") || template.contains("reasoning"),
        // `false`, or the gating mode ("auto", "manual")
        gated: v["gated"].as_bool().unwrap_or(v["gated"].is_string()),
    })
}

/// Write via a temporary file so readers never see a partial file.
pub async fn write_atomic(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        tokio::fs::create_dir_all(dir).await?;
    }
    let tmp = path.with_extension("tmp");
    tokio::fs::write(&tmp, bytes).await?;
    tokio::fs::rename(&tmp, path).await
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use axum::Router;
    use axum::extract::Path;
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    use axum::routing::get;

    use super::*;

    async fn mock(hits: Arc<AtomicUsize>, up: Arc<std::sync::atomic::AtomicBool>) -> String {
        let tree = move |Path((owner, name)): Path<(String, String)>| {
            let hits = hits.clone();
            let up = up.clone();
            async move {
                hits.fetch_add(1, Ordering::SeqCst);
                if !up.load(Ordering::SeqCst) {
                    return StatusCode::SERVICE_UNAVAILABLE.into_response();
                }
                if owner == "gated" {
                    return StatusCode::UNAUTHORIZED.into_response();
                }
                axum::Json(serde_json::json!([
                    {"type": "directory", "path": "BF16", "size": 0},
                    {"type": "file", "path": format!("{name}-Q4_K_M.gguf"), "size": 100, "lfs": {"oid": "ab"}},
                    {"type": "file", "path": "mmproj-F16.gguf", "size": 5},
                    {"type": "file", "path": "README.md", "size": 1}
                ]))
                .into_response()
            }
        };
        let search = || async {
            axum::Json(serde_json::json!([{
                "id": "o/m-GGUF", "downloads": 7, "pipeline_tag": "text-generation", "gated": "auto",
                "gguf": {"architecture": "qwen3", "total": 8000000000u64, "context_length": 40960,
                         "chat_template": "{% if tools %}<think>"}
            }]))
        };
        let app = Router::new()
            .route("/api/models/{owner}/{name}/tree/main", get(tree))
            .route("/api/models", get(search));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn listing_is_cached_and_survives_outages() {
        let hits = Arc::new(AtomicUsize::new(0));
        let up = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let base = mock(hits.clone(), up.clone()).await;
        let dir = tempfile::tempdir().unwrap();
        let hf = HfClient::new(reqwest::Client::new(), &base, dir.path().into());

        let l = hf.listing("o/m-GGUF", LISTING_TTL).await.unwrap();
        assert_eq!(l.quants[0].label, "Q4_K_M");
        assert_eq!(l.quants[0].files[0].sha256.as_deref(), Some("ab"));
        assert_eq!(l.mmproj.len(), 1);
        hf.listing("o/m-GGUF", LISTING_TTL).await.unwrap();
        assert_eq!(hits.load(Ordering::SeqCst), 1, "second call from cache");

        up.store(false, Ordering::SeqCst);
        let stale = hf.listing("o/m-GGUF", Duration::ZERO).await.unwrap();
        assert_eq!(stale, l, "stale cache when offline");
        assert_eq!(hits.load(Ordering::SeqCst), 2);
        assert!(hf.listing("o/other", Duration::ZERO).await.is_err(), "no cache, no network");

        up.store(true, Ordering::SeqCst);
        assert!(hf.listing("gated/m", LISTING_TTL).await.unwrap_err().contains("HF_TOKEN"));
        assert!(hf.listing("../etc", LISTING_TTL).await.is_err());
    }

    #[tokio::test]
    async fn search_maps_metadata() {
        let base = mock(Default::default(), Arc::new(true.into())).await;
        let hf = HfClient::new(reqwest::Client::new(), &base, tempfile::tempdir().unwrap().path().into());
        let hits = hf.search("m", 10).await.unwrap();
        assert_eq!(hits.len(), 1);
        let h = &hits[0];
        assert_eq!((h.repo.as_str(), h.params, h.context_length), ("o/m-GGUF", Some(8_000_000_000), Some(40960)));
        assert!(h.tools && h.reasoning && h.gated);
    }
}
