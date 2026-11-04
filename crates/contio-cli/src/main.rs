//! `contio`: command-line client for the contio daemon (spec §10).

mod format;

use std::io::{Read, Write};

use anyhow::{Context, bail};
use clap::{Parser, Subcommand};
use contio_core::hw::HardwareInfo;
use contio_core::model::ModelStatus;
use futures_util::StreamExt;
use serde_json::{Value, json};

#[derive(Debug, Parser)]
#[command(version, about = "Manage local LLMs served by contiod")]
struct Cli {
    /// Daemon address.
    #[arg(long, global = true, env = "CONTIO_URL", default_value = "http://127.0.0.1:8080")]
    url: String,

    /// Print raw JSON instead of tables.
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Daemon health and a model summary.
    Status,
    /// Hardware report (CPU, GPU, NPU, memory pools, power).
    Hw,
    /// List installed models and their state.
    #[command(visible_alias = "ps")]
    Models,
    /// Browse the catalog (or Hugging Face with --hf).
    Search {
        /// Words matched against id, name and family; empty = everything.
        query: Vec<String>,
        /// Only models with a version for this device.
        #[arg(long, value_parser = ["gpu", "cpu", "npu"])]
        device: Option<String>,
        /// Only models whose suggested quant fits.
        #[arg(long)]
        fits: bool,
        #[arg(long, value_parser = ["chat", "embedding", "reranker"])]
        kind: Option<String>,
        /// Search all GGUF repos on Hugging Face instead.
        #[arg(long)]
        hf: bool,
    },
    /// Quantizations of a model with sizes and fit: catalog id, `owner/repo`
    /// or `flm:<tag>`.
    Show {
        model: String,
        /// Compute memory fits for this context length (tokens).
        #[arg(short, long)]
        context: Option<u32>,
    },
    /// Download and register a model; installs a missing backend first.
    Install {
        /// Catalog id, Hugging Face `owner/repo`, or `flm:<tag>` (NPU).
        model: String,
        /// e.g. Q4_K_M [default: recommended, or the largest that fits]
        #[arg(long, short)]
        quant: Option<String>,
        /// A single file of the repo instead of a quant.
        #[arg(long, conflicts_with = "quant")]
        file: Option<String>,
        #[arg(long, short, value_parser = ["gpu", "cpu", "npu"])]
        device: Option<String>,
        /// Context length in tokens [default: from the catalog].
        #[arg(long, short)]
        context: Option<u32>,
        /// Name in the API [default: derived from the model].
        #[arg(long)]
        name: Option<String>,
        /// Skip the vision projector of multimodal models.
        #[arg(long)]
        no_vision: bool,
        /// Start the model when done.
        #[arg(long)]
        run: bool,
        #[arg(long)]
        detach: bool,
    },
    /// Remove an installed model (added through contio, not config.toml).
    Rm {
        model: String,
        /// Also delete its files from contio's model directory.
        #[arg(long)]
        delete_files: bool,
    },
    /// Change launch settings of an installed model.
    Set {
        model: String,
        #[arg(long)]
        context: Option<u32>,
        #[arg(long, conflicts_with = "unpin")]
        pin: bool,
        #[arg(long)]
        unpin: bool,
        #[arg(long, value_parser = ["gpu", "cpu"])]
        device: Option<String>,
        /// Unload after this many idle seconds (0 = daemon default).
        #[arg(long)]
        idle_ttl: Option<u64>,
    },
    /// Load a model (and keep it loaded until idle/evicted).
    Load { model: String },
    /// Unload a model.
    Unload { model: String },
    /// Send one prompt and stream the answer. Reads stdin if no prompt is given.
    Chat {
        model: String,
        prompt: Option<String>,
        /// Optional system prompt.
        #[arg(long, short)]
        system: Option<String>,
        #[arg(long, default_value_t = 2048)]
        max_tokens: u32,
        /// Also print the model's reasoning (dimmed, to stderr).
        #[arg(long)]
        show_reasoning: bool,
    },
    /// Other names for models: list, set, remove.
    Alias {
        #[command(subcommand)]
        action: Option<AliasAction>,
    },
    /// Inference backends: list, install, check prerequisites.
    Backends {
        #[command(subcommand)]
        action: Option<BackendsAction>,
    },
    /// Background jobs (downloads, installs).
    Jobs,
    /// Cancel a running job.
    Cancel { id: u64 },
    /// Run a failed or cancelled job again.
    Retry { id: u64 },
}

#[derive(Debug, Subcommand)]
enum AliasAction {
    /// All aliases (default).
    List,
    /// Create or retarget an alias: `contio alias set coder qwen3.6-35b-a3b`.
    Set {
        name: String,
        model: String,
        /// Model name sent to the backend (models co-loaded in one process).
        #[arg(long)]
        upstream_model: Option<String>,
    },
    /// Remove an alias (only ones added through contio).
    Rm { name: String },
}

#[derive(Debug, Subcommand)]
enum BackendsAction {
    /// Registered and installed backends (default).
    List,
    /// Install llama.cpp (source | prebuilt | podman) or FastFlowLM (source).
    Install {
        #[arg(value_parser = ["llama.cpp", "flm"])]
        backend: String,
        #[arg(long, value_parser = ["source", "prebuilt", "podman"])]
        method: String,
        /// Release tag / commit [default: the version tested by contio]
        #[arg(long)]
        version: Option<String>,
        /// Only check prerequisites.
        #[arg(long)]
        check: bool,
        #[arg(long)]
        detach: bool,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    if let Err(e) = run(cli).await {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

async fn run(cli: Cli) -> anyhow::Result<()> {
    let client = Client { http: reqwest::Client::new(), base: cli.url.trim_end_matches('/').to_owned() };
    match cli.command {
        Command::Status => {
            let health = client.get("/api/health").await?;
            let models: Vec<ModelStatus> = serde_json::from_value(client.get("/api/models").await?)?;
            if cli.json {
                println!("{}", json!({ "health": health, "models": models }));
            } else {
                println!("{}", format::status(&client.base, &health, &models));
            }
        }
        Command::Hw => {
            let raw = client.get("/api/hw").await?;
            if cli.json {
                println!("{raw}");
            } else {
                let hw: HardwareInfo = serde_json::from_value(raw)?;
                println!("{}", format::hardware(&hw));
            }
        }
        Command::Models => {
            let raw = client.get("/api/models").await?;
            if cli.json {
                println!("{raw}");
            } else {
                let models: Vec<ModelStatus> = serde_json::from_value(raw)?;
                println!("{}", format::models(&models));
            }
        }
        Command::Load { model } => {
            eprint!("loading {model}… ");
            let started = std::time::Instant::now();
            let status = client.post(&format!("/api/models/{model}/load"), &json!({})).await?;
            eprintln!("ready in {:.1}s", started.elapsed().as_secs_f32());
            if cli.json {
                println!("{status}");
            }
        }
        Command::Unload { model } => {
            let r = client.post(&format!("/api/models/{model}/unload"), &json!({})).await?;
            if r["unloaded"] == true {
                eprintln!("unloaded {model}");
            } else {
                eprintln!("{model} was not loaded");
            }
        }
        Command::Chat { model, prompt, system, max_tokens, show_reasoning } => {
            let prompt = match prompt {
                Some(p) => p,
                None => {
                    let mut s = String::new();
                    std::io::stdin().read_to_string(&mut s)?;
                    s
                }
            };
            if prompt.trim().is_empty() {
                bail!("empty prompt");
            }
            client.chat(&model, system.as_deref(), &prompt, max_tokens, show_reasoning).await?;
        }
        Command::Search { query, device, fits, kind, hf } => {
            let query = query.join(" ");
            if hf {
                if query.is_empty() {
                    bail!("--hf needs a search term");
                }
                let url = format!("/api/catalog/search?q={}", urlencode(&query));
                let raw = client.get(&url).await?;
                if cli.json { println!("{raw}") } else { println!("{}", format::hf_hits(&raw)) }
            } else {
                let raw = client.get("/api/catalog").await?;
                let filter = format::CatalogFilter { query, device, fits, kind };
                if cli.json { println!("{raw}") } else { println!("{}", format::catalog(&raw, &filter)) }
            }
        }
        Command::Show { model, context } => {
            let ctx = context.map(|c| format!("context={c}"));
            let raw = if model.contains('/') {
                let q = ctx.map(|c| format!("&{c}")).unwrap_or_default();
                client.get(&format!("/api/catalog/repo?repo={}{q}", urlencode(&model))).await?
            } else {
                let q = ctx.map(|c| format!("?{c}")).unwrap_or_default();
                client.get(&format!("/api/catalog/models/{}{q}", urlencode(&model))).await?
            };
            if cli.json { println!("{raw}") } else { println!("{}", format::detail(&raw)) }
        }
        Command::Install { model, quant, file, device, context, name, no_vision, run, detach } => {
            let mut body = json!({
                "quant": quant, "file": file, "device": device, "context": context, "id": name,
                "no_vision": no_vision, "load": run,
            });
            if let Some(tag) = model.strip_prefix("flm:") {
                body["flm"] = json!(tag);
            } else if model.contains('/') {
                body["repo"] = json!(model);
            } else {
                body["model"] = json!(model);
            }
            let r = client.post("/api/catalog/install", &body).await?;
            if let Some(result) = client.follow(r["job_id"].as_u64().unwrap_or_default(), detach).await? {
                let id = result["model"].as_str().unwrap_or("?");
                match result["kind"].as_str() {
                    Some("embedding") => eprintln!("installed as `{id}`: POST /v1/embeddings with \"model\": \"{id}\""),
                    Some("reranker") => eprintln!("installed as `{id}`: POST /v1/rerank with \"model\": \"{id}\""),
                    _ => eprintln!("installed as `{id}`; try it: contio chat {id} \"hello\"  (API model name: {id})"),
                }
            }
        }
        Command::Rm { model, delete_files } => {
            let r = client.delete(&format!("/api/models/{}?delete_files={delete_files}", urlencode(&model))).await?;
            let deleted = r["deleted_files"].as_array().map_or(0, Vec::len);
            eprintln!("removed {model}{}", if delete_files { format!(" ({deleted} files deleted)") } else { String::new() });
            let aliases = format::strings(&r["removed_aliases"]);
            if !aliases.is_empty() {
                eprintln!("removed its aliases: {}", aliases.join(", "));
            }
        }
        Command::Set { model, context, pin, unpin, device, idle_ttl } => {
            let pinned = if pin { Some(true) } else if unpin { Some(false) } else { None };
            let body = json!({ "context": context, "pinned": pinned, "device": device, "idle_ttl_secs": idle_ttl });
            let body = Value::Object(body.as_object().cloned().unwrap_or_default().into_iter().filter(|(_, v)| !v.is_null()).collect());
            if body.as_object().is_some_and(|o| o.is_empty()) {
                bail!("nothing to change (see `contio set --help`)");
            }
            client.patch(&format!("/api/models/{}", urlencode(&model)), &body).await?;
            eprintln!("updated {model} (restarts on next use if it was loaded)");
        }
        Command::Alias { action } => match action.unwrap_or(AliasAction::List) {
            AliasAction::List => {
                let raw = client.get("/api/aliases").await?;
                if cli.json { println!("{raw}") } else { println!("{}", format::aliases(&raw)) }
            }
            AliasAction::Set { name, model, upstream_model } => {
                let body = json!({ "model": model, "upstream_model": upstream_model });
                client.send(client.http.put(format!("{}/api/aliases/{}", client.base, urlencode(&name))).json(&body)).await?;
                eprintln!("{name} → {model}");
            }
            AliasAction::Rm { name } => {
                client.delete(&format!("/api/aliases/{}", urlencode(&name))).await?;
                eprintln!("removed alias {name}");
            }
        },
        Command::Backends { action } => match action.unwrap_or(BackendsAction::List) {
            BackendsAction::List => {
                let raw = client.get("/api/backends").await?;
                if cli.json { println!("{raw}") } else { println!("{}", format::backends(&raw)) }
            }
            BackendsAction::Install { backend, method, version, check, detach } => {
                let pf = client.get(&format!("/api/backends/preflight?backend={backend}&method={method}")).await?;
                if check || pf["ok"] != true {
                    println!("{}", format::preflight(&pf));
                    if pf["ok"] != true {
                        bail!("prerequisites missing");
                    }
                    return Ok(());
                }
                let r = client
                    .post("/api/backends/install", &json!({"backend": backend, "method": method, "version": version}))
                    .await?;
                if client.follow(r["job_id"].as_u64().unwrap_or_default(), detach).await?.is_some() {
                    eprintln!("installed and registered; models using {backend} can be loaded now");
                }
            }
        },
        Command::Jobs => {
            let raw = client.get("/api/jobs").await?;
            if cli.json { println!("{raw}") } else { println!("{}", format::jobs(&raw)) }
        }
        Command::Cancel { id } => {
            let r = client.post(&format!("/api/jobs/{id}/cancel"), &json!({})).await?;
            eprintln!("{}", if r["cancelled"] == true { "cancellation requested" } else { "job is not running" });
        }
        Command::Retry { id } => {
            let r = client.post(&format!("/api/jobs/{id}/retry"), &json!({})).await?;
            client.follow(r["job_id"].as_u64().unwrap_or_default(), false).await?;
        }
    }
    Ok(())
}

/// Percent-encode a path or query component.
fn urlencode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

struct Client {
    http: reqwest::Client,
    base: String,
}

impl Client {
    async fn get(&self, path: &str) -> anyhow::Result<Value> {
        let res = self.http.get(format!("{}{path}", self.base)).send().await.with_context(|| self.unreachable())?;
        Self::json(res).await
    }

    async fn post(&self, path: &str, body: &Value) -> anyhow::Result<Value> {
        self.send(self.http.post(format!("{}{path}", self.base)).json(body)).await
    }

    async fn patch(&self, path: &str, body: &Value) -> anyhow::Result<Value> {
        self.send(self.http.patch(format!("{}{path}", self.base)).json(body)).await
    }

    async fn delete(&self, path: &str) -> anyhow::Result<Value> {
        self.send(self.http.delete(format!("{}{path}", self.base))).await
    }

    async fn send(&self, req: reqwest::RequestBuilder) -> anyhow::Result<Value> {
        let res = req.send().await.with_context(|| self.unreachable())?;
        Self::json(res).await
    }

    /// Show a job's progress until it ends. Returns its result on success.
    /// Ctrl-C only stops watching; the job keeps running in the daemon.
    async fn follow(&self, id: u64, detach: bool) -> anyhow::Result<Option<Value>> {
        eprintln!("job {id} started (`contio jobs` to check, `contio cancel {id}` to stop)");
        if detach {
            return Ok(None);
        }
        let mut last_log = 0usize;
        loop {
            let job = self.get(&format!("/api/jobs/{id}")).await?;
            let log = job["log"].as_array().cloned().unwrap_or_default();
            if log.len() < last_log {
                last_log = 0; // ring buffer wrapped
            }
            for line in &log[last_log..] {
                eprintln!("\r\x1b[K\x1b[2m  {}\x1b[0m", line.as_str().unwrap_or_default());
            }
            last_log = log.len();
            eprint!("\r\x1b[K{}", format::job_line(&job));
            match job["state"].as_str() {
                Some("running") => tokio::time::sleep(std::time::Duration::from_millis(500)).await,
                Some("done") => {
                    eprintln!();
                    return Ok(Some(job["result"].clone()));
                }
                _ => {
                    eprintln!();
                    bail!("{}", job["error"].as_str().unwrap_or("job failed"));
                }
            }
        }
    }

    fn unreachable(&self) -> String {
        format!("can't reach contiod at {} (is it running? `cargo run -p contiod`)", self.base)
    }

    async fn json(res: reqwest::Response) -> anyhow::Result<Value> {
        let status = res.status();
        let body: Value = res.json().await.unwrap_or(Value::Null);
        if !status.is_success() {
            let msg = body["error"]["message"].as_str().unwrap_or("request failed");
            bail!("{msg} (HTTP {})", status.as_u16());
        }
        Ok(body)
    }

    async fn chat(
        &self,
        model: &str,
        system: Option<&str>,
        prompt: &str,
        max_tokens: u32,
        show_reasoning: bool,
    ) -> anyhow::Result<()> {
        let mut messages = Vec::new();
        if let Some(s) = system {
            messages.push(json!({"role": "system", "content": s}));
        }
        messages.push(json!({"role": "user", "content": prompt}));
        let body = json!({
            "model": model, "messages": messages, "max_tokens": max_tokens,
            "stream": true, "stream_options": {"include_usage": true},
        });
        let started = std::time::Instant::now();
        let res = self
            .http
            .post(format!("{}/v1/chat/completions", self.base))
            .json(&body)
            .send()
            .await
            .with_context(|| self.unreachable())?;
        if !res.status().is_success() {
            return Self::json(res).await.map(|_| ());
        }

        let mut stream = res.bytes_stream();
        let mut buf = String::new();
        let mut first_token = None;
        let mut usage = Value::Null;
        let (mut out, mut err) = (std::io::stdout(), std::io::stderr());
        while let Some(chunk) = stream.next().await {
            buf.push_str(&String::from_utf8_lossy(&chunk?));
            while let Some(end) = buf.find("\n\n") {
                let event: String = buf.drain(..end + 2).collect();
                let Some(data) = event.trim().strip_prefix("data: ") else { continue };
                if data == "[DONE]" {
                    continue;
                }
                let Ok(v) = serde_json::from_str::<Value>(data) else { continue };
                if v["usage"].is_object() {
                    usage = v["usage"].clone();
                }
                let delta = &v["choices"][0]["delta"];
                if let Some(r) = delta["reasoning_content"].as_str() {
                    // reasoning tokens count as generated tokens, shown or not
                    first_token.get_or_insert_with(|| started.elapsed());
                    if show_reasoning {
                        write!(err, "\x1b[2m{r}\x1b[0m")?;
                        err.flush()?;
                    }
                }
                if let Some(t) = delta["content"].as_str() {
                    first_token.get_or_insert_with(|| started.elapsed());
                    write!(out, "{t}")?;
                    out.flush()?;
                }
            }
        }
        writeln!(out)?;
        let total = started.elapsed().as_secs_f32();
        if let Some(n) = usage["completion_tokens"].as_u64() {
            let ttft = first_token.map_or(0.0, |d| d.as_secs_f32());
            eprintln!(
                "\x1b[2m{n} tokens in {total:.1}s · first token {ttft:.2}s · {:.1} tok/s\x1b[0m",
                n as f32 / (total - ttft).max(0.001)
            );
        }
        Ok(())
    }
}
