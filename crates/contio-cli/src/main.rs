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
    /// List configured models and their state.
    #[command(visible_alias = "ps")]
    Models,
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
    }
    Ok(())
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
        let res = self
            .http
            .post(format!("{}{path}", self.base))
            .json(body)
            .send()
            .await
            .with_context(|| self.unreachable())?;
        Self::json(res).await
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
