//! Minimal stand-in for an OpenAI-compatible backend (llama-server), used by
//! integration tests to exercise the real process lifecycle. Std only.
//!
//! Usage: `contio-fake-backend --port N`
//! Env:   `FAKE_STARTUP_MS` delay before listening, `FAKE_EXIT=1` crash on start.
//!
//! Routes: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`
//! (JSON, or SSE when the body has `"stream":true`), `POST /v1/embeddings`.
//! Non-streaming chat responses echo the request body under `echo`.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let port = args
        .iter()
        .position(|a| a == "--port")
        .and_then(|i| args.get(i + 1))
        .and_then(|p| p.parse::<u16>().ok())
        .expect("--port N required");

    if std::env::var("FAKE_EXIT").is_ok_and(|v| v == "1") {
        eprintln!("fake backend: simulated startup failure");
        std::process::exit(3);
    }
    if let Some(ms) = std::env::var("FAKE_STARTUP_MS").ok().and_then(|v| v.parse().ok()) {
        std::thread::sleep(Duration::from_millis(ms));
    }

    let listener = TcpListener::bind(("127.0.0.1", port)).expect("bind");
    eprintln!("fake backend listening on {port}");
    for stream in listener.incoming().flatten() {
        std::thread::spawn(move || {
            let _ = handle(stream);
        });
    }
}

fn handle(mut stream: TcpStream) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;
    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        if let Some((k, v)) = line.split_once(':')
            && k.eq_ignore_ascii_case("content-length")
        {
            content_length = v.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0; content_length];
    reader.read_exact(&mut body)?;
    let body = String::from_utf8_lossy(&body);

    let mut parts = request_line.split_whitespace();
    let (method, path) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
    match (method, path) {
        ("GET", "/health") => json(&mut stream, 200, r#"{"status":"ok"}"#),
        ("GET", "/v1/models") => json(&mut stream, 200, r#"{"object":"list","data":[{"id":"fake"}]}"#),
        ("POST", "/v1/chat/completions") if body.replace(' ', "").contains(r#""stream":true"#) => {
            stream.write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
            )?;
            for word in ["Hello", " from", " fake"] {
                let chunk = format!(
                    "data: {{\"object\":\"chat.completion.chunk\",\"choices\":[{{\"delta\":{{\"content\":\"{word}\"}}}}]}}\n\n"
                );
                stream.write_all(chunk.as_bytes())?;
                stream.flush()?;
                std::thread::sleep(Duration::from_millis(20));
            }
            stream.write_all(b"data: [DONE]\n\n")
        }
        ("POST", "/v1/chat/completions") => json(
            &mut stream,
            200,
            &format!(
                r#"{{"object":"chat.completion","choices":[{{"message":{{"role":"assistant","content":"fake"}}}}],"echo":{body}}}"#
            ),
        ),
        ("POST", "/v1/embeddings") => json(&mut stream, 200, r#"{"data":[{"embedding":[0.1,0.2]}]}"#),
        _ => json(&mut stream, 404, r#"{"error":{"message":"not found"}}"#),
    }
}

fn json(stream: &mut TcpStream, status: u16, body: &str) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}
