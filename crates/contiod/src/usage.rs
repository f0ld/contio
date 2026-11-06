//! API traffic per model: requests, prompt and generated tokens, seconds of
//! audio. "Session" counts since the daemon started (memory); daily totals
//! go to `bench.db` (`usage_daily`), so "today" and "all time" survive restarts.
//!
//! Every backend response goes through a [`Tapped`] response, which reads the
//! OpenAI `usage` object as the bytes pass by (SSE chunks, or the end of a
//! JSON body) and records the request when the response is done or dropped.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use axum::Json;
use axum::body::Bytes;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use contio_bench::store::{Store, UsageCounts};
use futures_util::{Stream, StreamExt};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::api::AppState;

/// How much of a JSON body's end is kept to find `usage` (it comes last).
const TAIL: usize = 64 * 1024;

#[derive(Clone)]
pub struct UsageTracker {
    /// Unix seconds.
    pub started_at: u64,
    session: Arc<Mutex<BTreeMap<String, UsageCounts>>>,
    store: Store,
    /// Seconds east of UTC, for day boundaries in local time.
    utc_offset: i64,
}

impl UsageTracker {
    pub fn new(store: Store) -> Self {
        Self {
            started_at: unix_now(),
            session: Default::default(),
            store,
            utc_offset: local_utc_offset(),
        }
    }

    pub fn record(&self, u: UsageCounts) {
        {
            let mut s = self.session.lock().expect("usage lock poisoned");
            s.entry(u.model.clone()).or_insert_with(|| UsageCounts { model: u.model.clone(), ..Default::default() }).add(&u);
        }
        let store = self.store.clone();
        let day = self.day(unix_now());
        let write = move || {
            if let Err(e) = store.add_usage(&day, &u) {
                tracing::warn!("recording usage: {e}");
            }
        };
        match tokio::runtime::Handle::try_current() {
            Ok(h) => {
                h.spawn_blocking(write);
            }
            Err(_) => write(),
        }
    }

    pub fn session(&self) -> Vec<UsageCounts> {
        self.session.lock().expect("usage lock poisoned").values().cloned().collect()
    }

    /// YYYY-MM-DD in local time.
    pub fn day(&self, unix: u64) -> String {
        let days = (unix as i64 + self.utc_offset).div_euclid(86_400);
        let (y, m, d) = civil_from_days(days);
        format!("{y:04}-{m:02}-{d:02}")
    }

    pub fn store(&self) -> &Store {
        &self.store
    }
}

#[derive(Debug, Deserialize)]
pub struct UsageQuery {
    /// Also report the last N days (today included).
    pub days: Option<u32>,
}

#[derive(Debug, Serialize)]
struct ModelUsage {
    model: String,
    session: UsageCounts,
    today: UsageCounts,
    total: UsageCounts,
    #[serde(skip_serializing_if = "Option::is_none")]
    period: Option<UsageCounts>,
}

/// `GET /api/usage[?days=N]`: per model, since the daemon started, today,
/// all time, and optionally the last N days.
pub async fn report(State(state): State<AppState>, Query(q): Query<UsageQuery>) -> Response {
    let t = state.usage.clone();
    let now = unix_now();
    let today = t.day(now);
    let since = q.days.filter(|d| *d > 0).map(|d| t.day(now.saturating_sub(u64::from(d - 1) * 86_400)));
    let read = {
        let t = t.clone();
        let (today, since) = (today.clone(), since.clone());
        move || -> Result<_, String> {
            let s = t.store();
            Ok((s.usage_totals(None)?, s.usage_totals(Some(&today))?, since.map(|d| s.usage_totals(Some(&d))).transpose()?))
        }
    };
    let (total, today_rows, period) = match tokio::task::spawn_blocking(read).await {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": {"message": e}}))).into_response(),
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": {"message": e.to_string()}}))).into_response(),
    };
    let session = t.session();
    let mut models: Vec<String> = total.iter().chain(&session).map(|u| u.model.clone()).collect();
    models.sort();
    models.dedup();
    let pick = |rows: &[UsageCounts], m: &str| {
        rows.iter().find(|u| u.model == m).cloned().unwrap_or_else(|| UsageCounts { model: m.to_owned(), ..Default::default() })
    };
    let rows: Vec<ModelUsage> = models
        .iter()
        .map(|m| ModelUsage {
            model: m.clone(),
            session: pick(&session, m),
            today: pick(&today_rows, m),
            total: pick(&total, m),
            period: period.as_ref().map(|p| pick(p, m)),
        })
        .collect();
    Json(json!({
        "session_started_at": t.started_at,
        "today": today,
        "period_since": since,
        "models": rows,
    }))
    .into_response()
}

fn unix_now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or_default()
}

/// `date +%z` (e.g. `+0100`) once at startup; UTC if unavailable.
fn local_utc_offset() -> i64 {
    let out = std::process::Command::new("date").arg("+%z").output().ok();
    let s = out.map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned()).unwrap_or_default();
    parse_offset(&s).unwrap_or(0)
}

fn parse_offset(s: &str) -> Option<i64> {
    let (sign, rest) = match s.as_bytes().first()? {
        b'+' => (1, &s[1..]),
        b'-' => (-1, &s[1..]),
        _ => return None,
    };
    if rest.len() != 4 {
        return None;
    }
    let h: i64 = rest[..2].parse().ok()?;
    let m: i64 = rest[2..].parse().ok()?;
    Some(sign * (h * 3600 + m * 60))
}

/// Days since 1970-01-01 → (year, month, day) (Howard Hinnant's algorithm).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

/// `usage` of an OpenAI-style response object.
fn counts(usage: &Value) -> Option<(u64, u64)> {
    let p = usage["prompt_tokens"].as_u64().or(usage["input_tokens"].as_u64());
    let c = usage["completion_tokens"].as_u64().or(usage["output_tokens"].as_u64());
    (p.is_some() || c.is_some()).then(|| (p.unwrap_or(0), c.unwrap_or(0)))
}

/// The last `"usage": {…}` object in (the end of) a JSON body.
pub fn usage_in_json_tail(tail: &[u8]) -> Option<(u64, u64)> {
    let text = String::from_utf8_lossy(tail);
    let key = text.rfind("\"usage\"")?;
    let start = key + text[key..].find('{')?;
    let mut depth = 0usize;
    for (i, ch) in text[start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    let v: Value = serde_json::from_str(&text[start..=start + i]).ok()?;
                    return counts(&v);
                }
            }
            _ => {}
        }
    }
    None
}

/// What one request has used so far; recorded once, when dropped.
struct Meter {
    tracker: Option<UsageTracker>,
    model: String,
    usage: Option<(u64, u64)>,
    audio_seconds: f64,
}

impl Drop for Meter {
    fn drop(&mut self) {
        if let Some(t) = self.tracker.take() {
            let (p, c) = self.usage.unwrap_or_default();
            t.record(UsageCounts {
                model: std::mem::take(&mut self.model),
                requests: 1,
                prompt_tokens: p,
                completion_tokens: c,
                audio_seconds: self.audio_seconds,
            });
        }
    }
}

/// A backend response whose usage is counted. Mirrors the parts of
/// `reqwest::Response` the gateway uses.
pub struct Tapped {
    inner: reqwest::Response,
    meter: Meter,
    /// Drop the usage-only SSE event: it was requested for counting, not by the client.
    strip_usage: bool,
}

impl Tapped {
    /// Not counted (tests, internal calls).
    pub fn untracked(inner: reqwest::Response) -> Self {
        Self { inner, meter: Meter { tracker: None, model: String::new(), usage: None, audio_seconds: 0.0 }, strip_usage: false }
    }

    pub fn new(inner: reqwest::Response, tracker: UsageTracker, model: &str) -> Self {
        let meter = Meter { tracker: Some(tracker), model: model.to_owned(), usage: None, audio_seconds: 0.0 };
        Self { inner, meter, strip_usage: false }
    }

    pub fn with_audio_seconds(mut self, seconds: f64) -> Self {
        self.meter.audio_seconds = seconds;
        self
    }

    pub fn strip_usage(&mut self, strip: bool) {
        self.strip_usage = strip;
    }

    pub fn status(&self) -> reqwest::StatusCode {
        self.inner.status()
    }

    pub fn headers(&self) -> &reqwest::header::HeaderMap {
        self.inner.headers()
    }

    pub async fn bytes(self) -> reqwest::Result<Bytes> {
        let Tapped { inner, mut meter, .. } = self;
        let b = inner.bytes().await?;
        meter.usage = usage_in_json_tail(&b[b.len().saturating_sub(TAIL)..]);
        Ok(b) // meter dropped: recorded
    }

    pub async fn text(self) -> reqwest::Result<String> {
        self.bytes().await.map(|b| String::from_utf8_lossy(&b).into_owned())
    }

    pub async fn json<T: DeserializeOwned>(self) -> Result<T, String> {
        let b = self.bytes().await.map_err(|e| e.to_string())?;
        serde_json::from_slice(&b).map_err(|e| format!("bad backend response: {e}"))
    }

    /// The body as it arrives; usage is read from SSE events (or a JSON
    /// body's end) and counted when the stream ends or is dropped.
    pub fn bytes_stream(self) -> impl Stream<Item = reqwest::Result<Bytes>> + Send + Unpin + 'static {
        let Tapped { inner, meter, strip_usage } = self;
        let sse = inner
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|ct| ct.starts_with("text/event-stream"));
        let tap = StreamTap { meter, sse, strip: sse && strip_usage, buf: Vec::new() };
        futures_util::stream::unfold(Some((inner.bytes_stream(), tap)), |state| async move {
            let (mut body, mut tap) = state?;
            match body.next().await {
                Some(Ok(b)) => {
                    let out = tap.observe(&b);
                    Some((Ok(out), Some((body, tap))))
                }
                Some(Err(e)) => Some((Err(e), Some((body, tap)))),
                // end: whatever is held back; dropping the tap records the request
                None => {
                    let rest = tap.finish();
                    (!rest.is_empty()).then_some((Ok(rest), None))
                }
            }
        })
        .boxed()
    }
}

/// Watches a body stream; owns the meter, so dropping the stream records the request.
struct StreamTap {
    meter: Meter,
    sse: bool,
    strip: bool,
    /// SSE: bytes after the last complete event. JSON: the body's tail.
    buf: Vec<u8>,
}

impl StreamTap {
    fn observe(&mut self, chunk: &Bytes) -> Bytes {
        if !self.sse {
            self.buf.extend_from_slice(chunk);
            if self.buf.len() > 2 * TAIL {
                self.buf.drain(..self.buf.len() - TAIL);
            }
            if let Some(u) = usage_in_json_tail(&self.buf) {
                self.meter.usage = Some(u);
            }
            return chunk.clone();
        }
        self.buf.extend_from_slice(chunk);
        let mut out = Vec::with_capacity(chunk.len());
        // complete events end with a blank line
        while let Some(end) = find_event_end(&self.buf) {
            let event: Vec<u8> = self.buf.drain(..end).collect();
            if !self.inspect(&event) {
                out.extend_from_slice(&event);
            }
        }
        if !self.strip {
            // pass bytes through unchanged; only the copy above was parsed
            return chunk.clone();
        }
        Bytes::from(out)
    }

    /// End of the body: a last event without its blank line.
    fn finish(&mut self) -> Bytes {
        if !self.sse {
            return Bytes::new();
        }
        let rest = std::mem::take(&mut self.buf);
        if rest.is_empty() || self.inspect(&rest) || !self.strip {
            return Bytes::new(); // without stripping, these bytes were already passed on
        }
        Bytes::from(rest)
    }

    /// Read usage from one SSE event; true if it should be dropped.
    fn inspect(&mut self, event: &[u8]) -> bool {
        let text = String::from_utf8_lossy(event);
        let data: String = text
            .lines()
            .filter_map(|l| l.strip_prefix("data:"))
            .map(str::trim_start)
            .collect::<Vec<_>>()
            .join("\n");
        let Ok(v) = serde_json::from_str::<Value>(&data) else { return false };
        let Some(u) = v.get("usage").filter(|u| u.is_object()).and_then(counts) else { return false };
        self.meter.usage = Some(u);
        self.strip && v["choices"].as_array().is_some_and(Vec::is_empty)
    }
}

/// End (exclusive, including the blank line) of the first complete SSE event.
fn find_event_end(buf: &[u8]) -> Option<usize> {
    let lf = buf.windows(2).position(|w| w == b"\n\n").map(|i| i + 2);
    let crlf = buf.windows(4).position(|w| w == b"\r\n\r\n").map(|i| i + 4);
    match (lf, crlf) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

/// Duration of WAV or FLAC audio from its header (other formats: 0).
pub fn audio_seconds(data: &[u8]) -> f64 {
    if data.len() >= 42 && &data[..4] == b"fLaC" {
        let si = &data[8..42];
        let rate = (u32::from(si[10]) << 12) | (u32::from(si[11]) << 4) | (u32::from(si[12]) >> 4);
        let samples = (u64::from(si[13] & 0x0f) << 32)
            | (u64::from(si[14]) << 24)
            | (u64::from(si[15]) << 16)
            | (u64::from(si[16]) << 8)
            | u64::from(si[17]);
        return if rate > 0 { samples as f64 / f64::from(rate) } else { 0.0 };
    }
    if data.len() >= 44 && &data[..4] == b"RIFF" && &data[8..12] == b"WAVE" {
        // walk the chunks for fmt (byte rate) and data (size)
        let (mut pos, mut byte_rate, mut size) = (12usize, 0u32, 0u32);
        while pos + 8 <= data.len() {
            let id = &data[pos..pos + 4];
            let len = u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]);
            if id == b"fmt " && pos + 20 <= data.len() {
                byte_rate = u32::from_le_bytes([data[pos + 16], data[pos + 17], data[pos + 18], data[pos + 19]]);
            } else if id == b"data" {
                size = len;
                break;
            }
            pos += 8 + len as usize + (len as usize & 1);
        }
        return if byte_rate > 0 { f64::from(size) / f64::from(byte_rate) } else { 0.0 };
    }
    0.0
}

impl From<reqwest::Response> for Tapped {
    fn from(r: reqwest::Response) -> Self {
        Self::untracked(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_and_offsets() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(20_398), (2025, 11, 6));
        assert_eq!(civil_from_days(11_016), (2000, 2, 29));
        assert_eq!(parse_offset("+0100"), Some(3600));
        assert_eq!(parse_offset("-0530"), Some(-19_800));
        assert_eq!(parse_offset("Z"), None);
    }

    #[test]
    fn usage_from_json_ends() {
        let body = br#"{"choices":[{"message":{"content":"say \"usage\" {"}}],"usage":{"prompt_tokens":12,"completion_tokens":7,"total_tokens":19}}"#;
        assert_eq!(usage_in_json_tail(body), Some((12, 7)));
        let emb = br#"{"data":[{"embedding":[0.1]}],"usage":{"prompt_tokens":5,"total_tokens":5}}"#;
        assert_eq!(usage_in_json_tail(emb), Some((5, 0)));
        assert_eq!(usage_in_json_tail(br#"{"text":"hi"}"#), None);
    }

    fn tap(strip: bool) -> StreamTap {
        StreamTap { meter: Meter { tracker: None, model: "m".into(), usage: None, audio_seconds: 0.0 }, sse: true, strip, buf: vec![] }
    }

    const STREAM: &str = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n\
                          data: {\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":2}}\n\n\
                          data: [DONE]\n\n";

    #[test]
    fn sse_usage_is_read_across_chunk_boundaries() {
        let mut t = tap(false);
        let bytes = STREAM.as_bytes();
        let mut out = Vec::new();
        for piece in bytes.chunks(7) {
            out.extend_from_slice(&t.observe(&Bytes::copy_from_slice(piece)));
        }
        assert_eq!(t.meter.usage, Some((9, 2)));
        assert_eq!(out, bytes, "unchanged when not stripping");
    }

    #[test]
    fn injected_usage_event_is_stripped() {
        let mut t = tap(true);
        let mut out = Vec::new();
        for piece in STREAM.as_bytes().chunks(11) {
            out.extend_from_slice(&t.observe(&Bytes::copy_from_slice(piece)));
        }
        let text = String::from_utf8(out).unwrap();
        assert_eq!(t.meter.usage, Some((9, 2)));
        assert!(text.contains("\"hi\"") && text.contains("[DONE]") && !text.contains("usage"), "{text}");
    }

    #[test]
    fn wav_and_flac_durations() {
        // 1 s of 16 kHz mono 16-bit WAV
        let mut wav = b"RIFF\0\0\0\0WAVEfmt \x10\0\0\0\x01\0\x01\0".to_vec();
        wav.extend_from_slice(&16_000u32.to_le_bytes());
        wav.extend_from_slice(&32_000u32.to_le_bytes());
        wav.extend_from_slice(b"\x02\0\x10\0data");
        wav.extend_from_slice(&32_000u32.to_le_bytes());
        wav.resize(wav.len() + 32_000, 0);
        assert!((audio_seconds(&wav) - 1.0).abs() < 1e-9);
        assert_eq!(audio_seconds(b"ID3 not audio we parse"), 0.0);
    }
}
