//! Server-sent events: parsing upstream streams and encoding our own.

use axum::body::Bytes;
use futures_util::stream::BoxStream;
use futures_util::{Stream, StreamExt, stream};
use serde_json::Value;

/// Turn an SSE byte stream into its `data:` payloads, one item per event.
/// Ends at `data: [DONE]` or end of input. Transport errors end the stream with
/// an `Err` item.
pub fn data_events<S, E>(bytes: S) -> BoxStream<'static, Result<String, String>>
where
    S: Stream<Item = Result<Bytes, E>> + Send + Unpin + 'static,
    E: std::fmt::Display,
{
    struct State<S> {
        input: S,
        buf: String,
        done: bool,
    }
    stream::unfold(State { input: bytes, buf: String::new(), done: false }, |mut st| async move {
        loop {
            if st.done {
                return None;
            }
            if let Some(event) = take_event(&mut st.buf) {
                let data = event_data(&event);
                match data.as_deref() {
                    None => continue, // comment / keep-alive / event without data
                    Some("[DONE]") => return None,
                    Some(_) => return Some((Ok(data.unwrap()), st)),
                }
            }
            match st.input.next().await {
                Some(Ok(chunk)) => st.buf.push_str(&String::from_utf8_lossy(&chunk).replace("\r\n", "\n")),
                Some(Err(e)) => {
                    st.done = true;
                    return Some((Err(e.to_string()), st));
                }
                None => {
                    st.done = true;
                    // a final event without trailing blank line
                    let rest = std::mem::take(&mut st.buf);
                    return event_data(&rest).filter(|d| d != "[DONE]").map(|d| (Ok(d), st));
                }
            }
        }
    })
    .boxed()
}

fn take_event(buf: &mut String) -> Option<String> {
    let end = buf.find("\n\n")?;
    let event = buf[..end].to_owned();
    buf.drain(..end + 2);
    Some(event)
}

fn event_data(event: &str) -> Option<String> {
    let lines: Vec<&str> = event
        .lines()
        .filter_map(|l| l.strip_prefix("data:"))
        .map(|d| d.strip_prefix(' ').unwrap_or(d))
        .collect();
    (!lines.is_empty()).then(|| lines.join("\n"))
}

/// Encode one SSE event.
pub fn encode(event: Option<&str>, data: &Value) -> Bytes {
    let mut s = String::new();
    if let Some(e) = event {
        s.push_str("event: ");
        s.push_str(e);
        s.push('\n');
    }
    s.push_str("data: ");
    s.push_str(&data.to_string());
    s.push_str("\n\n");
    Bytes::from(s)
}

pub fn done() -> Bytes {
    Bytes::from_static(b"data: [DONE]\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn parse(chunks: &[&str]) -> Vec<Result<String, String>> {
        let input = stream::iter(
            chunks.iter().map(|c| Ok::<_, std::io::Error>(Bytes::from(c.to_string()))).collect::<Vec<_>>(),
        );
        data_events(input).collect().await
    }

    #[tokio::test]
    async fn splits_events_across_chunk_boundaries() {
        let got = parse(&["data: {\"a\"", ":1}\n\nda", "ta: {\"b\":2}\r\n\r\n: keep-alive\n\ndata: [DONE]\n\ndata: x\n\n"]).await;
        assert_eq!(got, vec![Ok("{\"a\":1}".into()), Ok("{\"b\":2}".into())]);
    }

    #[tokio::test]
    async fn handles_missing_trailing_newline_and_multiline_data() {
        let got = parse(&["event: x\ndata: line1\ndata: line2\n\ndata: tail"]).await;
        assert_eq!(got, vec![Ok("line1\nline2".into()), Ok("tail".into())]);
    }

    #[test]
    fn encodes_named_events() {
        let b = encode(Some("message_stop"), &serde_json::json!({"type": "message_stop"}));
        assert_eq!(&b[..], b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n");
    }
}
