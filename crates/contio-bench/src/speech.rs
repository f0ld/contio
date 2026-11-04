//! Speech-to-text accuracy (spec §8.2): word error rate on 20 short
//! LibriSpeech test-clean utterances (`registry/bench/speech.json`, CC BY 4.0),
//! sent through `/v1/audio/transcriptions`, plus the real-time factor.

use std::collections::BTreeMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::perf::{Progress, Target, or_cancel};
use crate::speech_files::FILES;

const SET: &str = include_str!("../../../registry/bench/speech.json");

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Utterance {
    pub id: String,
    pub reference: String,
    pub hypothesis: String,
    pub words: usize,
    pub errors: usize,
    pub audio_s: f64,
    pub seconds: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SpeechReport {
    /// Word errors / reference words over all utterances.
    pub wer: f64,
    /// Processing time / audio time (below 1 = faster than real time).
    pub rtf: f64,
    pub utterances: Vec<Utterance>,
}

impl SpeechReport {
    pub fn metrics(&self) -> BTreeMap<String, f64> {
        BTreeMap::from([("speech_wer".to_owned(), self.wer), ("speech_rtf".to_owned(), self.rtf)])
    }
}

/// Lowercase words without punctuation (LibriSpeech references are uppercase
/// and unpunctuated; Whisper writes normal text).
pub fn normalize(text: &str) -> Vec<String> {
    let cleaned: String = text
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '\'' { c } else { ' ' })
        .collect();
    cleaned.split_whitespace().map(|w| w.trim_matches('\'').to_owned()).filter(|w| !w.is_empty()).collect()
}

/// Word-level edit distance (substitutions + insertions + deletions).
pub fn word_errors(reference: &[String], hypothesis: &[String]) -> usize {
    let mut prev: Vec<usize> = (0..=hypothesis.len()).collect();
    for (i, r) in reference.iter().enumerate() {
        let mut cur = vec![i + 1; hypothesis.len() + 1];
        for (j, h) in hypothesis.iter().enumerate() {
            let sub = prev[j] + usize::from(r != h);
            cur[j + 1] = sub.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[hypothesis.len()]
}

/// Duration from a FLAC STREAMINFO block.
pub fn flac_seconds(data: &[u8]) -> Option<f64> {
    if data.len() < 42 || &data[..4] != b"fLaC" {
        return None;
    }
    let si = &data[8..42];
    let rate = (u32::from(si[10]) << 12) | (u32::from(si[11]) << 4) | (u32::from(si[12]) >> 4);
    let samples = (u64::from(si[13] & 0x0f) << 32)
        | (u64::from(si[14]) << 24)
        | (u64::from(si[15]) << 16)
        | (u64::from(si[16]) << 8)
        | u64::from(si[17]);
    (rate > 0).then(|| samples as f64 / f64::from(rate))
}

pub async fn run(http: &reqwest::Client, target: &Target, progress: &dyn Progress) -> Result<SpeechReport, String> {
    let set: Value = serde_json::from_str(SET).expect("registry/bench/speech.json is validated by tests");
    let items = set["items"].as_array().cloned().unwrap_or_default();
    let mut out = Vec::new();
    for (i, item) in items.iter().enumerate() {
        let file = item["file"].as_str().unwrap_or_default();
        let audio = FILES.iter().find(|(n, _)| *n == file).map(|(_, b)| *b).ok_or(format!("{file} is not compiled in"))?;
        progress.step(&format!("transcribing {}/{}", i + 1, items.len()));
        let form = reqwest::multipart::Form::new()
            .text("model", target.model.clone())
            .text("response_format", "json")
            .part("file", reqwest::multipart::Part::bytes(audio.to_vec()).file_name(file.to_owned()));
        let started = Instant::now();
        let req = async {
            let res = http
                .post(format!("{}/v1/audio/transcriptions", target.base_url))
                .multipart(form)
                .send()
                .await
                .map_err(|e| format!("request failed: {e}"))?;
            let status = res.status();
            let v: Value = res.json().await.map_err(|e| format!("bad response: {e}"))?;
            if status.is_success() {
                Ok(v["text"].as_str().unwrap_or_default().trim().to_owned())
            } else {
                Err(format!("HTTP {status}: {}", v["error"]["message"].as_str().unwrap_or_default()))
            }
        };
        let hypothesis = or_cancel(progress, req).await?;
        let reference = item["text"].as_str().unwrap_or_default().to_owned();
        let (r, h) = (normalize(&reference), normalize(&hypothesis));
        out.push(Utterance {
            id: item["id"].as_str().unwrap_or_default().to_owned(),
            words: r.len(),
            errors: word_errors(&r, &h),
            audio_s: flac_seconds(audio).unwrap_or(0.0),
            seconds: started.elapsed().as_secs_f64(),
            reference,
            hypothesis,
        });
    }
    let words: usize = out.iter().map(|u| u.words).sum();
    let errors: usize = out.iter().map(|u| u.errors).sum();
    let audio: f64 = out.iter().map(|u| u.audio_s).sum();
    let busy: f64 = out.iter().map(|u| u.seconds).sum();
    Ok(SpeechReport {
        wer: errors as f64 / words.max(1) as f64,
        rtf: if audio > 0.0 { busy / audio } else { 0.0 },
        utterances: out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_is_compiled_in() {
        let set: Value = serde_json::from_str(SET).unwrap();
        let items = set["items"].as_array().unwrap();
        assert_eq!(items.len(), 20);
        for i in items {
            let f = i["file"].as_str().unwrap();
            let bytes = FILES.iter().find(|(n, _)| *n == f).unwrap_or_else(|| panic!("{f} missing")).1;
            let secs = flac_seconds(bytes).unwrap();
            assert!((1.0..30.0).contains(&secs), "{f}: {secs}");
        }
    }

    #[test]
    fn word_error_rate() {
        let r = normalize("YET THAT TASK WAS NOT SO EASY AS YOU MAY SUPPOSE");
        assert_eq!(word_errors(&r, &normalize("Yet that task was not so easy, as you may suppose.")), 0);
        assert_eq!(word_errors(&r, &normalize("Yet the task was not easy as you may suppose")), 2, "1 sub + 1 del");
        assert_eq!(word_errors(&normalize("a b"), &normalize("a b c")), 1, "insertion");
        assert_eq!(normalize("Don't—stop!"), ["don't", "stop"]);
    }
}
