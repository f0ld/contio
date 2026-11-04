//! Retrieval quality of embedding and reranker models (spec §8.2): rank 24
//! documents for 24 paraphrased queries (`registry/bench/retrieval.json`),
//! report recall@1 and mean reciprocal rank.
//!
//! Texts are sent without model-specific prefixes (some embedding models
//! expect e.g. `search_query: `), the way most clients call them.

use std::collections::BTreeMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::perf::{Progress, Target, or_cancel};

const SET: &str = include_str!("../../../registry/bench/retrieval.json");

#[derive(Debug, Clone, Deserialize)]
struct Set {
    documents: BTreeMap<String, String>,
    queries: Vec<Query>,
}

#[derive(Debug, Clone, Deserialize)]
struct Query {
    q: String,
    doc: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Embedding,
    Reranker,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct QueryResult {
    pub query: String,
    pub expected: String,
    /// 1 = the relevant document came first.
    pub rank: usize,
    pub top: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievalReport {
    pub mode: Mode,
    pub recall_at_1: f64,
    pub mrr: f64,
    pub seconds: f64,
    pub queries: Vec<QueryResult>,
}

impl RetrievalReport {
    pub fn metrics(&self) -> BTreeMap<String, f64> {
        BTreeMap::from([
            ("retrieval_recall1".to_owned(), self.recall_at_1),
            ("retrieval_mrr".to_owned(), self.mrr),
            ("retrieval_s".to_owned(), self.seconds),
        ])
    }
}

async fn post(http: &reqwest::Client, url: String, body: Value) -> Result<Value, String> {
    let res = http.post(url).json(&body).send().await.map_err(|e| format!("request failed: {e}"))?;
    let status = res.status();
    let v: Value = res.json().await.map_err(|e| format!("bad response: {e}"))?;
    if !status.is_success() {
        return Err(format!("HTTP {status}: {}", v["error"]["message"].as_str().unwrap_or_default()));
    }
    Ok(v)
}

async fn embed(http: &reqwest::Client, target: &Target, texts: &[&String]) -> Result<Vec<Vec<f64>>, String> {
    let v = post(http, format!("{}/v1/embeddings", target.base_url), json!({ "model": target.model, "input": texts })).await?;
    let mut rows: Vec<(usize, Vec<f64>)> = v["data"]
        .as_array()
        .ok_or("no `data` in the embeddings response")?
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let idx = d["index"].as_u64().map_or(i, |x| x as usize);
            (idx, d["embedding"].as_array().into_iter().flatten().filter_map(Value::as_f64).collect())
        })
        .collect();
    rows.sort_by_key(|r| r.0);
    if rows.len() != texts.len() || rows.iter().any(|r| r.1.is_empty()) {
        return Err(format!("expected {} embeddings, got {}", texts.len(), rows.len()));
    }
    Ok(rows.into_iter().map(|r| r.1).collect())
}

pub fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

/// Document indices, best first.
fn ranking(scores: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    idx
}

pub async fn run(http: &reqwest::Client, target: &Target, mode: Mode, progress: &dyn Progress) -> Result<RetrievalReport, String> {
    let set: Set = serde_json::from_str(SET).expect("registry/bench/retrieval.json is validated by tests");
    let ids: Vec<&String> = set.documents.keys().collect();
    let docs: Vec<&String> = set.documents.values().collect();
    let started = Instant::now();
    let mut results = Vec::new();
    match mode {
        Mode::Embedding => {
            progress.step("embedding the documents");
            let d = or_cancel(progress, embed(http, target, &docs)).await?;
            progress.step("embedding the queries");
            let qs: Vec<&String> = set.queries.iter().map(|q| &q.q).collect();
            let q = or_cancel(progress, embed(http, target, &qs)).await?;
            for (query, qv) in set.queries.iter().zip(&q) {
                let scores: Vec<f64> = d.iter().map(|dv| cosine(qv, dv)).collect();
                results.push(result(query, &ids, &ranking(&scores)));
            }
        }
        Mode::Reranker => {
            for (i, query) in set.queries.iter().enumerate() {
                progress.step(&format!("reranking {}/{}", i + 1, set.queries.len()));
                let body = json!({ "model": target.model, "query": query.q, "documents": docs });
                let v = or_cancel(progress, post(http, format!("{}/v1/rerank", target.base_url), body)).await?;
                let mut scores = vec![f64::NEG_INFINITY; docs.len()];
                for r in v["results"].as_array().ok_or("no `results` in the rerank response")? {
                    if let (Some(i), Some(s)) = (r["index"].as_u64(), r["relevance_score"].as_f64()) {
                        if let Some(slot) = scores.get_mut(i as usize) {
                            *slot = s;
                        }
                    }
                }
                results.push(result(query, &ids, &ranking(&scores)));
            }
        }
    }
    let n = results.len().max(1) as f64;
    Ok(RetrievalReport {
        mode,
        recall_at_1: results.iter().filter(|r| r.rank == 1).count() as f64 / n,
        mrr: results.iter().map(|r| 1.0 / r.rank as f64).sum::<f64>() / n,
        seconds: started.elapsed().as_secs_f64(),
        queries: results,
    })
}

fn result(query: &Query, ids: &[&String], order: &[usize]) -> QueryResult {
    let rank = order.iter().position(|&i| *ids[i] == query.doc).map_or(ids.len(), |p| p + 1);
    QueryResult { query: query.q.clone(), expected: query.doc.clone(), rank, top: ids[order[0]].clone() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_is_consistent() {
        let set: Set = serde_json::from_str(SET).unwrap();
        assert_eq!(set.documents.len(), 24);
        assert_eq!(set.queries.len(), 24);
        let mut used: Vec<&String> = set.queries.iter().map(|q| &q.doc).collect();
        used.sort();
        used.dedup();
        assert_eq!(used.len(), 24, "one query per document");
        assert!(set.queries.iter().all(|q| set.documents.contains_key(&q.doc)));
    }

    #[test]
    fn ranking_math() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-12);
        assert_eq!(cosine(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
        assert_eq!(ranking(&[0.1, 0.9, 0.5]), [1, 2, 0]);
        let ids = [&"a".to_owned(), &"b".to_owned()].map(|s| s);
        let q = Query { q: "x".into(), doc: "b".into() };
        assert_eq!(result(&q, &ids, &[1, 0]).rank, 1);
        assert_eq!(result(&q, &ids, &[0, 1]).rank, 2);
    }
}
