//! Tool-calling suite (spec §8.2): single, multiple-choice, parallel and
//! multi-turn function calls, declining to call when no tool fits, and JSON
//! schema output, sent through the real API path so chat-template bugs show
//! up. Engine labels aren't trusted (M0: an FLM model labelled tool-calling
//! made up a tool result), so this gates the "agent-ready" badge.
//!
//! Cases live in `registry/bench/tool_calls.json`.

use std::collections::BTreeMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::perf::{Progress, Target};

const SUITE_JSON: &str = include_str!("../../../registry/bench/tool_calls.json");

/// Room for thinking models to reason before calling (M0: 512 was too few).
const MAX_TOKENS: u32 = 4096;

/// Agent-ready: at least this overall pass rate…
pub const AGENT_READY_OVERALL: f64 = 0.9;
/// …and no category below this.
pub const AGENT_READY_CATEGORY: f64 = 0.75;

#[derive(Debug, Clone, Deserialize)]
pub struct Suite {
    pub tools: BTreeMap<String, Value>,
    pub cases: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Case {
    pub id: String,
    pub category: String,
    /// Tool names offered; all tools if absent, none if empty (JSON cases:
    /// llama.cpp can't combine a tool grammar with a response schema).
    #[serde(default)]
    pub tools: Option<Vec<String>>,
    pub prompt: String,
    #[serde(default)]
    pub response_format: Option<Value>,
    pub expect: Expect,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Expect {
    /// Exactly these calls, in any order.
    #[serde(default)]
    pub calls: Vec<ExpectedCall>,
    /// The model must answer in text.
    #[serde(default)]
    pub no_call: bool,
    /// Each entry must appear in the answer (case-insensitive); a list
    /// entry means any one of its strings.
    #[serde(default)]
    pub answer_contains: Vec<OneOrMany>,
    /// JSON output whose fields match (and the schema validates).
    #[serde(default)]
    pub json: Option<BTreeMap<String, Matcher>>,
    /// After the calls: their results (in `calls` order) and what comes next.
    #[serde(default)]
    pub then: Option<Box<Then>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Then {
    pub results: Vec<Value>,
    pub expect: Expect,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ExpectedCall {
    pub name: String,
    #[serde(default)]
    pub args: BTreeMap<String, Matcher>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Matcher {
    Contains { contains: String },
    AnyOf(Vec<Value>),
    Exact(Value),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum OneOrMany {
    One(String),
    Many(Vec<String>),
}

impl Suite {
    pub fn builtin() -> Self {
        serde_json::from_str(SUITE_JSON).expect("registry/bench/tool_calls.json is validated by tests")
    }

    fn tool_defs(&self, case: &Case) -> Vec<Value> {
        let names: Vec<&String> = match &case.tools {
            Some(n) => n.iter().collect(),
            None => self.tools.keys().collect(),
        };
        names
            .into_iter()
            .filter_map(|n| self.tools.get(n).map(|f| (n, f)))
            .map(|(n, f)| json!({ "type": "function", "function": { "name": n, "description": f["description"], "parameters": f["parameters"] } }))
            .collect()
    }
}

// --- checking -------------------------------------------------------------------

/// A tool call as the model made it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Call {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// Strings compare case-insensitively and trimmed, numbers by value. Types
/// must match: `"250"` is not `250` (a real tool would reject it).
pub fn value_matches(expected: &Value, actual: &Value) -> bool {
    match (expected, actual) {
        (Value::String(e), Value::String(a)) => e.trim().eq_ignore_ascii_case(a.trim()),
        (Value::Number(e), Value::Number(a)) => match (e.as_f64(), a.as_f64()) {
            (Some(e), Some(a)) => (e - a).abs() < 1e-6,
            _ => false,
        },
        (Value::Array(e), Value::Array(a)) => e.len() == a.len() && e.iter().zip(a).all(|(e, a)| value_matches(e, a)),
        (Value::Object(e), Value::Object(a)) => e.iter().all(|(k, v)| a.get(k).is_some_and(|x| value_matches(v, x))),
        _ => expected == actual,
    }
}

impl Matcher {
    pub fn matches(&self, actual: &Value) -> bool {
        match self {
            Self::Contains { contains } => {
                actual.as_str().is_some_and(|a| a.to_lowercase().contains(&contains.to_lowercase()))
            }
            Self::AnyOf(options) => options.iter().any(|o| value_matches(o, actual)),
            Self::Exact(v) => value_matches(v, actual),
        }
    }
}

/// Why `call` isn't `want`, or `None` if it is.
fn call_mismatch(want: &ExpectedCall, call: &Call) -> Option<String> {
    if want.name != call.name {
        return Some(format!("called {} instead of {}", call.name, want.name));
    }
    let args: Value = match serde_json::from_str(&call.arguments) {
        Ok(v @ Value::Object(_)) => v,
        _ => return Some(format!("{}: arguments are not a JSON object: {}", call.name, call.arguments)),
    };
    for (k, m) in &want.args {
        match args.get(k) {
            None => return Some(format!("{}: missing `{k}`", call.name)),
            Some(v) if !m.matches(v) => return Some(format!("{}: `{k}` = {v}", call.name)),
            Some(_) => {}
        }
    }
    None
}

/// Pair expected calls with actual ones (any order). Returns, per expected
/// call, the index of its actual call.
fn match_calls(want: &[ExpectedCall], got: &[Call]) -> Result<Vec<usize>, String> {
    if want.len() != got.len() {
        let names: Vec<&str> = got.iter().map(|c| c.name.as_str()).collect();
        return Err(format!("expected {} call(s), got {} {names:?}", want.len(), got.len()));
    }
    let mut used = vec![false; got.len()];
    let mut pairs = Vec::new();
    for w in want {
        let mut why = Vec::new();
        let found = got.iter().enumerate().position(|(i, c)| {
            if used[i] {
                return false;
            }
            match call_mismatch(w, c) {
                None => true,
                Some(m) => {
                    why.push(m);
                    false
                }
            }
        });
        match found {
            Some(i) => {
                used[i] = true;
                pairs.push(i);
            }
            None => return Err(format!("no call matches the expected {}: {}", w.name, why.join("; "))),
        }
    }
    Ok(pairs)
}

fn check_answer(expect: &Expect, text: &str) -> Result<(), String> {
    let lower = text.to_lowercase();
    for item in &expect.answer_contains {
        let options: Vec<&String> = match item {
            OneOrMany::One(s) => vec![s],
            OneOrMany::Many(v) => v.iter().collect(),
        };
        if !options.iter().any(|o| lower.contains(&o.to_lowercase())) {
            let snippet: String = text.chars().take(120).collect();
            return Err(format!("answer lacks {options:?}: {snippet:?}"));
        }
    }
    Ok(())
}

fn check_json(fields: &BTreeMap<String, Matcher>, resp: &Value, text: &str) -> Result<(), String> {
    if resp["contio"]["schema_valid"] == false {
        let errors = &resp["contio"]["schema_errors"];
        return Err(format!("schema validation failed: {errors}"));
    }
    let v: Value = serde_json::from_str(text.trim()).map_err(|e| format!("not JSON ({e}): {text:.80}"))?;
    for (k, m) in fields {
        match v.get(k) {
            Some(x) if m.matches(x) => {}
            other => return Err(format!("`{k}` = {}", other.map_or("missing".into(), Value::to_string))),
        }
    }
    Ok(())
}

// --- running --------------------------------------------------------------------

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CaseResult {
    pub id: String,
    pub category: String,
    pub runs: u32,
    pub passed: u32,
    /// One reason per failed run.
    pub failures: Vec<String>,
    /// Mean wall time per run.
    pub mean_s: f64,
    pub completion_tokens: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolsReport {
    pub cases: Vec<CaseResult>,
    /// Category → pass rate.
    pub categories: BTreeMap<String, f64>,
    pub pass_rate: f64,
    pub agent_ready: bool,
    pub reps: u32,
}

impl ToolsReport {
    fn new(cases: Vec<CaseResult>, reps: u32) -> Self {
        let mut by_cat: BTreeMap<String, (u32, u32)> = BTreeMap::new();
        for c in &cases {
            let e = by_cat.entry(c.category.clone()).or_default();
            e.0 += c.passed;
            e.1 += c.runs;
        }
        let rate = |(p, n): (u32, u32)| if n == 0 { 0.0 } else { f64::from(p) / f64::from(n) };
        let total = by_cat.values().fold((0, 0), |a, &(p, n)| (a.0 + p, a.1 + n));
        let categories: BTreeMap<String, f64> = by_cat.into_iter().map(|(k, v)| (k, rate(v))).collect();
        let pass_rate = rate(total);
        let agent_ready =
            pass_rate >= AGENT_READY_OVERALL && categories.values().all(|&r| r >= AGENT_READY_CATEGORY);
        Self { cases, categories, pass_rate, agent_ready, reps }
    }

    pub fn metrics(&self) -> BTreeMap<String, f64> {
        let mut m: BTreeMap<String, f64> = self.categories.iter().map(|(k, v)| (format!("tools_{k}"), *v)).collect();
        m.insert("tools_pass_rate".into(), self.pass_rate);
        m.insert("agent_ready".into(), if self.agent_ready { 1.0 } else { 0.0 });
        m
    }
}

struct Reply {
    resp: Value,
    message: Value,
    calls: Vec<Call>,
    text: String,
    completion_tokens: u64,
}

async fn complete(
    http: &reqwest::Client,
    target: &Target,
    messages: &[Value],
    tools: &[Value],
    response_format: Option<&Value>,
) -> Result<Reply, String> {
    // the model's own sampling: greedy decoding sends thinking models into
    // repetition loops (Qwen's model cards warn against it; seen here too),
    // and agents don't use it. Repetitions turn the randomness into a pass rate.
    let mut body = json!({ "model": target.model, "messages": messages, "max_tokens": MAX_TOKENS });
    if !tools.is_empty() {
        body["tools"] = json!(tools);
    }
    if let Some(f) = response_format {
        body["response_format"] = f.clone();
    }
    let res = http
        .post(format!("{}/v1/chat/completions", target.base_url))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    let status = res.status();
    let resp: Value = res.json().await.map_err(|e| format!("bad response: {e}"))?;
    if !status.is_success() {
        return Err(format!("HTTP {status}: {}", resp["error"]["message"].as_str().unwrap_or_default()));
    }
    let message = resp["choices"][0]["message"].clone();
    let calls = message["tool_calls"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|c| Call {
            id: c["id"].as_str().unwrap_or_default().to_owned(),
            name: c["function"]["name"].as_str().unwrap_or_default().to_owned(),
            arguments: match &c["function"]["arguments"] {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            },
        })
        .collect();
    Ok(Reply {
        text: message["content"].as_str().unwrap_or_default().to_owned(),
        completion_tokens: resp["usage"]["completion_tokens"].as_u64().unwrap_or(0),
        resp,
        message,
        calls,
    })
}

/// One run of one case; `Err` explains the failure.
async fn run_case(http: &reqwest::Client, target: &Target, suite: &Suite, case: &Case, tokens: &mut u64) -> Result<(), String> {
    let tools = suite.tool_defs(case);
    let mut messages = vec![json!({ "role": "user", "content": case.prompt })];
    let mut expect = &case.expect;
    for step in 1.. {
        let reply = complete(http, target, &messages, &tools, case.response_format.as_ref()).await?;
        *tokens += reply.completion_tokens;
        let at = |e: String| if step > 1 { format!("turn {step}: {e}") } else { e };
        if reply.resp["choices"][0]["finish_reason"] == "length" {
            return Err(at(format!("hit the {MAX_TOKENS}-token limit (thinking didn't finish)")));
        }
        if expect.no_call && !reply.calls.is_empty() {
            let names: Vec<&str> = reply.calls.iter().map(|c| c.name.as_str()).collect();
            return Err(at(format!("called {names:?} but should have answered")));
        }
        if reply.calls.is_empty() && !expect.calls.is_empty() {
            let snippet: String = reply.text.chars().take(100).collect();
            return Err(at(format!("no tool call; answered {snippet:?}")));
        }
        let pairs = match_calls(&expect.calls, &reply.calls).map_err(at)?;
        if let Some(fields) = &expect.json {
            check_json(fields, &reply.resp, &reply.text).map_err(at)?;
        }
        check_answer(expect, &reply.text).map_err(at)?;
        let Some(then) = &expect.then else { return Ok(()) };

        // feed the results back, paired with the calls they answer
        let mut assistant = reply.message.clone();
        assistant.as_object_mut().map(|o| o.remove("reasoning_content"));
        messages.push(assistant);
        for (want_idx, &got_idx) in pairs.iter().enumerate() {
            let call = &reply.calls[got_idx];
            let result = then.results.get(want_idx).cloned().unwrap_or(Value::Null);
            messages.push(json!({ "role": "tool", "tool_call_id": call.id, "name": call.name, "content": result.to_string() }));
        }
        expect = &then.expect;
    }
    unreachable!("the loop returns")
}

pub async fn run(
    http: &reqwest::Client,
    target: &Target,
    suite: &Suite,
    reps: u32,
    progress: &dyn Progress,
) -> Result<ToolsReport, String> {
    let reps = reps.max(1);
    let mut results = Vec::new();
    for (i, case) in suite.cases.iter().enumerate() {
        let mut r = CaseResult { id: case.id.clone(), category: case.category.clone(), ..Default::default() };
        let mut elapsed = 0.0;
        for rep in 0..reps {
            if progress.cancelled() {
                return Err("cancelled".into());
            }
            progress.step(&format!("tool calling {}/{}: {} (run {}/{reps})", i + 1, suite.cases.len(), case.id, rep + 1));
            let started = Instant::now();
            let outcome = run_case(http, target, suite, case, &mut r.completion_tokens).await;
            elapsed += started.elapsed().as_secs_f64();
            r.runs += 1;
            match outcome {
                Ok(()) => r.passed += 1,
                Err(e) => r.failures.push(e),
            }
        }
        r.mean_s = elapsed / f64::from(reps);
        results.push(r);
    }
    Ok(ToolsReport::new(results, reps))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str, args: Value) -> Call {
        Call { id: "c".into(), name: name.into(), arguments: args.to_string() }
    }

    #[test]
    fn builtin_suite_is_consistent() {
        let s = Suite::builtin();
        assert!(s.cases.len() >= 20);
        let mut ids = std::collections::HashSet::new();
        for c in &s.cases {
            assert!(ids.insert(&c.id), "duplicate {}", c.id);
            for t in c.tools.iter().flatten() {
                assert!(s.tools.contains_key(t), "{}: unknown tool {t}", c.id);
            }
            let mut e = Some(&c.expect);
            while let Some(x) = e {
                for w in &x.calls {
                    assert!(s.tools.contains_key(&w.name), "{}: expects unknown tool {}", c.id, w.name);
                    let props = &s.tools[&w.name]["parameters"]["properties"];
                    for k in w.args.keys() {
                        assert!(props.get(k).is_some(), "{}: {} has no parameter {k}", c.id, w.name);
                    }
                }
                if let Some(t) = &x.then {
                    assert_eq!(t.results.len(), x.calls.len(), "{}: one result per call", c.id);
                }
                e = x.then.as_ref().map(|t| &t.expect);
            }
        }
        let cats: std::collections::BTreeSet<&str> = s.cases.iter().map(|c| c.category.as_str()).collect();
        assert_eq!(cats.into_iter().collect::<Vec<_>>(), ["irrelevance", "json", "multi_turn", "multiple", "parallel", "simple"]);
    }

    #[test]
    fn matching_rules() {
        assert!(value_matches(&json!("Berlin"), &json!(" berlin ")));
        assert!(value_matches(&json!(250), &json!(250.0)));
        assert!(!value_matches(&json!(250), &json!("250")), "types must match");
        assert!(!value_matches(&json!(false), &json!("false")));
        let m: Matcher = serde_json::from_value(json!({"contains": "Austin"})).unwrap();
        assert!(m.matches(&json!("austin, TX")) && !m.matches(&json!("Dallas")));
        let any: Matcher = serde_json::from_value(json!(["Austin", "Austin, TX"])).unwrap();
        assert!(any.matches(&json!("AUSTIN, tx")));
    }

    #[test]
    fn parallel_calls_in_any_order() {
        let want: Vec<ExpectedCall> = serde_json::from_value(json!([
            {"name": "get_weather", "args": {"city": "Paris"}},
            {"name": "get_weather", "args": {"city": "Rome"}}
        ]))
        .unwrap();
        let got = [call("get_weather", json!({"city": "Rome"})), call("get_weather", json!({"city": "Paris", "unit": "celsius"}))];
        assert_eq!(match_calls(&want, &got).unwrap(), [1, 0]);
        let err = match_calls(&want, &got[..1]).unwrap_err();
        assert!(err.contains("expected 2 call(s), got 1"), "{err}");
        let wrong = [call("get_weather", json!({"city": "Rome"})), call("get_weather", json!({"city": "Oslo"}))];
        assert!(match_calls(&want, &wrong).unwrap_err().contains("Oslo"));
        let bad = [call("get_weather", json!("Paris")), call("get_weather", json!({"city": "Rome"}))];
        assert!(match_calls(&want, &bad).unwrap_err().contains("not a JSON object"));
    }

    #[test]
    fn answers_and_json() {
        let e: Expect = serde_json::from_value(json!({"answer_contains": ["A-1042", ["yes", "umbrella"]]})).unwrap();
        assert!(check_answer(&e, "Order A-1042 is open; bring an umbrella").is_ok());
        assert!(check_answer(&e, "Order A-1042 is open").is_err());
        let fields: BTreeMap<String, Matcher> = serde_json::from_value(json!({"age": 34})).unwrap();
        assert!(check_json(&fields, &json!({}), r#"{"name":"Maria","age":34}"#).is_ok());
        assert!(check_json(&fields, &json!({}), r#"{"age":"34"}"#).is_err());
        assert!(check_json(&fields, &json!({"contio": {"schema_valid": false}}), r#"{"age":34}"#).is_err());
    }

    #[test]
    fn agent_ready_needs_every_category() {
        let case = |cat: &str, passed| CaseResult { category: cat.into(), runs: 10, passed, ..Default::default() };
        let r = ToolsReport::new(vec![case("simple", 10), case("parallel", 10), case("json", 10)], 1);
        assert!(r.agent_ready && r.pass_rate == 1.0);
        let r = ToolsReport::new(vec![case("simple", 10); 9].into_iter().chain([case("parallel", 5)]).collect(), 1);
        assert!(r.pass_rate >= 0.9 && !r.agent_ready, "parallel at 50% blocks the badge");
        assert_eq!(r.metrics()["agent_ready"], 0.0);
    }
}
