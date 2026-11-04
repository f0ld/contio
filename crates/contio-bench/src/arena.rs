//! Side-by-side arena (spec §8.2): blind A/B votes on the same prompt become
//! Elo ratings. Ratings are recomputed from all votes in order, so they're
//! always reproducible from the stored votes.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::store::{Vote, Winner};

pub const START: f64 = 1000.0;
const K: f64 = 32.0;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Rating {
    pub model: String,
    pub elo: f64,
    pub games: u32,
    pub wins: u32,
    pub losses: u32,
    pub ties: u32,
}

/// Elo after every vote; "both bad" counts as a tie between the two.
pub fn ratings(votes: &[Vote]) -> Vec<Rating> {
    let mut r: BTreeMap<String, Rating> = BTreeMap::new();
    let entry = |r: &mut BTreeMap<String, Rating>, m: &str| {
        r.entry(m.to_owned())
            .or_insert_with(|| Rating { model: m.to_owned(), elo: START, games: 0, wins: 0, losses: 0, ties: 0 })
            .elo
    };
    for v in votes.iter().filter(|v| v.model_a != v.model_b) {
        let (ra, rb) = (entry(&mut r, &v.model_a), entry(&mut r, &v.model_b));
        let expected_a = 1.0 / (1.0 + 10f64.powf((rb - ra) / 400.0));
        let score_a = match v.winner {
            Winner::A => 1.0,
            Winner::B => 0.0,
            Winner::Tie | Winner::BothBad => 0.5,
        };
        for (m, delta, score) in [(&v.model_a, score_a - expected_a, score_a), (&v.model_b, expected_a - score_a, 1.0 - score_a)] {
            let x = r.get_mut(m.as_str()).expect("inserted above");
            x.elo += K * delta;
            x.games += 1;
            match score {
                s if s > 0.5 => x.wins += 1,
                s if s < 0.5 => x.losses += 1,
                _ => x.ties += 1,
            }
        }
    }
    let mut out: Vec<Rating> = r.into_values().collect();
    out.sort_by(|a, b| b.elo.total_cmp(&a.elo));
    out
}

/// What a judge model is asked; it sees the answers as "A" and "B".
pub fn judge_prompt(question: &str, first: &str, second: &str) -> String {
    format!(
        "You are judging two answers to the same request. Judge helpfulness, correctness and clarity; \
         ignore length unless it hurts the answer.\n\n## Request\n{question}\n\n## Answer A\n{first}\n\n\
         ## Answer B\n{second}\n\nReply with JSON: {{\"winner\": \"A\" | \"B\" | \"tie\", \"reason\": \"one sentence\"}}"
    )
}

/// Combine the verdicts of both orders (A/B as shown, then swapped): only a
/// consistent preference counts, else it's a tie (position bias).
pub fn combine(as_shown: Winner, swapped: Winner) -> Winner {
    let swapped_back = match swapped {
        Winner::A => Winner::B,
        Winner::B => Winner::A,
        w => w,
    };
    if as_shown == swapped_back { as_shown } else { Winner::Tie }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vote(a: &str, b: &str, w: Winner) -> Vote {
        Vote { id: 0, created_at: 0, prompt: String::new(), model_a: a.into(), model_b: b.into(), winner: w, judge: None }
    }

    #[test]
    fn elo_moves_toward_the_winner() {
        let r = ratings(&[vote("x", "y", Winner::A)]);
        assert_eq!(r[0].model, "x");
        assert!((r[0].elo - 1016.0).abs() < 1e-9 && (r[1].elo - 984.0).abs() < 1e-9);
        let r = ratings(&[vote("x", "y", Winner::Tie), vote("x", "y", Winner::BothBad)]);
        assert!(r.iter().all(|x| (x.elo - START).abs() < 1e-9 && x.ties == 2));
        // upsets move ratings more than expected wins
        let mut v: Vec<Vote> = (0..5).map(|_| vote("x", "y", Winner::A)).collect();
        let before = ratings(&v);
        v.push(vote("x", "y", Winner::B));
        let after = ratings(&v);
        let gain = after.iter().find(|r| r.model == "y").unwrap().elo - before.iter().find(|r| r.model == "y").unwrap().elo;
        assert!(gain > 16.0, "{gain}");
        assert!(ratings(&[vote("x", "x", Winner::A)]).is_empty(), "self-matches ignored");
    }

    #[test]
    fn judge_needs_a_consistent_preference() {
        assert_eq!(combine(Winner::A, Winner::B), Winner::A, "A won in both orders");
        assert_eq!(combine(Winner::A, Winner::A), Winner::Tie, "always picked the first: position bias");
        assert_eq!(combine(Winner::Tie, Winner::Tie), Winner::Tie);
        assert!(judge_prompt("q", "one", "two").contains("## Answer B\ntwo"));
    }
}
