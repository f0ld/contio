//! Does a model fit in its memory pool? (spec §4.3)

use contio_core::model::{KvGeometry, estimate_memory, estimate_whisper_memory};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Leaves room for other models or a bigger context.
    Fits,
    /// Fits, but needs most of the pool.
    Tight,
    TooBig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fit {
    pub verdict: Verdict,
    pub needed_bytes: u64,
    pub budget_bytes: u64,
}

/// Above this share of the pool a fit counts as tight.
const TIGHT_PERCENT: u64 = 85;

pub fn fit(needed_bytes: u64, budget_bytes: u64) -> Fit {
    let verdict = if needed_bytes > budget_bytes {
        Verdict::TooBig
    } else if needed_bytes * 100 > budget_bytes * TIGHT_PERCENT {
        Verdict::Tight
    } else {
        Verdict::Fits
    };
    Fit { verdict, needed_bytes, budget_bytes }
}

/// A GGUF model (+ projector) at `context` against a pool budget.
pub fn fit_gguf(weights: u64, kv: Option<&KvGeometry>, context: u32, budget_bytes: u64) -> Fit {
    fit(estimate_memory(weights, kv, Some(context)), budget_bytes)
}

/// A whisper.cpp model against a pool budget.
pub fn fit_whisper(weights: u64, budget_bytes: u64) -> Fit {
    fit(estimate_whisper_memory(weights), budget_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    #[test]
    fn verdicts() {
        assert_eq!(fit(10 * GIB, 57 * GIB).verdict, Verdict::Fits);
        assert_eq!(fit(50 * GIB, 57 * GIB).verdict, Verdict::Tight);
        assert_eq!(fit(58 * GIB, 57 * GIB).verdict, Verdict::TooBig);
    }

    #[test]
    fn gpt_oss_120b_does_not_fit_a_64_gib_carve_out() {
        // 63.39 GB MXFP4, 90 % of 64 GiB usable
        let kv = KvGeometry { full_bytes_per_token: 36 << 10, swa_bytes_per_token: 36 << 10, swa_window: 128 };
        let f = fit_gguf(63_390_000_000, Some(&kv), 32768, 64 * GIB * 9 / 10);
        assert_eq!(f.verdict, Verdict::TooBig);
    }
}
