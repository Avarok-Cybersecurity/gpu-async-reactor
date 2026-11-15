// SPDX-License-Identifier: MIT OR Apache-2.0

//! The deadline estimator (ADL, `reports/zav-async-dl.md` §1.2), moved from the
//! `rw/zav-async-dl` prototype into the core unchanged in behaviour.
//!
//! The prediction at a readback point is the minimum of the last `window` *exact* residuals
//! (residual = completion − wait start, observed by a poll that was already running when the
//! work finished, so a sleep's overshoot never feeds back). A wake that finds the work already
//! done is censored: it records `min(wake, prediction) / 2`, so a run of late wakes lowers the
//! prediction geometrically. Early costs CPU, late costs latency: the asymmetry is the point.
//!
//! Invariants: pure (ns in, ns out); a sleep never exceeds the plan's `max_sleep_ns`; a late
//! wake never raises the prediction.

use alloc::collections::BTreeMap;

use crate::core::plan::{DeadlinePlan, WINDOW_MAX};

/// What one wait observed at its readback point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The wait saw the work pending, then done `residual_ns` after the wait started.
    Exact {
        /// Completion minus wait start, ns.
        residual_ns: u64,
    },
    /// The first query after the sleep found the work already done `woke_ns` after the wait
    /// started: the true residual is at most that.
    Late {
        /// Wake minus wait start, ns: an upper bound on the residual.
        woke_ns: u64,
    },
}

/// One readback point's residual estimate (a ring of the last samples).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PointEstimate {
    ring: [u64; WINDOW_MAX],
    len: u8,
    next: u8,
}

impl Default for PointEstimate {
    fn default() -> Self {
        Self::new()
    }
}

impl PointEstimate {
    /// No samples yet: the first wait at a point does not sleep.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            ring: [0; WINDOW_MAX],
            len: 0,
            next: 0,
        }
    }

    /// The predicted residual (min of the kept samples), `None` before the first.
    #[must_use]
    pub fn predicted_ns(&self) -> Option<u64> {
        self.ring[..usize::from(self.len)].iter().copied().min()
    }

    /// How long to sleep from the wait's start; `None` = do not sleep.
    #[must_use]
    pub fn sleep_ns(&self, plan: &DeadlinePlan) -> Option<u64> {
        let s = self.predicted_ns()?.checked_sub(plan.margin_ns)?;
        (s >= plan.min_sleep_ns).then_some(s.min(plan.max_sleep_ns))
    }

    /// Record one wait's outcome.
    pub fn observe(&mut self, plan: &DeadlinePlan, outcome: Outcome) {
        let sample = match outcome {
            Outcome::Exact { residual_ns } => residual_ns,
            Outcome::Late { woke_ns } => woke_ns.min(self.predicted_ns().unwrap_or(woke_ns)) / 2,
        };
        let w = plan
            .window
            .clamp(1, u8::try_from(WINDOW_MAX).unwrap_or(u8::MAX));
        if self.next >= w {
            self.next = 0;
        }
        self.ring[usize::from(self.next)] = sample;
        self.next += 1;
        if self.len < w {
            self.len += 1;
        }
    }
}

/// The estimates of every readback point a host has waited at, keyed by `K` (the
/// host's point identity, e.g. a call site). Runtime measurement, not a decision input.
#[derive(Clone, Debug)]
pub struct PointTable<K> {
    points: BTreeMap<K, PointEstimate>,
}

impl<K: Ord> Default for PointTable<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Ord> PointTable<K> {
    /// No points yet.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            points: BTreeMap::new(),
        }
    }

    /// The sleep the next wait at `key` takes under `plan` (`None`: no sleep).
    #[must_use]
    pub fn sleep_ns(&self, key: &K, plan: &DeadlinePlan) -> Option<u64> {
        self.points.get(key)?.sleep_ns(plan)
    }

    /// Record one wait's outcome at `key`.
    pub fn observe(&mut self, key: K, plan: &DeadlinePlan, outcome: Outcome) {
        self.points.entry(key).or_default().observe(plan, outcome);
    }

    /// How many points have been seen.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// No point seen yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

#[cfg(test)]
#[path = "estimator_tests.rs"]
mod tests;
