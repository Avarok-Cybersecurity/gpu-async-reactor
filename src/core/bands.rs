// SPDX-License-Identifier: MIT OR Apache-2.0

//! the banded plan: which [`WaitPlan`] a wait
//! runs, chosen by the rows of the current decode step. The band edges are typed fields parsed
//! once from the lever text (ROTI, PLEX); the only per-tick input is the row count, published
//! once per step through [`StepRows`] (BPPT: O(1) per tick, nothing per token).
//!
//! Text: `<plan>` or `<plan>;rows>=<n>:<plan>;…`: the first band starts at 0 rows, every later
//! band names its first row count, the edges strictly increase, at most [`BANDS_MAX`] bands.
//! `<plan>` is [`WaitPlan`]'s text.
//!
//! Invariants: a parsed [`Bands`] is non-empty, starts at 0 rows and its edges strictly
//! increase, so [`Bands::select`] always finds exactly one band.

use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, Ordering};

use crate::core::error::Result;
use crate::error;

use crate::core::plan::{HostCaps, Resolved, WaitPlan};

/// The most bands a plan may state.
pub const BANDS_MAX: usize = 4;

/// One band: `plan` runs from `from_rows` rows up to the next band's edge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Band<P> {
    /// The first row count of the band.
    pub from_rows: u32,
    /// The plan the band runs.
    pub plan: P,
}

/// The bands, in edge order (module invariants).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bands<P> {
    bands: Vec<Band<P>>,
}

/// The requested bands, as the lever states them.
pub type WaitBands = Bands<WaitPlan>;

impl<P> Bands<P> {
    /// The plan of the band `rows` falls in.
    #[must_use]
    pub fn select(&self, rows: u32) -> &P {
        let i = self.bands.partition_point(|b| b.from_rows <= rows);
        // The first edge is 0, so `i >= 1` for every `rows`.
        &self.bands[i - 1].plan
    }

    /// The bands in edge order.
    pub fn iter(&self) -> impl Iterator<Item = &Band<P>> {
        self.bands.iter()
    }

    /// The same edges over `f` of each plan.
    #[must_use]
    pub fn map<Q>(&self, mut f: impl FnMut(&P) -> Q) -> Bands<Q> {
        Bands {
            bands: self
                .bands
                .iter()
                .map(|b| Band {
                    from_rows: b.from_rows,
                    plan: f(&b.plan),
                })
                .collect(),
        }
    }

    fn label_with(&self, f: impl Fn(&P) -> String) -> String {
        let mut out = String::new();
        for (i, b) in self.bands.iter().enumerate() {
            if i > 0 {
                out.push_str(&format!(";rows>={}:", b.from_rows));
            }
            out.push_str(&f(&b.plan));
        }
        out
    }
}

impl WaitBands {
    /// Parse the lever text (module docs).
    ///
    /// # Errors
    /// `InvalidArgument` naming the text and the first thing wrong with it, including every
    /// [`WaitPlan::parse`] refusal of a band's plan.
    pub fn parse(text: &str) -> Result<Self> {
        let refused = |why: String| {
            error!(InvalidPlan, "completion wait plan", format!("{text:?}: {why}"))
        };
        let mut bands: Vec<Band<WaitPlan>> = Vec::new();
        for (i, part) in text.split(';').enumerate() {
            if i == BANDS_MAX {
                return Err(refused(format!("more than {BANDS_MAX} bands")));
            }
            let (from_rows, plan) = if i == 0 {
                (0, part)
            } else {
                let (edge, plan) = part
                    .strip_prefix("rows>=")
                    .and_then(|p| p.split_once(':'))
                    .ok_or_else(|| refused(format!("band {i}: expected rows>=<n>:<plan>")))?;
                let edge: u32 = edge
                    .parse()
                    .map_err(|_| refused(format!("band {i}: {edge:?} is not a u32")))?;
                (edge, plan)
            };
            if let Some(prev) = bands.last()
                && from_rows <= prev.from_rows
            {
                return Err(refused(format!(
                    "band {i}: rows>={from_rows} does not exceed the previous edge {}",
                    prev.from_rows
                )));
            }
            bands.push(Band {
                from_rows,
                plan: WaitPlan::parse(plan)?,
            });
        }
        Ok(Self { bands })
    }

    /// The bands as `parse` reads them (round-trips exactly).
    #[must_use]
    pub fn label(&self) -> String {
        self.label_with(WaitPlan::label)
    }

    /// Each band resolved against `caps` ([`WaitPlan::resolve`]); pure.
    #[must_use]
    pub fn resolve(&self, caps: &HostCaps) -> Bands<Resolved> {
        self.map(|p| p.resolve(caps))
    }
}

impl Bands<Resolved> {
    /// The running bands' label, naming every fallback taken.
    #[must_use]
    pub fn label(&self) -> String {
        self.label_with(Resolved::label)
    }
}

/// The rows of the current decode step: written once per step by the router that
/// launches it, read by every wait of the host that owns the bands. 0 until the first step.
#[derive(Clone, Debug, Default)]
pub struct StepRows(Arc<AtomicU32>);

impl StepRows {
    /// A fresh cell at 0 rows.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The step about to launch has `rows` rows.
    pub fn publish(&self, rows: u32) {
        self.0.store(rows, Ordering::Relaxed);
    }

    /// The rows of the last step published.
    #[must_use]
    pub fn current(&self) -> u32 {
        self.0.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
#[path = "bands_tests.rs"]
mod tests;
