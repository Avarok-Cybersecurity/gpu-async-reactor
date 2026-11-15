// SPDX-License-Identifier: MIT OR Apache-2.0

//! Declared settings: a setting whose value may be unavailable on a host resolves once, at
//! startup, through a DECLARED fallback chain (never a guess), and reports what it did:
//! - available: the requested value runs;
//! - unavailable, not strict: a coded WARNING names the requested value, why it is unavailable
//!   and the fallback that runs;
//! - unavailable, strict: a coded refusal.
//!
//! [`Declared`] is the shape any such setting implements; [`ExecutorKind`] is the first.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::core::error::{Error, ErrorCode, Result};

/// Whether a setting may fall back when its value is unavailable.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Strictness {
    /// Fall back along the declared chain, with a coded warning.
    #[default]
    Fallback,
    /// Refuse: no fallback is acceptable.
    Strict,
}

/// A setting value with a declared fallback chain.
pub trait Declared: Copy + Eq + 'static {
    /// What the host offers, probed once by the host (never by the core).
    type Caps;

    /// The value's spelling.
    fn name(self) -> &'static str;

    /// Why `self` cannot run on `caps`, or `None` when it can.
    fn lacks(self, caps: &Self::Caps) -> Option<String>;

    /// The declared fallback of `self` (`None`: the end of the chain, which must run anywhere).
    fn fallback(self) -> Option<Self>;

    /// The code of the warning a fallback emits: (requested, why, running).
    const FALLBACK_CODE: ErrorCode;

    /// The code of the refusal a strict setting emits: (requested, why).
    const REFUSED_CODE: ErrorCode;
}

/// A resolved setting: what was asked, what runs, and the warning when they differ.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resolution<K> {
    /// The value as requested.
    pub requested: K,
    /// The value that runs.
    pub running: K,
    /// The coded warning when `running != requested`.
    pub warning: Option<Error>,
}

impl<K: Declared> Resolution<K> {
    /// `running`, naming the fallback when one was taken (boot report, metrics, bench record).
    #[must_use]
    pub fn label(&self) -> String {
        if self.running == self.requested {
            String::from(self.running.name())
        } else {
            format!(
                "{} (fallback from {})",
                self.running.name(),
                self.requested.name()
            )
        }
    }
}

/// Resolve `requested` against `caps`: walk the declared chain while the value cannot run.
///
/// # Errors
/// The setting's refusal code when `strictness` is [`Strictness::Strict`] and `requested`
/// cannot run.
pub fn resolve<K: Declared>(
    requested: K,
    strictness: Strictness,
    caps: &K::Caps,
) -> Result<Resolution<K>> {
    let mut running = requested;
    let mut why: Vec<String> = Vec::new();
    while let Some(reason) = running.lacks(caps) {
        if strictness == Strictness::Strict {
            return Err(Error::new(
                K::REFUSED_CODE,
                alloc::vec![String::from(requested.name()), reason],
            ));
        }
        why.push(format!("{}: {reason}", running.name()));
        match running.fallback() {
            Some(next) => running = next,
            None => break,
        }
    }
    let warning = (running != requested).then(|| {
        Error::new(
            K::FALLBACK_CODE,
            alloc::vec![
                String::from(requested.name()),
                why.join("; "),
                String::from(running.name()),
            ],
        )
    });
    Ok(Resolution {
        requested,
        running,
        warning,
    })
}

/// How a waiting thread runs a wait.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExecutorKind {
    /// The wait runs inline on the waiting thread: a blocking sleep, then a blocking park or
    /// poll. Other work proceeds only on other threads or through issue-ahead.
    Naive,
    /// The wait is a task of this crate's [`Executor`](crate::Executor): its idle sleeps to the
    /// earliest predicted deadline, then parks on the completion line; ready tasks run between.
    GpuReactor,
    /// As `GpuReactor`, with the deadline sleep an `IORING_OP_TIMEOUT` (ns) on an io_uring.
    IoUring,
}

/// What a host offers the executors, probed once.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutorCaps {
    /// `Ok` when a user-mode word wait runs (aarch64 `wfe`, x86 `mwaitx`/`umwait`), else why not.
    pub word_wait: core::result::Result<(), String>,
    /// `Ok` when an io_uring can be set up, else why not.
    pub io_uring: core::result::Result<(), String>,
}

impl ExecutorKind {
    /// Every kind, in declared order.
    pub const ALL: [ExecutorKind; 3] = [Self::Naive, Self::GpuReactor, Self::IoUring];

    /// Parse a spelling (`naive`, `gpu-reactor`, `io-uring`).
    ///
    /// # Errors
    /// [`InvalidExecutor`](crate::ErrorCode::InvalidExecutor).
    pub fn parse(text: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|k| k.name() == text)
            .ok_or_else(|| {
                let names: Vec<&str> = Self::ALL.iter().map(|k| k.name()).collect();
                crate::error!(InvalidExecutor, text, names.join(", "))
            })
    }
}

impl Declared for ExecutorKind {
    type Caps = ExecutorCaps;

    fn name(self) -> &'static str {
        match self {
            Self::Naive => "naive",
            Self::GpuReactor => "gpu-reactor",
            Self::IoUring => "io-uring",
        }
    }

    fn lacks(self, caps: &ExecutorCaps) -> Option<String> {
        match self {
            Self::Naive => None,
            Self::GpuReactor => caps.word_wait.as_ref().err().cloned(),
            Self::IoUring => caps
                .io_uring
                .as_ref()
                .err()
                .or(caps.word_wait.as_ref().err())
                .cloned(),
        }
    }

    fn fallback(self) -> Option<Self> {
        match self {
            Self::IoUring => Some(Self::GpuReactor),
            Self::GpuReactor => Some(Self::Naive),
            Self::Naive => None,
        }
    }

    const FALLBACK_CODE: ErrorCode = ErrorCode::ExecutorFallback;
    const REFUSED_CODE: ErrorCode = ErrorCode::ExecutorRefused;
}

#[cfg(test)]
#[path = "declared_tests.rs"]
mod tests;
