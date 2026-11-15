// SPDX-License-Identifier: MIT OR Apache-2.0

//! The user-mode wait on a [`CompletionLine`] that another agent (a GPU's store over a coherent
//! fabric, or another thread) writes. The CPU arms a monitor on the line, re-checks the
//! condition, and sleeps in hardware until the line is written or a bound passes:
//!
//! | target | arm | wait | bound |
//! |---|---|---|---|
//! | aarch64 Linux | `ldxr` (exclusive monitor; granule = the 64-byte line) | `wfet` (FEAT_WFxT) or `wfe` | counter deadline, or the kernel's ~100 us event stream |
//! | x86-64 Linux, Intel WAITPKG | `umonitor` | `umwait` (C0.2) | TSC deadline |
//! | x86-64 Linux, AMD MONITORX | `monitorx` | `mwaitx` with the timer enabled | TSC-rate timeout |
//!
//! Measured on GB10 (aarch64): the wake follows the store by 0.3-0.4 us; a 10 ms wait takes
//! ~95 parks at a 100 us bound (README, benchmarks). The x86 paths are compile-checked only.
//!
//! Invariants: every public item is safe; a park only reads the line through its atomics; a park
//! is bounded; [`WordWait::probe`] refuses a CPU that lacks the instructions.

use core::time::Duration;

use crate::core::line::CompletionLine;

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
#[path = "wordwait_arm64.rs"]
mod arch;

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[path = "wordwait_x86.rs"]
mod arch;

#[cfg(not(any(
    all(target_arch = "aarch64", target_os = "linux"),
    all(target_arch = "x86_64", target_os = "linux")
)))]
mod arch {
    use super::{CompletionLine, Duration};

    /// No user-mode word wait on this target.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct WordWait {
        _none: (),
    }

    impl WordWait {
        pub(super) fn probe_arch() -> Result<Self, &'static str> {
            Err("no user-mode word wait on this target (aarch64 or x86-64 Linux only)")
        }

        pub(super) fn park_arch(
            &self,
            line: &CompletionLine,
            ready: impl Fn(&CompletionLine) -> bool,
            _bound: Duration,
        ) -> bool {
            ready(line)
        }
    }
}

pub use arch::WordWait;
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
pub use arch::Flavor;

impl WordWait {
    /// This CPU's word wait, or why there is none. Probe once and keep the value.
    ///
    /// # Errors
    /// The CPU or target lacks a user-mode monitor/wait.
    pub fn probe() -> Result<Self, &'static str> {
        Self::probe_arch()
    }

    /// Park on `line` until `ready(line)` holds, a store to the line wakes the core, or `bound`
    /// passes; returns `ready(line)`. Any store to the line wakes the park, so one waiter can
    /// watch every slot; a single-word wait is
    /// `ready = |l| l.words[slot].load(Acquire) == expect`. May return `false` early.
    #[must_use]
    pub fn park(
        &self,
        line: &CompletionLine,
        ready: impl Fn(&CompletionLine) -> bool,
        bound: Duration,
    ) -> bool {
        self.park_arch(line, ready, bound)
    }
}
