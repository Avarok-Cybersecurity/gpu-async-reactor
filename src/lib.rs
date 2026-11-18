// SPDX-License-Identifier: MIT OR Apache-2.0

//! Await GPU work as a **memory descriptor**.
//!
//! A GPU stream stores a sequence number into a word of host-visible memory behind its work;
//! that word is to device completion what a file descriptor is to I/O readiness. This crate
//! makes it a first-class awaitable source and supplies the pieces around it:
//!
//! - [`CompletionLine`] / [`MemoryDescriptor`]: one 64-byte line of eight completion words, and
//!   one word reaching one sequence number.
//! - [`WaitPlan`] / [`WaitBands`]: how a host waits (block, poll, or park on the word with a
//!   status backstop), optionally after sleeping to a predicted completion; typed, parsed once.
//! - [`PointTable`]: the deadline estimator (min of recent exact residuals per readback point,
//!   late wakes censored downward).
//! - [`CompletionPort`] + [`wait`]: the per-backend port and the async driver that runs a plan.
//! - [`Executor`] + [`Reactor`]: a small single-threaded executor whose idle sleeps to the
//!   earliest predicted deadline, then parks on the completion line; ready tasks run between
//!   waits.
//! - [`ExecutorKind`] + [`resolve`]: the selectable executor (`naive`, `gpu-reactor`,
//!   `io-uring`) with a declared fallback chain, a coded warning, and an opt-in strict refusal.
//! - [`Error`] / [`ErrorCode`]: coded errors, one append-only registry.
//!
//! The core is `no_std` + `alloc` and forbids `unsafe`. Vendor code sits behind features in
//! [`backend`] (`cuda`, `hip`, `metal`, `wordwait`) and hosts in [`host`] (`std`, `tokio`,
//! `io-uring`). See the README for measured numbers and the safety model.
#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

#[path = "core/completion_core.rs"]
mod core;

#[path = "backend/vendor_backends.rs"]
pub mod backend;
#[path = "host/host_adapters.rs"]
pub mod host;

pub use crate::core::bands::{BANDS_MAX, Band, Bands, StepRows, WaitBands};
pub use crate::core::declared::{
    Declared, ExecutorCaps, ExecutorKind, Resolution, Strictness, resolve,
};
pub use crate::core::error::{Error, ErrorCode, Result};
pub use crate::core::estimator::{Outcome, PointEstimate, PointTable};
pub use crate::core::executor::{Executor, RunStats, block_on, run_inline, yield_now};
pub use crate::core::line::{CompletionLine, LINE_WORDS, MemoryDescriptor, Slot, reached};
pub use crate::core::manual::{ManualClock, ManualTimer};
pub use crate::core::plan::{
    Capability, DeadlinePlan, FALLBACKS, Fallback, HostCaps, Park, ParkKind, Resolved, WINDOW_MAX,
    WaitPlan,
};
pub use crate::core::reactor::{Alarmed, Idle, Reactor, Ready, Turn};
pub use crate::core::time::{Clock, DurNs, MonoNs, Timer};
pub use crate::core::wait::{CompletionPort, Pace, WaitKind, WaitReport, wait};
