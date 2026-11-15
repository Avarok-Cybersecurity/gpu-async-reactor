// SPDX-License-Identifier: MIT OR Apache-2.0

//! The executor: a single-threaded run loop over tasks of one future type (no `dyn`: a caller
//! with several kinds of task uses an enum or one `async` fn). It polls every ready task, and
//! only when none is ready does it hand the thread to the [`Reactor`], which sleeps to the
//! earliest predicted deadline or parks on a completion line. Ready work therefore always runs
//! between waits (FILL: the wait is filled with live work and idles only when nothing is ready),
//! and the idle time is counted ([`RunStats`]).
//!
//! Invariants: a task is polled only after its waker fired (or once at spawn); `run` returns
//! only when every task finished or the reactor reports a stall.

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::task::Wake;
use alloc::vec::Vec;
use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicBool, Ordering};
use core::task::{Context, Poll, Waker};

use crate::core::error::Result;
use crate::core::reactor::{Idle, Reactor, Turn};
use crate::core::time::{Clock, DurNs};

struct Flag(AtomicBool);

impl Wake for Flag {
    fn wake(self: Arc<Self>) {
        self.0.store(true, Ordering::Release);
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.store(true, Ordering::Release);
    }
}

struct Task<F> {
    fut: Option<Pin<Box<F>>>,
    flag: Arc<Flag>,
    waker: Waker,
}

/// What a [`Executor::run`] did: the work it ran and the idle it measured.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RunStats {
    /// Task polls.
    pub polls: u64,
    /// Reactor turns that woke a task without idling.
    pub woke: u64,
    /// Idles spent asleep to a deadline.
    pub slept: u64,
    /// Idles spent parked on a completion line.
    pub parked: u64,
    /// Clock time spent inside the idles, ns (the executor's measured idle).
    pub idle_ns: u64,
}

/// A single-threaded FILL executor over tasks of type `F`.
pub struct Executor<F: Future<Output = ()>> {
    tasks: Vec<Task<F>>,
}

impl<F: Future<Output = ()>> Default for Executor<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Future<Output = ()>> Executor<F> {
    /// No tasks.
    #[must_use]
    pub fn new() -> Self {
        Self { tasks: Vec::new() }
    }

    /// Add a task; it is polled on the next run.
    pub fn spawn(&mut self, fut: F) {
        let flag = Arc::new(Flag(AtomicBool::new(true)));
        let waker = Waker::from(Arc::clone(&flag));
        self.tasks.push(Task {
            fut: Some(Box::pin(fut)),
            flag,
            waker,
        });
    }

    /// Tasks not yet finished.
    #[must_use]
    pub fn unfinished(&self) -> usize {
        self.tasks.iter().filter(|t| t.fut.is_some()).count()
    }

    /// Run every task to completion. When none is ready the thread idles in `idle` as
    /// `reactor` decides; a single park lasts at most `park_bound`.
    ///
    /// # Errors
    /// [`Stalled`](crate::ErrorCode::Stalled): tasks remain and nothing can wake them.
    pub fn run<C: Clock, I: Idle>(
        &mut self,
        reactor: &Reactor<'_, C>,
        idle: &mut I,
        park_bound: DurNs,
    ) -> Result<RunStats> {
        let mut stats = RunStats::default();
        loop {
            reactor.fire();
            let mut progressed = false;
            for t in &mut self.tasks {
                let Some(fut) = t.fut.as_mut() else { continue };
                if !t.flag.0.swap(false, Ordering::Acquire) {
                    continue;
                }
                stats.polls += 1;
                progressed = true;
                if fut
                    .as_mut()
                    .poll(&mut Context::from_waker(&t.waker))
                    .is_ready()
                {
                    t.fut = None;
                }
            }
            let left = self.unfinished();
            if left == 0 {
                self.tasks.clear();
                return Ok(stats);
            }
            if progressed {
                continue;
            }
            let t0 = reactor.clock().monotonic_ns();
            match reactor.turn(idle, park_bound, left)? {
                Turn::Woke => stats.woke += 1,
                Turn::Slept => stats.slept += 1,
                Turn::Parked => stats.parked += 1,
            }
            stats.idle_ns += reactor.clock().monotonic_ns().since(t0).0;
        }
    }
}

/// Yield to the executor once: the task is woken at once and runs again after every other ready
/// task had its turn (cooperative slicing of CPU work between waits).
pub async fn yield_now() {
    let mut yielded = false;
    core::future::poll_fn(|cx| {
        if yielded {
            Poll::Ready(())
        } else {
            yielded = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    })
    .await;
}

/// Run one future to completion on an executor of its own; see [`Executor::run`].
///
/// # Errors
/// As [`Executor::run`].
pub fn block_on<T, F, C, I>(
    fut: F,
    reactor: &Reactor<'_, C>,
    idle: &mut I,
    park_bound: DurNs,
) -> Result<(T, RunStats)>
where
    F: Future<Output = T>,
    C: Clock,
    I: Idle,
{
    let mut out = None;
    let stats = {
        let mut ex = Executor::new();
        ex.spawn(async {
            out = Some(fut.await);
        });
        ex.run(reactor, idle, park_bound)?
    };
    match out {
        Some(v) => Ok((v, stats)),
        None => Err(crate::error!(Stalled, 1)),
    }
}

/// Poll `fut` once with a no-op waker: for hosts whose every future is ready on the first poll
/// (a thread that sleeps and parks inline). A pending poll is a bug in such a host, reported
/// as [`PendedInline`](crate::ErrorCode::PendedInline) naming `site`, never a hang.
///
/// # Errors
/// [`PendedInline`](crate::ErrorCode::PendedInline).
pub fn run_inline<F: Future>(fut: F, site: &str) -> Result<F::Output> {
    let mut fut = core::pin::pin!(fut);
    match fut.as_mut().poll(&mut Context::from_waker(Waker::noop())) {
        Poll::Ready(v) => Ok(v),
        Poll::Pending => Err(crate::error!(PendedInline, site)),
    }
}

#[cfg(test)]
#[path = "fill_executor_tests.rs"]
mod tests;
