// SPDX-License-Identifier: MIT OR Apache-2.0

//! The reactor: what the tasks of one executor are waiting on (memory descriptors and timers),
//! and the one decision the executor makes when no task is ready ([`Reactor::turn`]):
//! 1. wake every task whose descriptor completed or whose deadline passed;
//! 2. otherwise, with only timers pending, sleep to the earliest one (a predicted completion);
//! 3. otherwise park on the completion line of the earliest-registered descriptor, bounded by
//!    the park bound and the earliest deadline; any store to the line wakes the park, so every
//!    descriptor of that line is watched by the one park.
//!
//! Invariants: the reactor never blocks itself (the [`Idle`] host does); a descriptor future
//! resolves `true` only after an `Acquire` load saw its word reach the target.

use alloc::vec::Vec;
use core::cell::RefCell;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, Waker};

use crate::core::error::Result;
use crate::core::line::{CompletionLine, MemoryDescriptor, reached};
use crate::core::time::{Clock, DurNs, MonoNs, Timer};
use crate::error;

/// How an executor's thread idles when no task is ready: the host's blocking primitives. Only
/// the executor calls these, and only between task polls.
pub trait Idle {
    /// Sleep until the clock reads `at` (an early return is allowed).
    fn sleep_until(&mut self, at: MonoNs);

    /// Park until a store to `line` may have made `ready(line)` true, or `bound` passes; return
    /// `ready(line)`. An early `false` (a spurious wake) is allowed.
    fn park_line<R: Fn(&CompletionLine) -> bool>(
        &mut self,
        line: &CompletionLine,
        ready: R,
        bound: DurNs,
    ) -> bool;
}

struct Source<'l> {
    id: u64,
    desc: MemoryDescriptor<'l>,
    until: Option<MonoNs>,
    waker: Waker,
}

struct Alarm {
    id: u64,
    at: MonoNs,
    waker: Waker,
}

#[derive(Default)]
struct Table<'l> {
    next_id: u64,
    sources: Vec<Source<'l>>,
    alarms: Vec<Alarm>,
}

/// What one [`Reactor::turn`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Turn {
    /// At least one task was woken.
    Woke,
    /// The thread slept to a deadline.
    Slept,
    /// The thread parked on a completion line.
    Parked,
}

/// The reactor of one executor: registrations made by the futures it hands out.
pub struct Reactor<'l, C: Clock> {
    clock: C,
    table: RefCell<Table<'l>>,
}

impl<'l, C: Clock> Reactor<'l, C> {
    /// An empty reactor reading `clock`.
    #[must_use]
    pub fn new(clock: C) -> Self {
        Self {
            clock,
            table: RefCell::new(Table::default()),
        }
    }

    /// The clock the reactor reads.
    #[must_use]
    pub fn clock(&self) -> &C {
        &self.clock
    }

    /// A future that resolves `true` once `desc` completes, or `false` once the clock reads
    /// `until` first (no bound with `None`).
    #[must_use]
    pub fn ready(&self, desc: MemoryDescriptor<'l>, until: Option<MonoNs>) -> Ready<'_, 'l, C> {
        Ready {
            reactor: self,
            desc,
            until,
            id: None,
        }
    }

    /// Registered descriptors and timers (for a host's counters and tests).
    #[must_use]
    pub fn pending(&self) -> (usize, usize) {
        let t = self.table.borrow();
        (t.sources.len(), t.alarms.len())
    }

    /// The executor's step when no task is ready (module docs).
    ///
    /// # Errors
    /// [`Stalled`](crate::ErrorCode::Stalled) when nothing is registered: no store and no timer
    /// can ever wake the executor.
    pub fn turn<I: Idle>(&self, idle: &mut I, park_bound: DurNs, tasks_left: usize) -> Result<Turn> {
        if self.fire() {
            return Ok(Turn::Woke);
        }
        let now = self.clock.monotonic_ns();
        let (deadline, line) = {
            let t = self.table.borrow();
            let deadline = t
                .alarms
                .iter()
                .map(|a| a.at)
                .chain(t.sources.iter().filter_map(|s| s.until))
                .min();
            (deadline, t.sources.first().map(|s| s.desc.line()))
        };
        match (line, deadline) {
            (None, None) => Err(error!(Stalled, tasks_left)),
            (None, Some(at)) => {
                idle.sleep_until(at);
                Ok(Turn::Slept)
            }
            (Some(line), deadline) => {
                let bound = deadline.map_or(park_bound, |d| park_bound.min(d.since(now)));
                let watched: Vec<_> = {
                    let t = self.table.borrow();
                    t.sources
                        .iter()
                        .filter(|s| core::ptr::eq(s.desc.line(), line))
                        .map(|s| (s.desc.slot(), s.desc.target()))
                        .collect()
                };
                idle.park_line(
                    line,
                    |l| watched.iter().any(|&(slot, target)| reached(l, slot, target)),
                    bound,
                );
                Ok(Turn::Parked)
            }
        }
    }

    /// Wake every task whose descriptor completed or whose deadline passed, without idling;
    /// `true` if any was woken. The executor calls this between task polls, so a completion
    /// becomes ready work as soon as it lands, not only when the thread runs out of work.
    pub fn fire(&self) -> bool {
        let now = self.clock.monotonic_ns();
        let due = {
            let mut t = self.table.borrow_mut();
            let mut due = Vec::new();
            t.sources.retain_mut(|s| {
                let fire = s.desc.is_complete() || s.until.is_some_and(|u| now >= u);
                if fire {
                    due.push(s.waker.clone());
                }
                !fire
            });
            t.alarms.retain(|a| {
                let fire = now >= a.at;
                if fire {
                    due.push(a.waker.clone());
                }
                !fire
            });
            due
        };
        let any = !due.is_empty();
        for w in due {
            w.wake();
        }
        any
    }

    fn register_source(&self, id: &mut Option<u64>, desc: MemoryDescriptor<'l>, until: Option<MonoNs>, waker: &Waker) {
        let mut t = self.table.borrow_mut();
        if let Some(i) = *id
            && let Some(s) = t.sources.iter_mut().find(|s| s.id == i)
        {
            s.waker.clone_from(waker);
            return;
        }
        t.next_id += 1;
        let i = t.next_id;
        *id = Some(i);
        t.sources.push(Source {
            id: i,
            desc,
            until,
            waker: waker.clone(),
        });
    }

    fn register_alarm(&self, id: &mut Option<u64>, at: MonoNs, waker: &Waker) {
        let mut t = self.table.borrow_mut();
        if let Some(i) = *id
            && let Some(a) = t.alarms.iter_mut().find(|a| a.id == i)
        {
            a.waker.clone_from(waker);
            return;
        }
        t.next_id += 1;
        let i = t.next_id;
        *id = Some(i);
        t.alarms.push(Alarm {
            id: i,
            at,
            waker: waker.clone(),
        });
    }

    fn forget(&self, id: Option<u64>) {
        if let Some(i) = id {
            let mut t = self.table.borrow_mut();
            t.sources.retain(|s| s.id != i);
            t.alarms.retain(|a| a.id != i);
        }
    }
}

/// The future of [`Reactor::ready`].
pub struct Ready<'r, 'l, C: Clock> {
    reactor: &'r Reactor<'l, C>,
    desc: MemoryDescriptor<'l>,
    until: Option<MonoNs>,
    id: Option<u64>,
}

impl<C: Clock> Future for Ready<'_, '_, C> {
    type Output = bool;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<bool> {
        let this = self.get_mut();
        if this.desc.is_complete() {
            this.reactor.forget(this.id.take());
            return Poll::Ready(true);
        }
        if this
            .until
            .is_some_and(|u| this.reactor.clock.monotonic_ns() >= u)
        {
            this.reactor.forget(this.id.take());
            return Poll::Ready(false);
        }
        this.reactor
            .register_source(&mut this.id, this.desc, this.until, cx.waker());
        Poll::Pending
    }
}

impl<C: Clock> Drop for Ready<'_, '_, C> {
    fn drop(&mut self) {
        self.reactor.forget(self.id.take());
    }
}

/// The reactor's sleep future (its [`Timer`]).
pub struct Alarmed<'r, 'l, C: Clock> {
    reactor: &'r Reactor<'l, C>,
    at: MonoNs,
    id: Option<u64>,
}

impl<C: Clock> Future for Alarmed<'_, '_, C> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
        if this.reactor.clock.monotonic_ns() >= this.at {
            this.reactor.forget(this.id.take());
            return Poll::Ready(());
        }
        this.reactor.register_alarm(&mut this.id, this.at, cx.waker());
        Poll::Pending
    }
}

impl<C: Clock> Drop for Alarmed<'_, '_, C> {
    fn drop(&mut self) {
        self.reactor.forget(self.id.take());
    }
}

impl<'l, C: Clock> Reactor<'l, C> {
    /// A future that resolves once the clock reads `at`; the executor sleeps to the earliest one.
    #[must_use]
    pub fn alarm(&self, at: MonoNs) -> Alarmed<'_, 'l, C> {
        Alarmed {
            reactor: self,
            at,
            id: None,
        }
    }
}

impl<C: Clock> Timer for Reactor<'_, C> {
    async fn sleep_until(&self, at: MonoNs) {
        self.alarm(at).await;
    }
}

impl<C: Clock> Clock for Reactor<'_, C> {
    fn monotonic_ns(&self) -> MonoNs {
        self.clock.monotonic_ns()
    }
}

#[cfg(test)]
#[path = "reactor_tests.rs"]
pub(crate) mod tests;
