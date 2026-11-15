// SPDX-License-Identifier: MIT OR Apache-2.0

//! A virtual clock: the host for tests, simulation and replay. Time moves only when the owner
//! moves it, and every [`ManualClock::timer`] sleep fires when it does, so a wait or an executor
//! run is reproducible to the nanosecond on any machine.

use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, Waker};

use crate::core::time::{Clock, DurNs, MonoNs, Timer};

#[derive(Default)]
struct State {
    mono: MonoNs,
    next_id: u64,
    timers: BTreeMap<(MonoNs, u64), Waker>,
}

/// A clock that moves only when told to; clones share it.
#[derive(Clone, Default)]
pub struct ManualClock {
    state: Rc<RefCell<State>>,
}

/// The [`Timer`] of one [`ManualClock`]; it fires only when that clock moves.
#[derive(Clone)]
pub struct ManualTimer {
    state: Rc<RefCell<State>>,
}

impl ManualClock {
    /// A clock at 0 ns with no timers.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Another handle on the same clock.
    #[must_use]
    pub fn handle(&self) -> Self {
        self.clone()
    }

    /// The timer that fires as this clock moves.
    #[must_use]
    pub fn timer(&self) -> ManualTimer {
        ManualTimer {
            state: Rc::clone(&self.state),
        }
    }

    /// Move time forward by `step`, waking every due timer.
    pub fn advance(&self, step: DurNs) {
        let to = self.monotonic_ns().after(step);
        self.set_monotonic(to);
    }

    /// Set time to `at` (never backwards: an earlier `at` leaves the clock where it is), waking
    /// every due timer, earliest first.
    pub fn set_monotonic(&self, at: MonoNs) {
        let due: Vec<Waker> = {
            let mut s = self.state.borrow_mut();
            if at > s.mono {
                s.mono = at;
            }
            let now = s.mono;
            let pending = match now.0.checked_add(1) {
                Some(next) => s.timers.split_off(&(MonoNs(next), 0)),
                None => BTreeMap::new(),
            };
            core::mem::replace(&mut s.timers, pending)
                .into_values()
                .collect()
        };
        for w in due {
            w.wake();
        }
    }

    /// Sleeps that have been polled and are not yet due.
    #[must_use]
    pub fn pending_timers(&self) -> usize {
        self.state.borrow().timers.len()
    }

    /// The earliest pending sleep's deadline, always later than now.
    #[must_use]
    pub fn next_deadline(&self) -> Option<MonoNs> {
        self.state.borrow().timers.keys().next().map(|&(at, _)| at)
    }
}

impl Clock for ManualClock {
    fn monotonic_ns(&self) -> MonoNs {
        self.state.borrow().mono
    }
}

impl Timer for ManualTimer {
    async fn sleep_until(&self, at: MonoNs) {
        Sleep {
            state: &self.state,
            at,
            id: None,
        }
        .await;
    }
}

struct Sleep<'a> {
    state: &'a Rc<RefCell<State>>,
    at: MonoNs,
    id: Option<u64>,
}

impl Future for Sleep<'_> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
        let mut s = this.state.borrow_mut();
        if s.mono >= this.at {
            if let Some(id) = this.id.take() {
                s.timers.remove(&(this.at, id));
            }
            return Poll::Ready(());
        }
        let id = *this.id.get_or_insert_with(|| {
            s.next_id += 1;
            s.next_id
        });
        s.timers.insert((this.at, id), cx.waker().clone());
        Poll::Pending
    }
}

impl Drop for Sleep<'_> {
    fn drop(&mut self) {
        if let Some(id) = self.id {
            self.state.borrow_mut().timers.remove(&(self.at, id));
        }
    }
}
