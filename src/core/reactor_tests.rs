// SPDX-License-Identifier: MIT OR Apache-2.0

//! The reactor's idle decision, on a virtual clock and a scripted idle: what it sleeps to, what
//! it parks on, and that it never parks past a deadline.

use alloc::vec::Vec;
use core::sync::atomic::Ordering;

use super::*;
use crate::core::executor::Executor;
use crate::core::line::Slot;
use crate::core::manual::ManualClock;

/// An idle that records each call and plays the device: at `store_at` it writes `value` into
/// `slot` of the line it is handed.
pub(crate) struct Script {
    pub clock: ManualClock,
    pub calls: Vec<(&'static str, u64, u64)>,
    pub store: Option<(u64, usize, u64)>,
}

impl Script {
    pub(crate) fn new(clock: &ManualClock) -> Self {
        Self {
            clock: clock.handle(),
            calls: Vec::new(),
            store: None,
        }
    }
}

impl Idle for Script {
    fn sleep_until(&mut self, at: MonoNs) {
        self.calls.push(("sleep", self.clock.monotonic_ns().0, at.0));
        self.clock.set_monotonic(at);
    }

    fn park_line<R: Fn(&CompletionLine) -> bool>(
        &mut self,
        line: &CompletionLine,
        ready: R,
        bound: DurNs,
    ) -> bool {
        let now = self.clock.monotonic_ns().0;
        self.calls.push(("park", now, bound.0));
        // A store due by now lands between the reactor's last check and the arm: a park whose
        // predicate sees it returns at once, as the hardware park does.
        if let Some((at, slot, v)) = self.store
            && at <= now
        {
            line.words[slot].store(v, Ordering::Release);
            self.store = None;
        }
        if ready(line) {
            return true;
        }
        match self.store {
            Some((at, slot, v)) if at <= now + bound.0 => {
                self.clock.set_monotonic(MonoNs(at));
                line.words[slot].store(v, Ordering::Release);
                self.store = None;
            }
            _ => self.clock.advance(bound),
        }
        ready(line)
    }
}

const US: u64 = 1_000;

#[test]
fn the_idle_sleeps_to_the_predicted_deadline_then_parks_on_the_line() {
    let clock = ManualClock::new();
    let line = CompletionLine::default();
    let reactor = Reactor::new(clock.handle());
    let desc = MemoryDescriptor::new(&line, Slot::new(2).expect("slot"), 1);
    let done = core::cell::Cell::new(false);
    let mut ex = Executor::new();
    ex.spawn(async {
        reactor.sleep_until(MonoNs(500 * US)).await;
        assert!(reactor.ready(desc, None).await, "completes, never times out");
        done.set(true);
    });
    let mut idle = Script::new(&clock);
    idle.store = Some((620 * US, 2, 1));
    let stats = ex.run(&reactor, &mut idle, DurNs(100 * US)).expect("run");
    assert!(done.get());
    assert_eq!(
        idle.calls,
        [
            ("sleep", 0, 500 * US),
            ("park", 500 * US, 100 * US),
            ("park", 600 * US, 100 * US),
        ]
    );
    assert_eq!((stats.slept, stats.parked), (1, 2));
    assert_eq!(reactor.pending(), (0, 0), "every registration is gone");
}

#[test]
fn a_park_never_runs_past_the_earliest_deadline() {
    let clock = ManualClock::new();
    let line = CompletionLine::default();
    let reactor = Reactor::new(clock.handle());
    let desc = MemoryDescriptor::new(&line, Slot::new(0).expect("slot"), 7);
    let timed_out = core::cell::Cell::new(None);
    let mut ex = Executor::new();
    ex.spawn(async {
        timed_out.set(Some(!reactor.ready(desc, Some(MonoNs(30 * US))).await));
    });
    let mut idle = Script::new(&clock);
    ex.run(&reactor, &mut idle, DurNs(100 * US)).expect("run");
    assert_eq!(timed_out.get(), Some(true), "the bound ends the wait");
    assert_eq!(idle.calls, [("park", 0, 30 * US)], "bounded by the deadline");
}

#[test]
fn nothing_to_wait_on_is_a_coded_stall_not_a_hang() {
    let clock = ManualClock::new();
    let reactor: Reactor<'_, ManualClock> = Reactor::new(clock.handle());
    let mut ex = Executor::new();
    ex.spawn(core::future::pending::<()>());
    let e = ex
        .run(&reactor, &mut Script::new(&clock), DurNs(100 * US))
        .expect_err("stall");
    assert_eq!(e.code(), crate::ErrorCode::Stalled);
}

#[test]
fn one_park_watches_every_slot_of_its_line() {
    let clock = ManualClock::new();
    let line = CompletionLine::default();
    let reactor = Reactor::new(clock.handle());
    let a = MemoryDescriptor::new(&line, Slot::new(0).expect("0"), 1);
    let b = MemoryDescriptor::new(&line, Slot::new(5).expect("5"), 1);
    let order = core::cell::RefCell::new(Vec::new());
    let mut ex = Executor::new();
    for (name, d, until) in [("a", a, Some(MonoNs(900 * US))), ("b", b, None)] {
        let (order, reactor) = (&order, &reactor);
        ex.spawn(async move {
            let done = reactor.ready(d, until).await;
            order.borrow_mut().push((name, done, reactor.clock().monotonic_ns().0));
        });
    }
    let mut idle = Script::new(&clock);
    idle.store = Some((40 * US, 5, 1));
    ex.run(&reactor, &mut idle, DurNs(1_000 * US)).expect("run");
    assert_eq!(
        *order.borrow(),
        [("b", true, 40 * US), ("a", false, 900 * US)],
        "the store to slot 5 woke b through the park armed for a's line; a then timed out"
    );
    assert_eq!(idle.calls, [("park", 0, 900 * US), ("park", 40 * US, 860 * US)]);
}

#[test]
fn a_store_that_lands_just_before_the_park_returns_at_once_for_any_watched_slot() {
    let clock = ManualClock::new();
    let line = CompletionLine::default();
    let reactor = Reactor::new(clock.handle());
    let a = MemoryDescriptor::new(&line, Slot::new(0).expect("0"), 1);
    let b = MemoryDescriptor::new(&line, Slot::new(5).expect("5"), 1);
    let order = core::cell::RefCell::new(Vec::new());
    let mut ex = Executor::new();
    for (name, d, until) in [("a", a, Some(MonoNs(900 * US))), ("b", b, None)] {
        let (order, reactor) = (&order, &reactor);
        ex.spawn(async move {
            let done = reactor.ready(d, until).await;
            order.borrow_mut().push((name, done, reactor.clock().monotonic_ns().0));
        });
    }
    let mut idle = Script::new(&clock);
    idle.store = Some((0, 5, 1));
    ex.run(&reactor, &mut idle, DurNs(1_000 * US)).expect("run");
    assert_eq!(
        order.borrow()[0],
        ("b", true, 0),
        "the park's predicate covers slot 5 too, so it returned at once instead of sleeping the bound"
    );
}
