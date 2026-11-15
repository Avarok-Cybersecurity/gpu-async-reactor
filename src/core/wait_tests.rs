// SPDX-License-Identifier: MIT OR Apache-2.0

//! The wait driven from virtual time: a fake device finishes at `done_at`; the test host moves
//! the clock to the timer's deadline plus `overshoot` whenever the wait is asleep.

use core::pin::pin;
use core::task::{Context, Poll, Waker};

use super::*;
use crate::core::error::{Error, ErrorCode};
use crate::core::manual::ManualClock;
use crate::core::plan::DeadlinePlan;
use crate::error;

const DL: DeadlinePlan = DeadlinePlan {
    margin_ns: 150_000,
    min_sleep_ns: 20_000,
    max_sleep_ns: 50_000_000,
    window: 8,
};
const FLAG: Park = Park::FlagWait {
    park_ns: 100_000,
    backstop_ns: 1_000_000,
};
const MS: u64 = 1_000_000;

struct Fake {
    clock: ManualClock,
    done_at: u64,
    word_works: bool,
    fail: bool,
    armed: bool,
    queries: u32,
    blocks: u32,
}

impl Fake {
    fn new(clock: &ManualClock, done_at: u64) -> Self {
        Self {
            clock: clock.handle(),
            done_at,
            word_works: true,
            fail: false,
            armed: false,
            queries: 0,
            blocks: 0,
        }
    }
    fn now(&self) -> u64 {
        self.clock.monotonic_ns().0
    }
    fn to(&self, t: u64) {
        self.clock.set_monotonic(MonoNs(t.max(self.now())));
    }
}

impl CompletionPort for Fake {
    type Error = Error;

    fn query(&mut self) -> Result<bool, Error> {
        self.queries += 1;
        if self.fail {
            return Err(error!(Backend, "query", "device fault"));
        }
        Ok(self.now() >= self.done_at)
    }
    async fn block(&mut self) -> Result<(), Error> {
        self.blocks += 1;
        self.to(self.done_at);
        Ok(())
    }
    async fn poll(&mut self) -> Result<(), Error> {
        while !self.query()? {
            self.to(self.done_at);
        }
        Ok(())
    }
    fn arm(&mut self) -> Result<(), Error> {
        self.armed = true;
        Ok(())
    }
    async fn park(&mut self, bound: DurNs) -> Result<bool, Error> {
        let until = self.now() + bound.0;
        if self.armed && self.word_works && self.done_at <= until {
            self.to(self.done_at);
            return Ok(true);
        }
        self.to(until);
        Ok(false)
    }
}

/// Drive one wait to the end; every time it sleeps, wake it `overshoot` after its deadline.
fn run(plan: WaitPlan, sleep: Option<u64>, fake: &mut Fake, overshoot: u64) -> Result<WaitReport, Error> {
    let clock = fake.clock.handle();
    let timer = clock.timer();
    let mut fut = pin!(wait(&plan, sleep, fake, &clock, &timer));
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(r) = fut.as_mut().poll(&mut cx) {
            return r;
        }
        let at = clock.next_deadline().expect("pending without a timer");
        clock.set_monotonic(MonoNs(at.0 + overshoot));
    }
}

fn plan(predict: Option<DeadlinePlan>, park: Park) -> WaitPlan {
    WaitPlan { predict, park }
}

#[test]
fn block_uses_only_the_vendor_wait() {
    let clock = ManualClock::new();
    let mut f = Fake::new(&clock, MS);
    let r = run(plan(None, Park::Block), Some(800_000), &mut f, 0).unwrap();
    assert_eq!((f.blocks, f.queries, f.armed), (1, 0, false));
    assert_eq!(
        (r.kind, r.outcome, r.wait_ns, r.slept_ns),
        (WaitKind::Blocked, None, MS, 0)
    );
}

#[test]
fn done_work_returns_at_once_without_arming() {
    let clock = ManualClock::new();
    for park in [Park::QueryPoll, FLAG] {
        let mut f = Fake::new(&clock, 0);
        let r = run(plan(Some(DL), park), Some(800_000), &mut f, 0).unwrap();
        assert_eq!(
            (r.kind, r.slept_ns, f.queries, f.armed),
            (WaitKind::Immediate, 0, 1, false)
        );
        assert_eq!(r.outcome, Some(Outcome::Exact { residual_ns: 0 }));
    }
}

#[test]
fn predict_sleeps_to_the_estimate_then_parks() {
    for park in [Park::QueryPoll, FLAG] {
        let clock = ManualClock::new();
        let mut f = Fake::new(&clock, MS);
        let r = run(plan(Some(DL), park), Some(800_000), &mut f, 0).unwrap();
        assert_eq!(
            (r.kind, r.slept_ns, r.wait_ns),
            (WaitKind::Parked, 800_000, MS),
            "{park:?}"
        );
        assert_eq!(r.outcome, Some(Outcome::Exact { residual_ns: MS }));
        assert_eq!(f.armed, park == FLAG);
        if park == FLAG {
            assert_eq!(r.parks, 2, "200 us left after the sleep: two 100 us parks");
        }
    }
}

#[test]
fn an_overshooting_sleep_is_reported_late() {
    let clock = ManualClock::new();
    let mut f = Fake::new(&clock, MS);
    let r = run(
        plan(Some(DL), Park::QueryPoll),
        Some(800_000),
        &mut f,
        400_000,
    )
    .unwrap();
    assert_eq!(r.kind, WaitKind::Late);
    assert_eq!(r.outcome, Some(Outcome::Late { woke_ns: 1_200_000 }));
}

#[test]
fn a_sleep_is_taken_only_when_the_plan_predicts() {
    let clock = ManualClock::new();
    let mut f = Fake::new(&clock, MS);
    let r = run(plan(None, Park::QueryPoll), Some(800_000), &mut f, 0).unwrap();
    assert_eq!(
        (r.kind, r.slept_ns, clock.pending_timers()),
        (WaitKind::Parked, 0, 0)
    );
    let mut f = Fake::new(&clock, 2 * MS);
    let r = run(plan(Some(DL), Park::QueryPoll), None, &mut f, 0).unwrap();
    assert_eq!(r.slept_ns, 0, "a cold point does not sleep");
}

#[test]
fn a_lost_word_is_caught_by_the_backstop_query() {
    let clock = ManualClock::new();
    let mut f = Fake::new(&clock, MS);
    f.word_works = false;
    let r = run(plan(None, FLAG), None, &mut f, 0).unwrap();
    assert_eq!(r.kind, WaitKind::Parked);
    assert_eq!(
        (r.parks, r.backstop_queries),
        (10, 1),
        "ten 100 us parks, then the 1 ms backstop"
    );
    assert!(f.now() >= f.done_at);
}

#[test]
fn a_device_fault_ends_the_wait_with_its_error() {
    for (park, done_at) in [(Park::QueryPoll, MS), (FLAG, 0)] {
        let clock = ManualClock::new();
        let mut f = Fake::new(&clock, done_at);
        f.fail = true;
        let e =
            run(plan(Some(DL), park), Some(800_000), &mut f, 0).expect_err("fault must surface");
        assert_eq!(e.code(), ErrorCode::Backend);
    }
}

#[test]
fn no_plan_returns_before_the_work_is_done() {
    let plans = [
        plan(None, Park::Block),
        plan(None, Park::QueryPoll),
        plan(Some(DL), Park::QueryPoll),
        plan(None, FLAG),
        plan(Some(DL), FLAG),
    ];
    for p in plans {
        for done_at in [0, 10_000, 170_000, 999_999, MS, 7 * MS] {
            for (sleep, over) in [(None, 0), (Some(500_000), 0), (Some(3 * MS), 2_000_000)] {
                for word_works in [true, false] {
                    let clock = ManualClock::new();
                    let mut f = Fake::new(&clock, done_at);
                    f.word_works = word_works;
                    let r = run(p, sleep, &mut f, over).unwrap();
                    assert!(
                        f.now() >= done_at,
                        "{p:?} returned at {} before {done_at}",
                        f.now()
                    );
                    assert_eq!(r.wait_ns, f.now(), "wait_ns is start to return");
                }
            }
        }
    }
}
