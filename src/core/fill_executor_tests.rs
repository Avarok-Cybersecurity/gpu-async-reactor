// SPDX-License-Identifier: MIT OR Apache-2.0

//! FILL: ready work runs while another task waits, and the executor idles only when no task is
//! ready.

use core::cell::{Cell, RefCell};

use alloc::vec::Vec;

use super::*;
use crate::core::line::{CompletionLine, MemoryDescriptor, Slot};
use crate::core::manual::ManualClock;
use crate::core::reactor::tests::Script;
use crate::core::time::{MonoNs, Timer};

const US: u64 = 1_000;

/// A future that is pending `n` times (waking itself), then ready: CPU work in slices.
struct Work<'a> {
    left: u32,
    log: &'a RefCell<Vec<u64>>,
    clock: &'a ManualClock,
    /// Plays the device: stores 1 into slot 1 of this line when the clock reaches 30 us.
    device: &'a CompletionLine,
}

impl Future for Work<'_> {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        self.log.borrow_mut().push(self.clock.monotonic_ns().0);
        if self.left == 0 {
            return Poll::Ready(());
        }
        self.left -= 1;
        self.clock.advance(crate::core::time::DurNs(10 * US));
        if self.clock.monotonic_ns().0 == 30 * US {
            self.device.words[1].store(1, core::sync::atomic::Ordering::Release);
        }
        cx.waker().wake_by_ref();
        Poll::Pending
    }
}

#[test]
fn ready_work_runs_between_waits_and_the_executor_idles_only_when_none_is_ready() {
    let clock = ManualClock::new();
    let line = CompletionLine::default();
    let reactor = Reactor::new(clock.handle());
    let desc = MemoryDescriptor::new(&line, Slot::new(1).expect("slot"), 1);
    let work_log = RefCell::new(Vec::new());
    let waited_at = Cell::new(0u64);
    let mut ex: Executor<core::pin::Pin<alloc::boxed::Box<dyn Future<Output = ()> + '_>>> =
        Executor::new();
    ex.spawn(alloc::boxed::Box::pin(async {
        assert!(reactor.ready(desc, None).await);
        waited_at.set(reactor.clock().monotonic_ns().0);
    }));
    ex.spawn(alloc::boxed::Box::pin(Work {
        left: 5,
        log: &work_log,
        clock: &clock,
        device: &line,
    }));
    let mut idle = Script::new(&clock);
    let stats = ex.run(&reactor, &mut idle, DurNs(100 * US)).expect("run");
    assert_eq!(
        *work_log.borrow(),
        [0, 10 * US, 20 * US, 30 * US, 40 * US, 50 * US],
        "the work ran without waiting for the device"
    );
    assert_eq!(
        idle.calls.len(),
        0,
        "no idle while work was ready; the store landed during the work"
    );
    assert_eq!(waited_at.get(), 30 * US, "the completion became ready work at the next loop, between work slices");
    assert_eq!(stats.idle_ns, 0);
}

#[test]
fn block_on_returns_the_output_and_counts_its_idle() {
    let clock = ManualClock::new();
    let reactor: Reactor<'_, ManualClock> = Reactor::new(clock.handle());
    let mut idle = Script::new(&clock);
    let (v, stats) = block_on(
        async {
            reactor.sleep_until(MonoNs(70 * US)).await;
            42
        },
        &reactor,
        &mut idle,
        DurNs(100 * US),
    )
    .expect("block_on");
    assert_eq!(v, 42);
    assert_eq!((stats.slept, stats.idle_ns), (1, 70 * US));
}

#[test]
fn run_inline_refuses_a_pending_future_with_a_code() {
    let e = run_inline(core::future::pending::<()>(), "test site").expect_err("pending");
    assert_eq!(e.code(), crate::ErrorCode::PendedInline);
    assert_eq!(alloc::format!("{e}"), "test site: a future pended on an inline host");
    assert_eq!(run_inline(async { 3 }, "x").expect("ready"), 3);
}
