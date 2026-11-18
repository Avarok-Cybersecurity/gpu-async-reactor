// SPDX-License-Identifier: MIT OR Apache-2.0

//! A [`CudaPort`] poll under the crate's [`Executor`] lets every other ready task run between its
//! status queries. In 0.1.1 `poll` was `while !query() { spin_loop() }` inside an `async fn`: it
//! never pended, so a task waiting on the GPU starved every other task on the executor, including
//! the one that would let the GPU finish. Here the stream waits on a word of the line that a
//! second task writes; a watchdog writes it after 3 s so a spinning poll fails instead of hanging.
//!
//! Needs an NVIDIA GPU, so `#[ignore]`d; run it with
//! `cargo test --features cuda --test cuda_poll_yields -- --include-ignored`.
#![cfg(all(feature = "cuda", target_os = "linux"))]
#![allow(unsafe_code)]

use std::cell::Cell;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::time::Duration;

use gpu_async_reactor::backend::cuda::{CudaPort, MappedLine};
use gpu_async_reactor::host::std_host::{StdClock, StdIdle};
use gpu_async_reactor::{
    DurNs, Executor, Pace, Reactor, Slot, WaitKind, WaitPlan, wait, yield_now,
};

#[path = "support/cuda_driver.rs"]
mod cuda_driver;
use cuda_driver::{Driver, check};

/// `CU_STREAM_WAIT_VALUE_GEQ`.
const GEQ: u32 = 0;
const GATE: usize = 7;

type Task<'a> = Pin<Box<dyn Future<Output = ()> + 'a>>;

/// Run a `plan` wait on a stream gated by word [`GATE`] next to a task that writes that word.
/// Returns whether the other task had run when the wait returned, and how the wait ended.
fn wait_beside_the_task_that_opens_the_gate(plan: &str) -> (bool, WaitKind) {
    let d = Driver::load();
    d.own_primary();
    let stream = d.stream();
    let line = MappedLine::new().expect("line");
    let gate = line.device_address() + 8 * u64::try_from(GATE).expect("gate");
    // WHY: blocks the stream on a word of the line until the host writes it.
    // SAFETY: `stream` is live; `gate` is a word of the line's live device-mapped page, which
    // outlives the stream's use of it (the stream is synchronized below before the line drops).
    // NO SAFE ALTERNATIVE: the driver API is C.
    check("cuStreamWaitValue64", unsafe {
        (d.wait_value)(stream.raw(), gate, 1, GEQ)
    });
    let plan = WaitPlan::parse(plan).expect("plan");
    let clock = StdClock::new();
    let reactor = Reactor::new(clock);
    let (opened, seen) = (Cell::new(false), Cell::new(None));
    let words = line.line();
    let (done_tx, done_rx) = mpsc::channel::<()>();
    std::thread::scope(|s| {
        s.spawn(move || {
            if done_rx.recv_timeout(Duration::from_secs(3)).is_err() {
                words.words[GATE].store(1, Ordering::Release);
            }
        });
        let mut seq = 0;
        let mut ex: Executor<Task<'_>> = Executor::new();
        ex.spawn(Box::pin(async {
            let mut port = CudaPort::paced(
                &line,
                stream,
                Slot::new(0).expect("slot"),
                &mut seq,
                None,
                Pace::Yield,
            );
            let r = wait(&plan, None, &mut port, &clock, &reactor)
                .await
                .expect("wait");
            seen.set(Some((opened.get(), r.kind)));
        }));
        ex.spawn(Box::pin(async {
            yield_now().await;
            opened.set(true);
            words.words[GATE].store(1, Ordering::Release);
        }));
        ex.run(&reactor, &mut StdIdle::new(clock), DurNs(1_000_000))
            .expect("run");
        let _ = done_tx.send(());
    });
    // WHY: settles the stream before its line and context go.
    // SAFETY: `stream` is live and not used after `cuStreamDestroy`.
    // NO SAFE ALTERNATIVE: the driver API is C.
    unsafe {
        check("cuStreamSynchronize", (d.stream_sync)(stream.raw()));
        check("cuStreamDestroy", (d.stream_destroy)(stream.raw()));
    }
    drop(line);
    d.drop_primary();
    seen.get().expect("the wait task finished")
}

#[test]
#[ignore = "needs an NVIDIA GPU and driver"]
fn a_query_poll_yields_to_the_task_that_lets_the_gpu_finish() {
    let (opened, kind) = wait_beside_the_task_that_opens_the_gate("poll");
    assert!(
        opened,
        "the poll never yielded: the other task ran only after the 3 s watchdog"
    );
    assert_eq!(kind, WaitKind::Parked);
}

/// A flag wait with no word wait: each park is one paced step, not a return straight back into
/// the park loop (0.1.1 spun there without yielding, the same starvation).
#[test]
#[ignore = "needs an NVIDIA GPU and driver"]
fn a_flag_wait_without_a_word_wait_yields_to_the_task_that_lets_the_gpu_finish() {
    let (opened, kind) =
        wait_beside_the_task_that_opens_the_gate("flag:park_us=100,backstop_us=1000");
    assert!(
        opened,
        "the park loop never yielded: the other task ran only after the 3 s watchdog"
    );
    assert_eq!(kind, WaitKind::Parked);
}
