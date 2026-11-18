// SPDX-License-Identifier: MIT OR Apache-2.0

//! A signal during a [`UringIdle`] sleep does not cut that sleep or any later one short. In
//! 0.1.1 an interrupted `submit_and_wait` left the timeout in flight; every later sleep then
//! consumed the previous sleep's completion and returned at once, so the executor spun to each
//! deadline. Each timeout now carries its own `user_data` and a sleep returns only on its own
//! completion. One test in this binary: it installs a process-wide signal handler.
#![cfg(all(feature = "io-uring", target_os = "linux"))]
#![allow(unsafe_code)]

use std::sync::mpsc;
use std::time::{Duration, Instant};

use gpu_async_reactor::host::std_host::{StdClock, StdIdle};
use gpu_async_reactor::host::uring::UringIdle;
use gpu_async_reactor::{Clock, Idle};

extern "C" fn on_signal(_: libc::c_int) {}

/// Install a no-op `SIGUSR1` handler without `SA_RESTART`, so the signal interrupts a blocking
/// `io_uring_enter` with `EINTR` instead of killing the process.
fn install_handler() {
    // WHY: the test needs a signal that interrupts a system call without ending the process.
    // SAFETY: `act` is zeroed (a valid `sigaction`) and then given a handler of the signature the
    // kernel calls; the handler does nothing, so it is async-signal-safe.
    // NO SAFE ALTERNATIVE: `sigaction(2)` is only reachable through libc.
    unsafe {
        let mut act: libc::sigaction = std::mem::zeroed();
        act.sa_sigaction = on_signal as extern "C" fn(libc::c_int) as libc::sighandler_t;
        libc::sigemptyset(&raw mut act.sa_mask);
        act.sa_flags = 0;
        assert_eq!(
            libc::sigaction(libc::SIGUSR1, &raw const act, std::ptr::null_mut()),
            0
        );
    }
}

const SLEEP: Duration = Duration::from_millis(200);

#[test]
fn a_signal_mid_sleep_does_not_shorten_that_sleep_or_the_next() {
    install_handler();
    let (tid_tx, tid_rx) = mpsc::channel();
    let sleeper = std::thread::spawn(move || {
        let clock = StdClock::new();
        let mut idle = UringIdle::new(clock, StdIdle::new(clock)).expect("io_uring on this host");
        // WHY: the main thread signals this thread by its pthread id.
        // SAFETY: `pthread_self` has no preconditions.
        // NO SAFE ALTERNATIVE: std exposes no pthread id.
        tid_tx
            .send(unsafe { libc::pthread_self() })
            .expect("send tid");
        let mut slept = Vec::new();
        for _ in 0..3 {
            let t = Instant::now();
            idle.sleep_until(clock.monotonic_ns().after(gpu_async_reactor::DurNs(
                u64::try_from(SLEEP.as_nanos()).expect("ns"),
            )));
            slept.push(t.elapsed());
        }
        (slept, idle.failed_timeouts())
    });
    let tid = tid_rx.recv().expect("tid");
    std::thread::sleep(SLEEP / 4);
    // WHY: interrupt the sleeper's first `io_uring_enter`.
    // SAFETY: `tid` names a thread that is alive (it is joined below, after this call).
    // NO SAFE ALTERNATIVE: `pthread_kill(3)` is only reachable through libc.
    assert_eq!(unsafe { libc::pthread_kill(tid, libc::SIGUSR1) }, 0);
    let (slept, failed) = sleeper.join().expect("sleeper");
    for (i, d) in slept.iter().enumerate() {
        assert!(
            *d >= SLEEP,
            "sleep {i} returned after {d:?}, before its {SLEEP:?} deadline: {slept:?}"
        );
    }
    assert_eq!(
        failed, 0,
        "an interrupted wait is resumed, not counted as a failed timeout"
    );
}
