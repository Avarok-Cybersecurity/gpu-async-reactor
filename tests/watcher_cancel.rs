// SPDX-License-Identifier: MIT OR Apache-2.0

//! A cancelled [`Watch`](gpu_async_reactor::host::std_host::Watch) leaves nothing behind in its
//! [`Watcher`]. In 0.1.0 a `Watch` polled once and then dropped (an await under a timeout) kept
//! its entry, its `Arc` of the line and its waker forever, and the watcher thread spun a whole
//! core on the abandoned line. One test in this binary, so the process CPU it reads is the
//! watcher's alone.
#![cfg(feature = "std")]

use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Waker};
use std::time::{Duration, Instant};

use gpu_async_reactor::host::std_host::{StdClock, StdIdle, Watcher};
use gpu_async_reactor::{CompletionLine, Slot};

/// User + system CPU ticks of this process (`/proc/self/stat` fields 14 and 15).
#[cfg(target_os = "linux")]
fn cpu_ticks() -> u64 {
    let stat = std::fs::read_to_string("/proc/self/stat").expect("/proc/self/stat");
    let rest = &stat[stat.rfind(')').expect("comm") + 2..];
    let f: Vec<&str> = rest.split_whitespace().collect();
    f[11].parse::<u64>().expect("utime") + f[12].parse::<u64>().expect("stime")
}

#[test]
fn a_hundred_thousand_cancelled_waits_leave_the_watcher_idle() {
    let watcher: Watcher<CompletionLine> =
        Watcher::spawn(StdIdle::new(StdClock::new()), Duration::from_millis(1));
    let line = Arc::new(CompletionLine::default());
    let slot = Slot::new(0).expect("slot 0");
    for target in 1..=100_000u64 {
        let watch = pin!(watcher.wait(Arc::clone(&line), slot, target));
        assert!(
            watch.poll(&mut Context::from_waker(Waker::noop())).is_pending(),
            "nothing stored yet"
        );
    }
    // The registrations are bounded: every one went with its `Watch`. The watcher may hold the
    // line for the rest of one park (bound 1 ms), so it gets 50 bounds to let go.
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(Arc::strong_count(&line), 1, "a cancelled wait left its line registered");

    // And the watcher sleeps.
    #[cfg(target_os = "linux")]
    {
        let (t0, c0) = (Instant::now(), cpu_ticks());
        std::thread::sleep(Duration::from_millis(1000));
        let ticks = cpu_ticks() - c0;
        let per_s = ticks * 1000 / u64::try_from(t0.elapsed().as_millis()).expect("ms").max(1);
        // 100 ticks/s is one whole core; an idle watcher costs ~0.
        assert!(per_s <= 10, "the watcher burned {per_s} ticks/s with nothing to wait for");
    }
    #[cfg(not(target_os = "linux"))]
    let _ = Instant::now();
}
