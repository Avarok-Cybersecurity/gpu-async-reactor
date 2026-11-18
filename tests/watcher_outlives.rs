// SPDX-License-Identifier: MIT OR Apache-2.0

//! A [`Watch`] that outlives its [`Watcher`] is still woken. In 0.1.1 dropping the watcher
//! stopped and joined its thread at once, so a `Watch` still pending (or polled for the first
//! time afterwards) registered into a table nobody scanned and its waker never fired: the await
//! hung. The watcher thread now runs until every `Watch` it handed out has resolved or been
//! dropped, and then exits. One test in this binary, so the thread count it reads is its own.
#![cfg(feature = "std")]

use std::future::Future;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, mpsc};
use std::task::{Context, Poll, Wake, Waker};
use std::time::{Duration, Instant};

use gpu_async_reactor::host::std_host::{StdClock, StdIdle, Watch, Watcher};
use gpu_async_reactor::{CompletionLine, Slot};

struct Signal(Mutex<mpsc::Sender<()>>);

impl Wake for Signal {
    fn wake(self: Arc<Self>) {
        let _ = self.0.lock().expect("sender").send(());
    }
}

/// Threads of this process (`/proc/self/task`).
#[cfg(target_os = "linux")]
fn threads() -> usize {
    std::fs::read_dir("/proc/self/task")
        .expect("/proc/self/task")
        .count()
}

#[cfg(not(target_os = "linux"))]
fn threads() -> usize {
    0
}

/// Wait up to 3 s for this process to have `n` threads; the count it saw last.
fn settle_to(n: usize) -> usize {
    let t = Instant::now();
    let mut now = threads();
    while now != n && t.elapsed() < Duration::from_secs(3) {
        std::thread::sleep(Duration::from_millis(1));
        now = threads();
    }
    now
}

#[test]
fn a_watch_that_outlives_its_watcher_is_woken_and_the_thread_then_exits() {
    let base = threads();
    let line = Arc::new(CompletionLine::default());
    let slot = Slot::new(0).expect("slot");
    let watcher: Watcher<CompletionLine> =
        Watcher::spawn(StdIdle::new(StdClock::new()), Duration::from_millis(1));
    let polled: Watch<CompletionLine> = watcher.wait(Arc::clone(&line), slot, 1);
    let unpolled: Watch<CompletionLine> = watcher.wait(Arc::clone(&line), slot, 2);
    let (tx, rx) = mpsc::channel();
    let waker = Waker::from(Arc::new(Signal(Mutex::new(tx))));
    let mut polled = Box::pin(polled);
    assert!(
        polled
            .as_mut()
            .poll(&mut Context::from_waker(&waker))
            .is_pending()
    );
    drop(watcher);

    // Negative: nothing was stored, so the outliving watch is not woken and stays pending.
    assert!(
        rx.recv_timeout(Duration::from_millis(50)).is_err(),
        "woken before any store"
    );
    assert!(
        polled
            .as_mut()
            .poll(&mut Context::from_waker(&waker))
            .is_pending()
    );

    line.words[0].store(1, Ordering::Release);
    assert!(
        rx.recv_timeout(Duration::from_secs(3)).is_ok(),
        "a watch that outlived its watcher was never woken after its word was written"
    );
    assert_eq!(
        polled.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Ready(())
    );

    // A watch first polled after its watcher was dropped registers with a live thread too.
    let (tx2, rx2) = mpsc::channel();
    let waker2 = Waker::from(Arc::new(Signal(Mutex::new(tx2))));
    let mut unpolled = Box::pin(unpolled);
    assert!(
        unpolled
            .as_mut()
            .poll(&mut Context::from_waker(&waker2))
            .is_pending()
    );
    line.words[0].store(2, Ordering::Release);
    assert!(
        rx2.recv_timeout(Duration::from_secs(3)).is_ok(),
        "a watch first polled after its watcher was dropped was never woken"
    );
    assert_eq!(
        unpolled.as_mut().poll(&mut Context::from_waker(&waker2)),
        Poll::Ready(())
    );

    // Once the last watch is gone the watcher thread exits (it is not leaked).
    if cfg!(target_os = "linux") {
        drop(polled);
        drop(unpolled);
        assert_eq!(
            settle_to(base),
            base,
            "the watcher thread outlived its last watch"
        );
    }
}
