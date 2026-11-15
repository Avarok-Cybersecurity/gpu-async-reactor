// SPDX-License-Identifier: MIT OR Apache-2.0

//! The [`Watcher`] wakes outside its lock, and wakes every `Watch` on one word. In 0.1.0 the
//! watcher called wakers while holding its table lock, so a waker that polled or dropped another
//! `Watch` of the same watcher (an inline executor does) deadlocked the watcher thread; and two
//! `Watch`es on one (line, slot, target) shared one entry, so the first was never woken.
#![cfg(feature = "std")]

use std::future::Future;
use std::pin::pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::task::{Context, Poll, Wake, Waker};
use std::time::Duration;

use gpu_async_reactor::host::std_host::{StdClock, StdIdle, Watch, Watcher};
use gpu_async_reactor::{CompletionLine, Slot};

fn slot(i: usize) -> Slot {
    Slot::new(i).expect("slot")
}

fn watcher() -> Watcher<CompletionLine> {
    Watcher::spawn(StdIdle::new(StdClock::new()), Duration::from_millis(1))
}

/// A waker that, when woken, registers and cancels another wait on the same watcher, then
/// reports. It holds the watcher strongly, so a failing run never joins a stuck watcher thread.
struct Reenter {
    watcher: Arc<Watcher<CompletionLine>>,
    line: Arc<CompletionLine>,
    done: Mutex<mpsc::Sender<()>>,
}

impl Wake for Reenter {
    fn wake(self: Arc<Self>) {
        let mut other = pin!(self.watcher.wait(Arc::clone(&self.line), slot(5), 1));
        assert!(other.as_mut().poll(&mut Context::from_waker(Waker::noop())).is_pending());
        let _ = self.done.lock().expect("sender").send(());
    }
}

#[test]
fn a_waker_that_uses_the_same_watcher_does_not_deadlock_it() {
    let shared = Arc::new(watcher());
    let line = Arc::new(CompletionLine::default());
    let (tx, rx) = mpsc::channel();
    let waker = Waker::from(Arc::new(Reenter {
        watcher: Arc::clone(&shared),
        line: Arc::clone(&line),
        done: Mutex::new(tx),
    }));
    let watch: Watch<CompletionLine> = shared.wait(Arc::clone(&line), slot(0), 1);
    let mut watch = pin!(watch);
    assert!(watch.as_mut().poll(&mut Context::from_waker(&waker)).is_pending());
    line.words[0].store(1, Ordering::Release);
    assert!(
        rx.recv_timeout(Duration::from_secs(3)).is_ok(),
        "the waker never finished: the watcher deadlocked on its own lock"
    );
    assert_eq!(watch.as_mut().poll(&mut Context::from_waker(Waker::noop())), Poll::Ready(()));
}

struct Count(AtomicUsize);

impl Wake for Count {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn two_waits_on_one_word_are_both_woken() {
    let w = watcher();
    let line = Arc::new(CompletionLine::default());
    let (a, b) = (Arc::new(Count(AtomicUsize::new(0))), Arc::new(Count(AtomicUsize::new(0))));
    let (wa, wb) = (Waker::from(Arc::clone(&a)), Waker::from(Arc::clone(&b)));
    let mut first = pin!(w.wait(Arc::clone(&line), slot(2), 4));
    let mut second = pin!(w.wait(Arc::clone(&line), slot(2), 4));
    assert!(first.as_mut().poll(&mut Context::from_waker(&wa)).is_pending());
    assert!(second.as_mut().poll(&mut Context::from_waker(&wb)).is_pending());
    line.words[2].store(4, Ordering::Release);
    let t = std::time::Instant::now();
    while (a.0.load(Ordering::SeqCst) == 0 || b.0.load(Ordering::SeqCst) == 0)
        && t.elapsed() < Duration::from_secs(3)
    {
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(a.0.load(Ordering::SeqCst), 1, "the first wait was woken");
    assert_eq!(b.0.load(Ordering::SeqCst), 1, "the second wait was woken");
}
