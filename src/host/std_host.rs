// SPDX-License-Identifier: MIT OR Apache-2.0

//! std hosts:
//! - [`StdClock`]: `Instant` from an origin, as the core's [`Clock`]; its [`Timer`] sleeps the
//!   calling thread (an inline host: the future is ready on its first poll).
//! - [`StdIdle`]: the executor's idle: `thread::sleep` to a deadline; a park on a line through
//!   the `wordwait` backend when one was probed, else a yield loop to the bound.
//! - [`Watcher`]: a background thread that waits on completion lines and wakes the [`Waker`]s
//!   of futures awaiting them, for consumers on any runtime that are not latency-critical.

use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context, Poll, Waker};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use std::vec::Vec;

use crate::core::line::{CompletionLine, Slot, reached};
use crate::core::reactor::Idle;
use crate::core::time::{Clock, DurNs, MonoNs, Timer};

/// The monotonic clock of the process, from an origin.
#[derive(Clone, Copy, Debug)]
pub struct StdClock {
    origin: Instant,
}

impl StdClock {
    /// A clock whose 0 is now.
    #[must_use]
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
        }
    }

    /// A clock whose 0 is `origin`.
    #[must_use]
    pub fn from_origin(origin: Instant) -> Self {
        Self { origin }
    }
}

impl Default for StdClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for StdClock {
    fn monotonic_ns(&self) -> MonoNs {
        MonoNs(u64::try_from(self.origin.elapsed().as_nanos()).unwrap_or(u64::MAX))
    }
}

impl Timer for StdClock {
    async fn sleep_until(&self, at: MonoNs) {
        let left = at.since(self.monotonic_ns());
        if left.0 > 0 {
            thread::sleep(Duration::from_nanos(left.0));
        }
    }
}

/// How a [`StdIdle`] parks on a line.
#[derive(Clone, Copy, Debug)]
enum LinePark {
    #[cfg(feature = "wordwait")]
    Word(crate::backend::wordwait::WordWait),
    Yield,
}

/// The executor's idle on a std thread.
#[derive(Clone, Copy, Debug)]
pub struct StdIdle {
    clock: StdClock,
    park: LinePark,
}

impl StdIdle {
    /// An idle that parks through the CPU's word wait when the `wordwait` feature is on and the
    /// CPU has one; otherwise it yields to the bound (the declared fallback, see
    /// [`StdIdle::parks_in_hardware`]).
    #[must_use]
    pub fn new(clock: StdClock) -> Self {
        #[cfg(feature = "wordwait")]
        if let Ok(w) = crate::backend::wordwait::WordWait::probe() {
            return Self {
                clock,
                park: LinePark::Word(w),
            };
        }
        Self {
            clock,
            park: LinePark::Yield,
        }
    }

    /// An idle that parks through `wait` (a word wait probed by the caller).
    #[cfg(feature = "wordwait")]
    #[must_use]
    pub fn with_word_wait(clock: StdClock, wait: crate::backend::wordwait::WordWait) -> Self {
        Self {
            clock,
            park: LinePark::Word(wait),
        }
    }

    /// Whether a park sleeps the core in hardware (`false`: it yields the thread in a loop).
    #[must_use]
    pub fn parks_in_hardware(&self) -> bool {
        !matches!(self.park, LinePark::Yield)
    }
}

impl Idle for StdIdle {
    fn sleep_until(&mut self, at: MonoNs) {
        let left = at.since(self.clock.monotonic_ns());
        if left.0 > 0 {
            thread::sleep(Duration::from_nanos(left.0));
        }
    }

    fn park_line<R: Fn(&CompletionLine) -> bool>(
        &mut self,
        line: &CompletionLine,
        ready: R,
        bound: DurNs,
    ) -> bool {
        match self.park {
            #[cfg(feature = "wordwait")]
            LinePark::Word(w) => w.park(line, ready, Duration::from_nanos(bound.0)),
            LinePark::Yield => {
                let until = Instant::now() + Duration::from_nanos(bound.0);
                while !ready(line) {
                    if Instant::now() >= until {
                        return false;
                    }
                    thread::yield_now();
                }
                true
            }
        }
    }
}

fn line_of<L: AsRef<CompletionLine>>(l: &Arc<L>) -> &CompletionLine {
    (**l).as_ref()
}

struct Entry<L> {
    line: Arc<L>,
    slot: Slot,
    target: u64,
    waker: Waker,
}

struct Shared<L> {
    entries: Mutex<Vec<Entry<L>>>,
    stop: std::sync::atomic::AtomicBool,
}

/// A thread that watches completion lines and wakes the futures awaiting them. `L` is anything
/// that holds a line (a backend's mapped line, an `Arc`'d plain line).
pub struct Watcher<L: AsRef<CompletionLine> + Send + Sync + 'static> {
    shared: Arc<Shared<L>>,
    thread: Option<JoinHandle<()>>,
}

impl<L: AsRef<CompletionLine> + Send + Sync + 'static> Watcher<L> {
    /// Start the watcher thread; it parks on a watched line for at most `park_bound` at a time
    /// and sleeps (thread park) while nothing is watched.
    #[must_use]
    pub fn spawn(mut idle: StdIdle, park_bound: Duration) -> Self {
        let shared = Arc::new(Shared {
            entries: Mutex::new(Vec::new()),
            stop: std::sync::atomic::AtomicBool::new(false),
        });
        let s = Arc::clone(&shared);
        let bound = DurNs(u64::try_from(park_bound.as_nanos()).unwrap_or(u64::MAX));
        let thread = thread::spawn(move || {
            while !s.stop.load(std::sync::atomic::Ordering::Acquire) {
                let first = {
                    let mut e = s.entries.lock().unwrap_or_else(PoisonError::into_inner);
                    e.retain(|x| {
                        let done = reached(line_of(&x.line), x.slot, x.target);
                        if done {
                            x.waker.wake_by_ref();
                        }
                        !done
                    });
                    e.first().map(|x| (Arc::clone(&x.line), x.slot, x.target))
                };
                match first {
                    None => thread::park(),
                    Some((line, slot, target)) => {
                        let _ = idle.park_line(
                            line_of(&line),
                            |l| reached(l, slot, target),
                            bound,
                        );
                    }
                }
            }
        });
        Self {
            shared,
            thread: Some(thread),
        }
    }

    /// A future that resolves once word `slot` of `line` reaches `target`.
    #[must_use]
    pub fn wait(&self, line: Arc<L>, slot: Slot, target: u64) -> Watch<L> {
        Watch {
            shared: Arc::clone(&self.shared),
            thread: self.thread.as_ref().map(|t| t.thread().clone()),
            line,
            slot,
            target,
        }
    }
}

impl<L: AsRef<CompletionLine> + Send + Sync + 'static> Drop for Watcher<L> {
    fn drop(&mut self) {
        self.shared
            .stop
            .store(true, std::sync::atomic::Ordering::Release);
        if let Some(t) = self.thread.take() {
            t.thread().unpark();
            let _ = t.join();
        }
    }
}

/// The future of [`Watcher::wait`]: `Send`, usable on any runtime.
pub struct Watch<L: AsRef<CompletionLine> + Send + Sync + 'static> {
    shared: Arc<Shared<L>>,
    thread: Option<thread::Thread>,
    line: Arc<L>,
    slot: Slot,
    target: u64,
}

impl<L: AsRef<CompletionLine> + Send + Sync + 'static> Future for Watch<L> {
    type Output = ();

    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if reached(line_of(&self.line), self.slot, self.target) {
            return Poll::Ready(());
        }
        {
            let mut e = self
                .shared
                .entries
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            match e
                .iter_mut()
                .find(|x| Arc::ptr_eq(&x.line, &self.line) && x.slot == self.slot && x.target == self.target)
            {
                Some(x) => x.waker.clone_from(cx.waker()),
                None => e.push(Entry {
                    line: Arc::clone(&self.line),
                    slot: self.slot,
                    target: self.target,
                    waker: cx.waker().clone(),
                }),
            }
        }
        if let Some(t) = &self.thread {
            t.unpark();
        }
        // A store between the first check and the registration is seen by the watcher's next
        // pass; checking again here makes it immediate.
        if reached(line_of(&self.line), self.slot, self.target) {
            return Poll::Ready(());
        }
        Poll::Pending
    }
}
