// SPDX-License-Identifier: MIT OR Apache-2.0

//! std hosts:
//! - [`StdClock`]: `Instant` from an origin, as the core's [`Clock`]; its [`Timer`] sleeps the
//!   calling thread (an inline host: the future is ready on its first poll).
//! - [`StdIdle`]: the executor's idle: `thread::sleep` to a deadline; a park on a line through
//!   the `wordwait` backend when one was probed, else a yield loop to the bound.
//! - [`Watcher`]: a background thread that waits on completion lines and wakes the [`Waker`]s
//!   of futures awaiting them, for consumers on any runtime that are not latency-critical. Each
//!   [`Watch`] owns one registration, removed when it resolves or is dropped; wakers run with no
//!   lock held.

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

/// One [`Watch`]'s registration.
struct Entry<L> {
    id: u64,
    line: Arc<L>,
    slot: Slot,
    target: u64,
    waker: Waker,
}

/// The registrations, and the id the next one gets.
struct Table<L> {
    next: u64,
    entries: Vec<Entry<L>>,
}

struct Shared<L> {
    table: Mutex<Table<L>>,
    stop: std::sync::atomic::AtomicBool,
    /// [`Watch`]es handed out and not yet dropped: the thread serves them after `stop`.
    watches: std::sync::atomic::AtomicUsize,
}

impl<L> Shared<L> {
    fn lock(&self) -> std::sync::MutexGuard<'_, Table<L>> {
        self.table.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// A thread that watches completion lines and wakes the futures awaiting them. `L` is anything
/// that holds a line (a backend's mapped line, an `Arc`'d plain line).
///
/// Dropping the watcher stops its thread once every [`Watch`] it handed out has resolved or been
/// dropped: a `Watch` that outlives its watcher is still woken. With no `Watch` alive the drop
/// joins the thread; otherwise the thread exits on its own when the last one goes.
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
            table: Mutex::new(Table {
                next: 0,
                entries: Vec::new(),
            }),
            stop: std::sync::atomic::AtomicBool::new(false),
            watches: std::sync::atomic::AtomicUsize::new(0),
        });
        let s = Arc::clone(&shared);
        let bound = DurNs(u64::try_from(park_bound.as_nanos()).unwrap_or(u64::MAX));
        let thread = thread::spawn(move || {
            while !(s.stop.load(std::sync::atomic::Ordering::Acquire)
                && s.watches.load(std::sync::atomic::Ordering::Acquire) == 0)
            {
                let (done, first) = {
                    let mut t = s.lock();
                    let done: Vec<Entry<L>> = t
                        .entries
                        .extract_if(.., |x| reached(line_of(&x.line), x.slot, x.target))
                        .collect();
                    let first = t.entries.first().map(|x| (Arc::clone(&x.line), x.slot, x.target));
                    (done, first)
                };
                // Woken with the lock released: a waker may poll or drop a `Watch` of this
                // watcher (an inline executor does), which takes the lock.
                for e in done {
                    e.waker.wake();
                }
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
        self.shared
            .watches
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        Watch {
            shared: Arc::clone(&self.shared),
            thread: self.thread.as_ref().map(|t| t.thread().clone()),
            line,
            slot,
            target,
            id: None,
        }
    }
}

impl<L: AsRef<CompletionLine> + Send + Sync + 'static> Drop for Watcher<L> {
    fn drop(&mut self) {
        self.shared
            .stop
            .store(true, std::sync::atomic::Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            t.thread().unpark();
            // No `Watch` is alive and none can be made (that takes `&self`): the thread is
            // leaving. Otherwise it is detached and exits after the last `Watch` drops.
            if self.shared.watches.load(std::sync::atomic::Ordering::SeqCst) == 0 {
                let _ = t.join();
            }
        }
    }
}

/// The future of [`Watcher::wait`]: `Send`, usable on any runtime. Dropping it before it
/// resolves (a cancelled await) removes its registration.
pub struct Watch<L: AsRef<CompletionLine> + Send + Sync + 'static> {
    shared: Arc<Shared<L>>,
    thread: Option<thread::Thread>,
    line: Arc<L>,
    slot: Slot,
    target: u64,
    /// This future's registration in the watcher, once polled pending.
    id: Option<u64>,
}

impl<L: AsRef<CompletionLine> + Send + Sync + 'static> Watch<L> {
    fn reached(&self) -> bool {
        reached(line_of(&self.line), self.slot, self.target)
    }

    /// Remove this future's registration, if it still has one. The entry (its waker and line) is
    /// dropped after the lock is released.
    fn deregister(&mut self) {
        let Some(id) = self.id.take() else { return };
        let removed = {
            let mut t = self.shared.lock();
            t.entries
                .iter()
                .position(|x| x.id == id)
                .map(|i| t.entries.swap_remove(i))
        };
        drop(removed);
    }
}

impl<L: AsRef<CompletionLine> + Send + Sync + 'static> Future for Watch<L> {
    type Output = ();

    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
        if this.reached() {
            this.deregister();
            return Poll::Ready(());
        }
        {
            let mut t = this.shared.lock();
            let mine = this.id.and_then(|id| t.entries.iter_mut().find(|x| x.id == id));
            match mine {
                Some(x) => x.waker.clone_from(cx.waker()),
                // Not registered yet, or the watcher took the entry out to wake it.
                None => {
                    let id = t.next;
                    t.next = t.next.wrapping_add(1);
                    t.entries.push(Entry {
                        id,
                        line: Arc::clone(&this.line),
                        slot: this.slot,
                        target: this.target,
                        waker: cx.waker().clone(),
                    });
                    this.id = Some(id);
                }
            }
        }
        if let Some(t) = &this.thread {
            t.unpark();
        }
        // A store between the first check and the registration is seen by the watcher's next
        // pass; checking again here makes it immediate.
        if this.reached() {
            this.deregister();
            return Poll::Ready(());
        }
        Poll::Pending
    }
}

impl<L: AsRef<CompletionLine> + Send + Sync + 'static> Drop for Watch<L> {
    fn drop(&mut self) {
        self.deregister();
        self.shared
            .watches
            .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        // A stopped watcher's thread exits once this was the last watch; it may be parked.
        // SeqCst pairs with the watcher's drop (store `stop`, then load `watches`): either this
        // sees `stop`, or that drop sees this decrement and joins the thread itself.
        if self.shared.stop.load(std::sync::atomic::Ordering::SeqCst)
            && let Some(t) = &self.thread
        {
            t.unpark();
        }
    }
}
