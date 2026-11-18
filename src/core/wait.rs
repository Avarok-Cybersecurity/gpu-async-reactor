// SPDX-License-Identifier: MIT OR Apache-2.0

//! The completion port and the driver that runs a [`WaitPlan`] over it.
//!
//! One wait, given the plan and the sleep the point's estimate asks for:
//! 1. [`Park::Block`]: the port's blocking wait, nothing else.
//! 2. Otherwise one status query; done → `Immediate` (an exact residual).
//! 3. [`Park::FlagWait`]: arm the device word behind the work.
//! 4. With a sleep: await the [`Timer`] to `start + sleep`, then query; done → `Late`
//!    (censored).
//! 5. Park: [`Park::QueryPoll`] → the port polls to done; [`Park::FlagWait`] → bounded parks
//!    on the word with a status query every `backstop_ns`, then the port polls to done (the
//!    query is the authority). The outcome is an exact residual.
//!
//! Invariants: never returns `Ok` before the port's query or blocking wait reported done; a
//! port error ends the wait with that error; no sleep is longer than the estimate allows.

use crate::core::estimator::Outcome;
use crate::core::plan::{Park, WaitPlan};
use crate::core::time::{Clock, DurNs, MonoNs, Timer};

/// One wait on device work, implemented per backend (the core names no vendor). `query` answers
/// at once; the others take time and are awaited: a host that runs them on the waiting thread
/// returns a ready future, an executor's pend. Errors are the port's own type and pass through
/// [`wait`] unchanged.
#[expect(
    async_fn_in_trait,
    reason = "the waiter is single-threaded by design; no Send bound is wanted"
)]
pub trait CompletionPort {
    /// The backend's error.
    type Error;

    /// The status: `Ok(true)` done, `Ok(false)` pending; `Err` = the work failed.
    ///
    /// # Errors
    /// The device reports a failure.
    fn query(&mut self) -> Result<bool, Self::Error>;

    /// [`Park::Block`]: the vendor's own blocking wait.
    ///
    /// # Errors
    /// The device reports a failure.
    async fn block(&mut self) -> Result<(), Self::Error>;

    /// [`Park::QueryPoll`] and the tail of [`Park::FlagWait`]: query until done, pacing between
    /// two queries as the host declared ([`Pace`]): a port run under an executor must pend
    /// between them, never spin in place.
    ///
    /// # Errors
    /// The device reports a failure.
    async fn poll(&mut self) -> Result<(), Self::Error>;

    /// [`Park::FlagWait`]: enqueue the completion word's write behind the pending work.
    ///
    /// # Errors
    /// The enqueue fails.
    fn arm(&mut self) -> Result<(), Self::Error>;

    /// [`Park::FlagWait`]: park on the armed word for at most `bound`; `Ok(true)` once the word
    /// has been written. May return `Ok(false)` early (a spurious wake).
    ///
    /// # Errors
    /// The park itself fails (never the device status: that is `query`'s).
    async fn park(&mut self, bound: DurNs) -> Result<bool, Self::Error>;
}

/// What a port does between two status queries of [`CompletionPort::poll`], and in a park that
/// has no hardware wait to sleep in: the host's declared choice, given when the port is made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pace {
    /// Spin on the calling thread. Only for an inline host (one wait per thread, the future run
    /// to completion on its first poll): its thread has nothing else to run.
    Spin,
    /// Pend once to the executor between two queries, so every other ready task runs before the
    /// next one (FILL). Under [`Executor`](crate::Executor) a poll never starves ready work.
    Yield,
}

impl Pace {
    /// One step between two status queries: [`Pace::Spin`] is ready on its first poll,
    /// [`Pace::Yield`] pends exactly once (waking itself) so the executor runs every other ready
    /// task first. For ports outside this crate as much as for its own backends.
    pub async fn step(self) {
        match self {
            Self::Spin => core::hint::spin_loop(),
            Self::Yield => crate::core::executor::yield_now().await,
        }
    }
}

/// How one wait ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitKind {
    /// [`Park::Block`]: the port's blocking wait.
    Blocked,
    /// The first query found the work done.
    Immediate,
    /// The query right after the sleep found the work done (the sleep overshot).
    Late,
    /// The park saw the work finish.
    Parked,
}

/// What one wait did, for the estimator (`outcome`) and the host's counters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WaitReport {
    /// How it ended.
    pub kind: WaitKind,
    /// The sample for the point's estimate (`None` for [`WaitKind::Blocked`]).
    pub outcome: Option<Outcome>,
    /// Start to return.
    pub wait_ns: u64,
    /// Time asleep on the timer (0 when the wait did not sleep).
    pub slept_ns: u64,
    /// Parks on the device word (flag waits only).
    pub parks: u64,
    /// Status queries a flag wait's backstop issued.
    pub backstop_queries: u64,
}

/// Run `plan` over `port`. `sleep_ns` is the point's estimate (`PointTable::sleep_ns`)
/// and is used only when `plan.predict` is set.
///
/// # Errors
/// Any port error, unchanged.
pub async fn wait<P, C, T>(
    plan: &WaitPlan,
    sleep_ns: Option<u64>,
    port: &mut P,
    clock: &C,
    timer: &T,
) -> Result<WaitReport, P::Error>
where
    P: CompletionPort,
    C: Clock,
    T: Timer,
{
    let t0 = clock.monotonic_ns();
    let since = |c: &C| c.monotonic_ns().since(t0).0;
    let mut report = WaitReport {
        kind: WaitKind::Blocked,
        outcome: None,
        wait_ns: 0,
        slept_ns: 0,
        parks: 0,
        backstop_queries: 0,
    };
    if plan.park == Park::Block {
        port.block().await?;
        report.wait_ns = since(clock);
        return Ok(report);
    }
    if port.query()? {
        report.wait_ns = since(clock);
        report.kind = WaitKind::Immediate;
        report.outcome = Some(Outcome::Exact {
            residual_ns: report.wait_ns,
        });
        return Ok(report);
    }
    if matches!(plan.park, Park::FlagWait { .. }) {
        port.arm()?;
    }
    if let Some(s) = plan.predict.and(sleep_ns) {
        timer.sleep_until(MonoNs(t0.0.saturating_add(s))).await;
        report.slept_ns = since(clock);
        if port.query()? {
            report.wait_ns = since(clock);
            report.kind = WaitKind::Late;
            report.outcome = Some(Outcome::Late {
                woke_ns: report.slept_ns,
            });
            return Ok(report);
        }
    }
    if let Park::FlagWait {
        park_ns,
        backstop_ns,
    } = plan.park
    {
        let mut last_query = clock.monotonic_ns();
        loop {
            report.parks += 1;
            if port.park(DurNs(park_ns)).await? {
                break;
            }
            let now = clock.monotonic_ns();
            if now.since(last_query).0 >= backstop_ns {
                report.backstop_queries += 1;
                last_query = now;
                if port.query()? {
                    break;
                }
            }
        }
    }
    port.poll().await?;
    report.wait_ns = since(clock);
    report.kind = WaitKind::Parked;
    report.outcome = Some(Outcome::Exact {
        residual_ns: report.wait_ns,
    });
    Ok(report)
}

#[cfg(test)]
#[path = "wait_tests.rs"]
mod tests;
