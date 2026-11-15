// SPDX-License-Identifier: MIT OR Apache-2.0

//! Time ports (SBIO): the core reads time only through [`Clock`] and sleeps only through
//! [`Timer`], so a host can replay any wait from virtual time and two hosts given the same clock
//! produce the same decisions.

/// Monotonic nanoseconds from a host-chosen origin.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MonoNs(pub u64);

/// A duration in nanoseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DurNs(pub u64);

impl MonoNs {
    /// The time from `earlier` to `self` (0 if `earlier` is later).
    #[must_use]
    pub const fn since(self, earlier: MonoNs) -> DurNs {
        DurNs(self.0.saturating_sub(earlier.0))
    }

    /// `self + d`, saturating.
    #[must_use]
    pub const fn after(self, d: DurNs) -> MonoNs {
        MonoNs(self.0.saturating_add(d.0))
    }
}

/// The host's monotonic clock: the core's only time source.
pub trait Clock {
    /// Monotonic nanoseconds from the host's origin.
    fn monotonic_ns(&self) -> MonoNs;
}

/// The host's sleep. A host that runs a wait inline returns a future that is ready on its first
/// poll (it slept already); an executor's timer pends until its deadline.
#[expect(
    async_fn_in_trait,
    reason = "the waiter is single-threaded by design; no Send bound is wanted"
)]
pub trait Timer {
    /// Resolves once the host clock reads `at` or later.
    async fn sleep_until(&self, at: MonoNs);
}

impl<C: Clock + ?Sized> Clock for &C {
    fn monotonic_ns(&self) -> MonoNs {
        (**self).monotonic_ns()
    }
}
