// SPDX-License-Identifier: MIT OR Apache-2.0

//! tokio: a [`Clock`] + [`Timer`] over `tokio::time`, so [`wait`](crate::wait) runs as a tokio
//! task (its predicted sleep is a tokio timer; tokio's timer wheel has 1 ms resolution, so this
//! host suits consumers that are not latency-critical). Pair it with
//! [`Watcher`](crate::host::std_host::Watcher) to await completion lines from tokio tasks.

use tokio::time::Instant;

use crate::core::time::{Clock, MonoNs, Timer};

/// tokio's clock from an origin (paused time in tests moves it too).
#[derive(Clone, Copy, Debug)]
pub struct TokioClock {
    origin: Instant,
}

impl TokioClock {
    /// A clock whose 0 is now.
    #[must_use]
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl Default for TokioClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for TokioClock {
    fn monotonic_ns(&self) -> MonoNs {
        MonoNs(u64::try_from(self.origin.elapsed().as_nanos()).unwrap_or(u64::MAX))
    }
}

impl Timer for TokioClock {
    async fn sleep_until(&self, at: MonoNs) {
        tokio::time::sleep_until(self.origin + core::time::Duration::from_nanos(at.0)).await;
    }
}
