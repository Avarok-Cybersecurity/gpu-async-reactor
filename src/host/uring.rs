// SPDX-License-Identifier: MIT OR Apache-2.0

//! io_uring: an [`Idle`] whose deadline sleep is one `IORING_OP_TIMEOUT` with a nanosecond
//! timespec (no timer-slack rounding, and the same ring can carry network I/O), and whose line
//! park is the [`StdIdle`] park (the CPU's word wait). The executor kind `io-uring`.

use io_uring::{IoUring, opcode, types};

use crate::core::line::CompletionLine;
use crate::core::reactor::Idle;
use crate::core::time::{Clock, DurNs, MonoNs};
use crate::host::std_host::{StdClock, StdIdle};

/// `ETIME`: the timeout expired (the expected completion of a timeout op).
const ETIME: i32 = 62;

/// The io_uring idle.
pub struct UringIdle {
    ring: IoUring,
    clock: StdClock,
    park: StdIdle,
    failed_timeouts: u64,
}

impl UringIdle {
    /// A ring of 8 entries, or why the host cannot make one (kernel without io_uring, seccomp,
    /// `io_uring_disabled`).
    ///
    /// # Errors
    /// The `io_uring_setup` error.
    pub fn new(clock: StdClock, park: StdIdle) -> std::io::Result<Self> {
        Ok(Self {
            ring: IoUring::new(8)?,
            clock,
            park,
            failed_timeouts: 0,
        })
    }

    /// Timeout ops the ring refused; each such sleep ran as a thread sleep instead (reported,
    /// never silent: a host exports this next to its executor label).
    #[must_use]
    pub fn failed_timeouts(&self) -> u64 {
        self.failed_timeouts
    }

    /// Whether this host can set up an io_uring (`Err` = why not), probed once.
    ///
    /// # Errors
    /// The `io_uring_setup` error.
    pub fn probe() -> std::io::Result<()> {
        IoUring::new(2).map(drop)
    }

    fn timeout(&mut self, ns: u64) -> std::io::Result<()> {
        let ts = types::Timespec::new()
            .sec(ns / 1_000_000_000)
            .nsec(u32::try_from(ns % 1_000_000_000).unwrap_or(0));
        let sqe = opcode::Timeout::new(&raw const ts).build();
        // WHY: submits the timeout entry that bounds the idle on the io_uring.
        // SAFETY: the entry points at `ts`, which lives until `submit_and_wait` below returns with
        // the entry's completion; the ring is ours alone and was just drained.
        // NO SAFE ALTERNATIVE: `SubmissionQueue::push` is unsafe because the kernel reads the entry's pointers after it returns.
        unsafe { self.ring.submission().push(&sqe) }
            .map_err(|_| std::io::Error::other("submission queue full"))?;
        self.ring.submit_and_wait(1)?;
        let cqe = self.ring.completion().next();
        match cqe.map(|c| c.result()) {
            Some(r) if r == -ETIME || r >= 0 => Ok(()),
            Some(r) => Err(std::io::Error::from_raw_os_error(-r)),
            None => Ok(()),
        }
    }
}

impl Idle for UringIdle {
    fn sleep_until(&mut self, at: MonoNs) {
        let left = at.since(self.clock.monotonic_ns());
        if left.0 > 0 && self.timeout(left.0).is_err() {
            self.failed_timeouts += 1;
            self.park.sleep_until(at);
        }
    }

    fn park_line<R: Fn(&CompletionLine) -> bool>(
        &mut self,
        line: &CompletionLine,
        ready: R,
        bound: DurNs,
    ) -> bool {
        self.park.park_line(line, ready, bound)
    }
}
