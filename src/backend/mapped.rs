// SPDX-License-Identifier: MIT OR Apache-2.0

//! The parent of the CUDA and HIP lines: one page-locked, device-mapped host page holding one
//! [`CompletionLine`], which a stream stores sequence numbers into behind its work. Each vendor
//! supplies only its driver table ([`LineDriver`]); the ownership rules live here once:
//! - a store is accepted only from a stream of the domain (CUDA context, HIP device) the line was
//!   made in, so `Drop` can wait for it;
//! - `Drop` frees the page only after that domain settled; if it cannot, the page is leaked,
//!   never freed under a pending store (CWE-416);
//! - the host reads the line only through atomics; the device writes only the words named.

#![allow(unsafe_code)]

use core::cell::Cell;
use core::ffi::c_void;
use core::ptr::NonNull;
use core::sync::atomic::Ordering;

use crate::core::line::{CompletionLine, Slot};

/// The page the line lives in: a whole page, so no other allocation shares the line with a
/// device writer.
pub(crate) const PAGE: usize = 4096;

/// Why a line could not be made or a store not queued.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineError {
    /// The driver returned this nonzero status.
    Status(i32),
    /// The driver library or one of its entry points is missing.
    Unavailable,
    /// No current context/device on the calling thread.
    NoDomain,
    /// The stream belongs to another context/device than the line's.
    ForeignStream,
}

impl LineError {
    /// One integer per error, as the line's first users logged them: the driver's status, `-1`
    /// unavailable, `-3` no domain, `-4` foreign stream.
    #[must_use]
    pub fn status(self) -> i32 {
        match self {
            Self::Status(s) => s,
            Self::Unavailable => -1,
            Self::NoDomain => -3,
            Self::ForeignStream => -4,
        }
    }
}

impl core::fmt::Display for LineError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Status(s) => write!(f, "driver status {s}"),
            Self::Unavailable => f.write_str("the driver or one of its entry points is missing"),
            Self::NoDomain => f.write_str("no current context on this thread"),
            Self::ForeignStream => f.write_str("the stream belongs to another context"),
        }
    }
}

impl core::error::Error for LineError {}

impl From<LineError> for crate::Error {
    fn from(e: LineError) -> Self {
        crate::error!(Backend, "completion line", e.status())
    }
}

/// What a line needs of a vendor driver. Crate-private: the line is its only caller and passes
/// only values the driver handed out.
pub(crate) trait LineDriver {
    /// The vendor's stream handle.
    type Stream: Copy;
    /// The calling thread's current domain (context or device).
    fn domain_current(&self) -> Result<u64, LineError>;
    /// A page-locked, device-mapped host page of `bytes`.
    fn alloc_host(&self, bytes: usize) -> Result<NonNull<c_void>, LineError>;
    /// The device address of `host`, a page from [`Self::alloc_host`] not yet freed.
    fn device_address(&self, host: NonNull<c_void>) -> Result<u64, LineError>;
    /// Free a page from [`Self::alloc_host`], once.
    fn free_host(&self, host: NonNull<c_void>);
    /// The domain `stream` belongs to.
    fn stream_domain(&self, stream: Self::Stream) -> Result<u64, LineError>;
    /// Queue a store of `value` at device address `addr` on `stream`.
    fn write_value(&self, stream: Self::Stream, addr: u64, value: u64) -> Result<(), LineError>;
    /// Wait for every queued work of `domain`; `false` if the driver could not.
    fn settle(&self, domain: u64) -> bool;
}

/// A device-mapped host line over driver `D`.
pub(crate) struct Line<D: LineDriver> {
    host: NonNull<CompletionLine>,
    device: u64,
    driver: D,
    domain: u64,
    queued: Cell<bool>,
}

// WHY: a line owns a raw mapped page, which withholds the auto `Send`; the executor moves the owner between threads.
// SAFETY: the page is owned by this value alone and the host touches it only through atomics
// (which are `Sync`); the driver table is `Send` by the bound. `queued` is a `Cell`, so the value
// is `Send` but not `Sync`, which is all a line needs.
// NO SAFE ALTERNATIVE: a raw-pointer owner is never `Send` by derivation, and wrapping the page would add an indirection on every host read.
unsafe impl<D: LineDriver + Send> Send for Line<D> {}

impl<D: LineDriver> Line<D> {
    /// A fresh line reading all 0, mapped for the calling thread's current domain.
    pub(crate) fn with_driver(driver: D) -> Result<Self, LineError> {
        let domain = driver.domain_current();
        let page = driver.alloc_host(PAGE)?;
        let device = driver.device_address(page);
        // Built before the checks, so every error below frees the page through `Drop`.
        let line = Self {
            host: page.cast::<CompletionLine>(),
            device: device.unwrap_or(0),
            driver,
            domain: domain.unwrap_or(0),
            queued: Cell::new(false),
        };
        device?;
        if domain? == 0 {
            return Err(LineError::NoDomain);
        }
        for w in &line.line().words {
            w.store(0, Ordering::Release);
        }
        Ok(line)
    }

    /// The driver table.
    #[cfg(feature = "cuda")]
    pub(crate) fn driver(&self) -> &D {
        &self.driver
    }

    /// The device address of word 0 (for a stream-side wait on the line).
    pub(crate) fn device_address(&self) -> u64 {
        self.device
    }

    /// The line, as the host reads it.
    pub(crate) fn line(&self) -> &CompletionLine {
        // WHY: reads the owned mapped page as a `CompletionLine` without copying it.
        // SAFETY: `host` is the live, page-aligned (so 64-aligned, as `CompletionLine` needs) page
        // this value owns until `Drop`; the device writes it only with aligned 8-byte stores to
        // its words and every host access goes through the atomics, so nothing aliases it
        // non-atomically.
        // NO SAFE ALTERNATIVE: a `NonNull<T>` deref is unsafe; the page must stay in place because the device writes it.
        unsafe { self.host.as_ref() }
    }

    /// Queue a store of `value` into word `slot` on `stream`, behind the work queued there.
    pub(crate) fn enqueue_store(
        &self,
        stream: D::Stream,
        slot: Slot,
        value: u64,
    ) -> Result<(), LineError> {
        let offset = 8 * u64::try_from(slot.index()).map_err(|_| LineError::Unavailable)?;
        if self.driver.stream_domain(stream)? != self.domain {
            return Err(LineError::ForeignStream);
        }
        // Set before the call: a store the driver queued despite a failing status still makes
        // `Drop` settle first.
        self.queued.set(true);
        self.driver.write_value(stream, self.device + offset, value)
    }

    /// Every store queued into the line has landed (`true` at once when none was queued).
    fn settled(&self) -> bool {
        !self.queued.get() || self.driver.settle(self.domain)
    }
}

impl<D: LineDriver> Drop for Line<D> {
    fn drop(&mut self) {
        if !self.settled() {
            // A store may still be pending into the page: leak 4 KiB rather than free memory the
            // device may yet write.
            return;
        }
        self.driver.free_host(self.host.cast::<c_void>());
    }
}

#[cfg(test)]
#[path = "mapped_tests.rs"]
mod tests;
