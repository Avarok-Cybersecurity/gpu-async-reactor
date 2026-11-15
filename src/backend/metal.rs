// SPDX-License-Identifier: MIT OR Apache-2.0

//! Metal (Apple): an `MTLSharedEvent` as the completion source. A command buffer encodes
//! `encodeSignalEvent:value:` behind its work; [`SharedEventPort`] queries `signaledValue` and
//! parks in `waitUntilSignaledValue:timeoutMS:`. On an M3 Max this wakes 36-56 us sooner than
//! `waitUntilCompleted` at +20-25 % waiter CPU (README, benchmarks). Safe code only.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice, MTLSharedEvent};

use crate::core::time::DurNs;
use crate::core::wait::CompletionPort;

/// A shared event and the value that completes the current wait.
pub struct SharedEventPort {
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    target: u64,
}

impl SharedEventPort {
    /// A new shared event on `device`, at value 0, or `None` if the device cannot make one.
    #[must_use]
    pub fn new(device: &ProtocolObject<dyn MTLDevice>) -> Option<Self> {
        Some(Self {
            event: device.newSharedEvent()?,
            target: 0,
        })
    }

    /// The event, for `encodeSignalEvent:value:` on a command buffer.
    #[must_use]
    pub fn event(&self) -> &ProtocolObject<dyn MTLSharedEvent> {
        &self.event
    }

    /// The value the current wait completes at.
    #[must_use]
    pub fn target(&self) -> u64 {
        self.target
    }

    /// Advance to the next value and return it: encode `encodeSignalEvent(event, value)` behind
    /// the work, then [`wait`](crate::wait) on the port.
    pub fn next_value(&mut self) -> u64 {
        self.target = self.target.wrapping_add(1);
        self.target
    }
}

impl CompletionPort for SharedEventPort {
    type Error = crate::Error;

    fn query(&mut self) -> crate::Result<bool> {
        Ok(self.event.signaledValue() >= self.target)
    }

    async fn block(&mut self) -> crate::Result<()> {
        while !self.event.waitUntilSignaledValue_timeoutMS(self.target, u64::MAX) {}
        Ok(())
    }

    async fn poll(&mut self) -> crate::Result<()> {
        while !self.query()? {
            core::hint::spin_loop();
        }
        Ok(())
    }

    /// Nothing to enqueue: the signal was encoded with the work ([`SharedEventPort::next_value`]).
    fn arm(&mut self) -> crate::Result<()> {
        Ok(())
    }

    async fn park(&mut self, bound: DurNs) -> crate::Result<bool> {
        let ms = bound.0.div_ceil(1_000_000).max(1);
        Ok(self.event.waitUntilSignaledValue_timeoutMS(self.target, ms))
    }
}
