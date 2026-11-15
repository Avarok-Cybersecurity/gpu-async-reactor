// SPDX-License-Identifier: MIT OR Apache-2.0

//! The completion line and the memory descriptor.
//!
//! A [`CompletionLine`] is one 64-byte line of eight 64-bit completion words. A device writes a
//! sequence number into a word behind its work; a host waits on the line. On cores whose
//! exclusive-reservation granule is the line (aarch64 with a 64-byte ERG, x86 monitor ranges),
//! any store to the line wakes one waiter, so one waiter multiplexes up to [`LINE_WORDS`]
//! streams.
//!
//! A [`MemoryDescriptor`] names one word and the sequence number that completes it: the GPU
//! analogue of a file descriptor's readiness, and the unit the executor waits on.

use core::sync::atomic::{AtomicU64, Ordering};

/// Completion words per line.
pub const LINE_WORDS: usize = 8;

/// A 64-byte aligned line of completion words.
#[repr(C, align(64))]
#[derive(Debug, Default)]
pub struct CompletionLine {
    /// One word per slot; a device writes a sequence number into its slot's word.
    pub words: [AtomicU64; LINE_WORDS],
}

impl AsRef<CompletionLine> for CompletionLine {
    fn as_ref(&self) -> &CompletionLine {
        self
    }
}

const _: () = assert!(core::mem::size_of::<CompletionLine>() == 64);
const _: () = assert!(core::mem::align_of::<CompletionLine>() == 64);

/// The slot of a word in a [`CompletionLine`] (`0..LINE_WORDS`, checked once at construction).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Slot(u8);

impl Slot {
    /// Slot `i`, or `None` when `i >= LINE_WORDS`.
    #[must_use]
    pub const fn new(i: usize) -> Option<Self> {
        if i < LINE_WORDS {
            // The bound above makes the narrowing exact.
            #[expect(clippy::cast_possible_truncation, reason = "i < LINE_WORDS = 8")]
            Some(Self(i as u8))
        } else {
            None
        }
    }

    /// The index into [`CompletionLine::words`].
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// One awaitable completion: word `slot` of `line` reaching `target`. Sequence numbers increase
/// by one per store, so "reached" is `word >= target` in wrapping order (a later store that lands
/// before the waiter looks still completes it).
#[derive(Clone, Copy, Debug)]
pub struct MemoryDescriptor<'l> {
    line: &'l CompletionLine,
    slot: Slot,
    target: u64,
}

impl<'l> MemoryDescriptor<'l> {
    /// Word `slot` of `line` reaching `target`.
    #[must_use]
    pub const fn new(line: &'l CompletionLine, slot: Slot, target: u64) -> Self {
        Self { line, slot, target }
    }

    /// The line the word lives in.
    #[must_use]
    pub const fn line(&self) -> &'l CompletionLine {
        self.line
    }

    /// The word's slot.
    #[must_use]
    pub const fn slot(&self) -> Slot {
        self.slot
    }

    /// The sequence number that completes the descriptor.
    #[must_use]
    pub const fn target(&self) -> u64 {
        self.target
    }

    /// Whether the word has reached the target (an `Acquire` load: the device's writes before
    /// its store are visible after this returns `true`).
    #[must_use]
    pub fn is_complete(&self) -> bool {
        reached(self.line, self.slot, self.target)
    }
}

/// Whether word `slot` of `line` has reached `target` in wrapping sequence order.
#[must_use]
pub fn reached(line: &CompletionLine, slot: Slot, target: u64) -> bool {
    let w = line.words[slot.index()].load(Ordering::Acquire);
    w.wrapping_sub(target) < (1 << 63)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_word_below_its_target_is_pending_and_at_or_past_it_is_complete() {
        let line = CompletionLine::default();
        let s = Slot::new(3).expect("slot 3");
        let d = MemoryDescriptor::new(&line, s, 5);
        assert!(!d.is_complete(), "0 < 5");
        line.words[3].store(4, Ordering::Release);
        assert!(!d.is_complete(), "4 < 5");
        line.words[3].store(5, Ordering::Release);
        assert!(d.is_complete(), "5 == 5");
        line.words[3].store(6, Ordering::Release);
        assert!(d.is_complete(), "a later store still completes it");
        line.words[2].store(99, Ordering::Release);
        assert!(MemoryDescriptor::new(&line, Slot::new(2).expect("2"), 99).is_complete());
    }

    #[test]
    fn reached_is_wrapping_order() {
        let line = CompletionLine::default();
        let s = Slot::new(0).expect("slot 0");
        line.words[0].store(1, Ordering::Release);
        assert!(reached(&line, s, u64::MAX), "1 follows u64::MAX after a wrap");
        line.words[0].store(u64::MAX, Ordering::Release);
        assert!(!reached(&line, s, 1), "u64::MAX precedes 1");
    }

    #[test]
    fn a_slot_past_the_line_is_refused() {
        assert!(Slot::new(LINE_WORDS).is_none());
        assert_eq!(Slot::new(LINE_WORDS - 1).map(Slot::index), Some(LINE_WORDS - 1));
    }
}
