// SPDX-License-Identifier: MIT OR Apache-2.0

//! aarch64 Linux: `ldxr` arms the core's exclusive monitor on the line; a store to any word of
//! it clears the monitor and raises the event `wfe`/`wfet` waits for. The monitor is left armed
//! on return (as Linux's `__cmpwait`): a `clrex` would itself raise the event and make the next
//! park return at once.

use core::arch::asm;
use core::time::Duration;

use crate::core::line::CompletionLine;

/// `HWCAP2_WFXT` (arm64 `asm/hwcap.h`): FEAT_WFxT, the timed wait-for-event.
const HWCAP2_WFXT: u64 = 1 << 31;

/// This core's word-wait capabilities, read once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WordWait {
    /// FEAT_WFxT present: a park is bounded by its own deadline.
    pub timed: bool,
    /// The generic timer's frequency (`cntfrq_el0`), Hz.
    pub counter_hz: u64,
}

fn counter() -> u64 {
    let v: u64;
    // WHY: the virtual counter is a system register only `mrs` reads.
    // SAFETY: `mrs cntvct_el0` is readable at EL0 on Linux (CNTKCTL_EL1.EL0VCTEN is set by the
    // kernel for the vDSO); it touches no memory and no flags.
    // NO SAFE ALTERNATIVE: std's clock is a syscall or vDSO call 20-50x slower than the register,
    // and the park deadline must be in counter ticks.
    unsafe {
        asm!("mrs {v}, cntvct_el0", v = out(reg) v, options(nomem, nostack, preserves_flags))
    };
    v
}

impl WordWait {
    /// Read the counter frequency and whether FEAT_WFxT is present (every aarch64 Linux core
    /// has `ldxr` + `wfe`).
    pub(super) fn probe_arch() -> Result<Self, &'static str> {
        let hz: u64;
        // WHY: the counter frequency is a system register only `mrs` reads.
        // SAFETY: `cntfrq_el0` is readable at EL0 on Linux; no memory, no flags.
        // NO SAFE ALTERNATIVE: no std or libc call returns the generic timer's frequency.
        unsafe {
            asm!("mrs {v}, cntfrq_el0", v = out(reg) hz, options(nomem, nostack, preserves_flags))
        };
        // WHY: `getauxval` is a foreign call.
        // SAFETY: it only reads the process's auxiliary vector and takes a plain integer.
        // NO SAFE ALTERNATIVE: std exposes no hardware-capability query.
        let hwcap2 = unsafe { libc::getauxval(libc::AT_HWCAP2) };
        Ok(Self {
            timed: hwcap2 & HWCAP2_WFXT != 0,
            counter_hz: hz,
        })
    }

    /// See [`WordWait::park`](super::WordWait::park). `ldxr` loads the line's first word and
    /// arms the exclusive monitor on its granule; then `ready` is checked (so a store before the
    /// arm is seen, and one after it raises the event); `wfet`/`wfe` waits for that event.
    pub(super) fn park_arch(
        &self,
        line: &CompletionLine,
        ready: impl Fn(&CompletionLine) -> bool,
        bound: Duration,
    ) -> bool {
        let ticks = u64::try_from(
            bound.as_nanos().saturating_mul(u128::from(self.counter_hz)) / 1_000_000_000,
        )
        .unwrap_or(u64::MAX);
        let deadline = counter().saturating_add(ticks);
        let _armed: u64;
        // WHY: an exclusive load is the only way to arm the monitor the store's event clears.
        // SAFETY: `line.words[0].as_ptr()` is valid, 64-aligned and live for the borrow; `ldxr`
        // only reads.
        // NO SAFE ALTERNATIVE: `AtomicU64::load` compiles to `ldar`/`ldr`, which arm no monitor.
        unsafe {
            asm!("ldxr {v}, [{p}]", p = in(reg) line.words[0].as_ptr(), v = out(reg) _armed,
                 options(nostack, preserves_flags, readonly));
        }
        if !ready(line) {
            if self.timed {
                // WHY: `wfet` (FEAT_WFxT) parks the core until an event or the counter deadline.
                // SAFETY: it reads only the register operand and waits; no memory access. Encoded
                // as `.inst` (`wfet x0`) because the assembler may predate the mnemonic.
                // NO SAFE ALTERNATIVE: no std or libc call waits on the exclusive monitor.
                unsafe {
                    asm!(".inst 0xd5031000", in("x0") deadline, options(nomem, nostack, preserves_flags))
                };
            } else {
                // WHY: `wfe` parks the core until an event (the store, an interrupt, the
                // kernel's event stream).
                // SAFETY: it only waits; no memory access.
                // NO SAFE ALTERNATIVE: as above.
                unsafe { asm!("wfe", options(nomem, nostack, preserves_flags)) };
            }
        }
        // No `clrex` here. Clearing the monitor moves it from exclusive to open, which
        // itself raises the event, so the next park's `wfe` returned at once and every park spun
        // (~60 ns per call, ~1e5 per 10 ms). As Linux's `__cmpwait`, the monitor is left armed:
        // every `stxr` the compiler emits follows its own `ldxr`, which re-arms it.
        ready(line)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    use core::sync::atomic::Ordering;

    fn slot_is(slot: usize, v: u64) -> impl Fn(&CompletionLine) -> bool {
        move |l| l.words[slot].load(Ordering::Acquire) == v
    }

    #[test]
    fn a_slot_already_at_expect_returns_true_at_once() {
        let w = WordWait::probe().expect("aarch64 Linux has a word wait");
        assert!(w.counter_hz > 0);
        let line = CompletionLine::default();
        line.words[5].store(7, Ordering::Release);
        let t = std::time::Instant::now();
        assert!(w.park(&line, slot_is(5, 7), Duration::from_millis(50)));
        assert!(
            t.elapsed() < Duration::from_millis(5),
            "must not park when the slot is set"
        );
    }

    #[test]
    fn a_slot_never_written_returns_false_within_the_bound() {
        let w = WordWait::probe().expect("aarch64 Linux has a word wait");
        let line = CompletionLine::default();
        let t = std::time::Instant::now();
        let mut fired = false;
        while t.elapsed() < Duration::from_millis(2) {
            fired |= w.park(&line, slot_is(0, 1), Duration::from_micros(100));
        }
        assert!(!fired, "an unwritten slot must never read as written");
    }

    #[test]
    fn a_store_to_another_slot_of_the_line_wakes_the_park() {
        // The monitor is armed on word 0; the writer stores word 6: the line-wide granule must
        // still wake the park (the multiplexing case).
        let w = WordWait::probe().expect("aarch64 Linux has a word wait");
        let line = std::sync::Arc::new(CompletionLine::default());
        let writer = {
            let line = std::sync::Arc::clone(&line);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(2));
                line.words[6].store(5, Ordering::Release);
            })
        };
        let t = std::time::Instant::now();
        while !w.park(&line, slot_is(6, 5), Duration::from_millis(1)) {
            assert!(
                t.elapsed() < Duration::from_secs(2),
                "the store never woke the park"
            );
        }
        writer.join().expect("writer");
    }
    /// Park calls until a store 10 ms out, on `w`.
    fn parks_until_a_store_in_10_ms(w: WordWait) -> u64 {
        let line = std::sync::Arc::new(CompletionLine::default());
        let writer = {
            let line = std::sync::Arc::clone(&line);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(10));
                line.words[0].store(1, Ordering::Release);
            })
        };
        let mut parks = 0u64;
        while !w.park(&line, slot_is(0, 1), Duration::from_micros(100)) {
            parks += 1;
        }
        writer.join().expect("writer");
        parks
    }

    #[test]
    fn a_park_sleeps_until_the_store_instead_of_spinning() {
        // Each park returns on the store, its 100 us bound or the kernel's event stream: ~100
        // calls in 10 ms on GB10 (10 kHz stream), ~450 on a GitHub Neoverse runner (faster
        // stream). A park that spins returns in ~60 ns: ~1e5 calls (the `clrex` bug,
        // 2026-11-14, measured 160,636). The bound sits 20x under a spin and >10x over the
        // fastest event stream seen, so it separates the two on any host. Both the timed and
        // the plain arm, back to back (the monitor state of one park must not wake the next).
        let probed = WordWait::probe().expect("aarch64 Linux has a word wait");
        let plain = WordWait {
            timed: false,
            ..probed
        };
        for w in [probed, plain, probed] {
            let parks = parks_until_a_store_in_10_ms(w);
            assert!(parks <= 5_000, "{parks} parks in ~10 ms (timed {}): the park spins", w.timed);
        }
    }
}
