// SPDX-License-Identifier: MIT OR Apache-2.0

//! x86-64 Linux: Intel WAITPKG (`umonitor` + `umwait`, TSC deadline, C0.2) or AMD MONITORX
//! (`monitorx` + `mwaitx`, timer enabled). Both arm a monitor on the line's address range, so a
//! store to the line wakes the core. Compile-checked; no AMD or Intel box ran it yet.

use core::arch::asm;
use core::arch::x86_64::{__cpuid, __cpuid_count, _rdtsc};
use core::time::Duration;

use crate::core::line::CompletionLine;

/// Which monitor/wait pair this CPU has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flavor {
    /// Intel WAITPKG: `umonitor` / `umwait`.
    Umwait,
    /// AMD MONITORX: `monitorx` / `mwaitx`.
    Mwaitx,
}

/// The CPU's user-mode monitor/wait and its TSC rate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WordWait {
    /// The instruction pair.
    pub flavor: Flavor,
    /// TSC ticks per second (CPUID leaf 0x15 when present, else measured at probe).
    pub tsc_hz: u64,
}

/// CPUID.(EAX=7,ECX=0):ECX[5], WAITPKG.
const WAITPKG: u32 = 1 << 5;
/// CPUID.(EAX=8000_0001h):ECX[29], MONITORX.
const MONITORX: u32 = 1 << 29;

fn tsc() -> u64 {
    // WHY: issues the monitor/wait instruction (or reads the counter/ID registers) the word wait needs.
    // SAFETY: `rdtsc` reads the time-stamp counter; it touches no memory (`_rdtsc` is an
    // `unsafe fn` only because it is a target intrinsic, and every x86-64 core has it).
    // NO SAFE ALTERNATIVE: inline asm is unsafe by definition and `core::arch` has no safe form of these instructions.
    #[allow(unused_unsafe)]
    unsafe {
        _rdtsc()
    }
}

fn cpuid(leaf: u32, sub: u32) -> (u32, u32, u32, u32) {
    // WHY: reads the time-stamp counter or CPU id leaves to calibrate and gate the word wait.
    // SAFETY: `cpuid` exists on every x86-64 core and only writes the returned registers.
    // NO SAFE ALTERNATIVE: `core::arch` x86 intrinsics are `unsafe fn`; there is no safe wrapper.
    #[allow(unused_unsafe)]
    let r = unsafe { __cpuid_count(leaf, sub) };
    (r.eax, r.ebx, r.ecx, r.edx)
}

fn tsc_hz() -> u64 {
    // WHY: reads the time-stamp counter or CPU id leaves to calibrate and gate the word wait.
    // SAFETY: as `cpuid` above.
    // NO SAFE ALTERNATIVE: `core::arch` x86 intrinsics are `unsafe fn`; there is no safe wrapper.
    #[allow(unused_unsafe)]
    let max = unsafe { __cpuid(0) }.eax;
    if max >= 0x15 {
        let (den, num, crystal, _) = cpuid(0x15, 0);
        if den != 0 && num != 0 && crystal != 0 {
            return u64::from(crystal) * u64::from(num) / u64::from(den);
        }
    }
    let t0 = std::time::Instant::now();
    let c0 = tsc();
    while t0.elapsed() < Duration::from_millis(2) {
        core::hint::spin_loop();
    }
    let ns = u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX).max(1);
    let ticks = tsc().wrapping_sub(c0);
    u64::try_from(u128::from(ticks) * 1_000_000_000 / u128::from(ns)).unwrap_or(u64::MAX)
}

impl WordWait {
    pub(super) fn probe_arch() -> Result<Self, &'static str> {
        let (_, _, ecx7, _) = cpuid(7, 0);
        let flavor = if ecx7 & WAITPKG != 0 {
            Flavor::Umwait
        } else {
            let (_, _, ecx, _) = cpuid(0x8000_0001, 0);
            if ecx & MONITORX == 0 {
                return Err("no user-mode monitor/wait: the CPU has neither WAITPKG nor MONITORX");
            }
            Flavor::Mwaitx
        };
        Ok(Self {
            flavor,
            tsc_hz: tsc_hz(),
        })
    }

    pub(super) fn park_arch(
        &self,
        line: &CompletionLine,
        ready: impl Fn(&CompletionLine) -> bool,
        bound: Duration,
    ) -> bool {
        let ticks = u64::try_from(
            bound.as_nanos().saturating_mul(u128::from(self.tsc_hz)) / 1_000_000_000,
        )
        .unwrap_or(u64::MAX);
        let addr = line.words[0].as_ptr();
        match self.flavor {
            Flavor::Umwait => {
                // WHY: issues the monitor/wait instruction (or reads the counter/ID registers) the word wait needs.
                // SAFETY: `umonitor` arms the monitor on the address range holding `addr`, a live,
                // 64-aligned word borrowed for this call; it reads no memory and writes none.
                // NO SAFE ALTERNATIVE: inline asm is unsafe by definition and `core::arch` has no safe form of these instructions.
                unsafe { asm!("umonitor {a}", a = in(reg) addr, options(nostack, preserves_flags)) };
                if !ready(line) {
                    let deadline = tsc().saturating_add(ticks);
                    // WHY: issues the monitor/wait instruction (or reads the counter/ID registers) the word wait needs.
                    // SAFETY: `umwait` (control 0 = C0.2) waits until the monitored range is
                    // written, an interrupt, or the TSC reaches EDX:EAX; it touches no memory and
                    // only sets CF, which the flags clobber covers.
                    // NO SAFE ALTERNATIVE: inline asm is unsafe by definition and `core::arch` has no safe form of these instructions.
                    #[expect(clippy::cast_possible_truncation, reason = "EDX:EAX split of a u64")]
                    unsafe {
                        asm!(
                            "umwait {c:e}",
                            c = in(reg) 0u32,
                            in("edx") (deadline >> 32) as u32,
                            in("eax") deadline as u32,
                            options(nostack, nomem),
                        )
                    };
                }
            }
            Flavor::Mwaitx => {
                // WHY: issues the monitor/wait instruction (or reads the counter/ID registers) the word wait needs.
                // SAFETY: `monitorx` arms the monitor on the range holding RAX, a live, 64-aligned
                // word borrowed for this call; ECX/EDX are the reserved extension and hint words.
                // NO SAFE ALTERNATIVE: inline asm is unsafe by definition and `core::arch` has no safe form of these instructions.
                unsafe {
                    asm!("monitorx", in("rax") addr, in("ecx") 0u32, in("edx") 0u32,
                         options(nostack, preserves_flags))
                };
                if !ready(line) {
                    let timeout = u32::try_from(ticks).unwrap_or(u32::MAX);
                    // WHY: issues the monitor/wait instruction (or reads the counter/ID registers) the word wait needs.
                    // SAFETY: `mwaitx` (EAX 0xF0 = stay in C0, ECX bit 1 = timer enabled, EBX =
                    // timeout) waits for a write to the monitored range, an interrupt or the
                    // timeout. RBX is reserved by the compiler, so it is swapped in and back out
                    // around the instruction; no memory is touched.
                    // NO SAFE ALTERNATIVE: inline asm is unsafe by definition and `core::arch` has no safe form of these instructions.
                    unsafe {
                        asm!(
                            "xchg {t:r}, rbx",
                            "mwaitx",
                            "xchg {t:r}, rbx",
                            t = inout(reg) u64::from(timeout) => _,
                            in("eax") 0xF0u32,
                            in("ecx") 2u32,
                            options(nostack, nomem),
                        )
                    };
                }
            }
        }
        ready(line)
    }
}
