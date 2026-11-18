// SPDX-License-Identifier: MIT OR Apache-2.0

//! A [`MappedLine`] against a real CUDA driver: the line stays readable after whoever made the
//! context it was created in lets that context go (CWE-416 in 0.1.0: the page was freed with the
//! context and the safe `line()` faulted).
//!
//! The raw driver calls below stand in for a safe context owner's constructor and `Drop` (for
//! example `cudarc`'s `CudaContext`, which retains and releases the primary context, or destroys
//! a non-primary one). The tests need an NVIDIA GPU, so they are `#[ignore]`d; run them with
//! `cargo test --features cuda --test cuda_line_lifetime -- --include-ignored --exact <name>`,
//! one process per test (a fault on a broken build ends the process).
#![cfg(all(feature = "cuda", target_os = "linux"))]
#![allow(unsafe_code)]

use std::sync::atomic::Ordering;

use gpu_async_reactor::Slot;
use gpu_async_reactor::backend::cuda::MappedLine;
use gpu_async_reactor::backend::LineError;

#[path = "support/cuda_driver.rs"]
mod cuda_driver;
use cuda_driver::{Driver, check};

/// A primary-context owner made the line's context current, made the line, then dropped its
/// reference (the count reached 0). 0.1.0: the page went with the context, so the next read
/// through the safe API faulted, and a second line could be handed the same page.
#[test]
#[ignore = "needs an NVIDIA GPU and driver"]
fn a_line_stays_valid_after_the_primary_context_owner_lets_go() {
    let d = Driver::load();
    d.own_primary();
    let stale = MappedLine::new().expect("line");
    stale.line().words[3].store(0x1111, Ordering::Release);
    d.drop_primary();
    assert_eq!(stale.line().words[3].load(Ordering::Acquire), 0x1111);

    d.own_primary();
    let fresh = MappedLine::new().expect("second line");
    assert_ne!(
        core::ptr::from_ref(stale.line()),
        core::ptr::from_ref(fresh.line()),
        "a live line's page is never handed to another line"
    );
    stale.line().words[3].store(0xDEAD_BEEF, Ordering::Release);
    assert_eq!(fresh.line().words[3].load(Ordering::Acquire), 0, "no cross-line store");
    drop(stale);
    fresh.line().words[4].store(7, Ordering::Release);
    assert_eq!(fresh.line().words[4].load(Ordering::Acquire), 7);
    drop(fresh);
    d.drop_primary();
}

/// A non-primary context was current when the line was made, and its owner destroyed it. The
/// line belongs to the device's primary context (which it retains), so the destroyed context's
/// streams are refused and the page outlives it. 0.1.0: the store was accepted, the page died with
/// the context, and the read faulted.
#[test]
#[ignore = "needs an NVIDIA GPU and driver"]
fn a_line_outlives_a_destroyed_non_primary_context_and_refuses_its_streams() {
    let d = Driver::load();
    let mut ctx = 0u64;
    // WHY: plays a non-primary context owner's constructor.
    // SAFETY: `ctx` is a live out-pointer; the new context becomes current.
    // NO SAFE ALTERNATIVE: the driver API is C.
    check("cuCtxCreate", unsafe { (d.create)(&raw mut ctx, 0, d.device) });
    let stream = d.stream();
    let line = MappedLine::new().expect("line");
    assert_eq!(
        line.enqueue_store(stream, Slot::new(1).expect("slot"), 5),
        Err(LineError::ForeignStream),
        "a stream of a context the line does not keep alive"
    );
    // WHY: plays the owner's `Drop`.
    // SAFETY: the stream and context were made above and are not used again.
    // NO SAFE ALTERNATIVE: the driver API is C.
    unsafe {
        check("cuStreamDestroy", (d.stream_destroy)(stream.raw()));
        check("cuCtxDestroy", (d.destroy)(ctx));
    }
    line.line().words[1].store(9, Ordering::Release);
    assert_eq!(line.line().words[1].load(Ordering::Acquire), 9);
}

/// The line still does its job: a store queued on a stream of the primary context lands in it,
/// and the line is dropped after its owner let go.
#[test]
#[ignore = "needs an NVIDIA GPU and driver"]
fn a_store_from_a_primary_context_stream_lands_in_the_line() {
    let d = Driver::load();
    d.own_primary();
    let stream = d.stream();
    let line = MappedLine::new().expect("line");
    line.enqueue_store(stream, Slot::new(2).expect("slot"), 42)
        .expect("store queued");
    // WHY: waits for the queued store.
    // SAFETY: `stream` is live.
    // NO SAFE ALTERNATIVE: the driver API is C.
    check("cuStreamSynchronize", unsafe { (d.stream_sync)(stream.raw()) });
    assert_eq!(line.line().words[2].load(Ordering::Acquire), 42);
    // WHY: the owner's teardown.
    // SAFETY: the stream is not used again.
    // NO SAFE ALTERNATIVE: the driver API is C.
    check("cuStreamDestroy", unsafe { (d.stream_destroy)(stream.raw()) });
    d.drop_primary();
    drop(line);
}
