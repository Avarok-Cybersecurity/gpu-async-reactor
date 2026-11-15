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
use gpu_async_reactor::backend::cuda::{CuStream, MappedLine};
use gpu_async_reactor::backend::LineError;
use libloading::Library;

type Init = unsafe extern "C" fn(u32) -> i32;
type DeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
type PrimaryRetain = unsafe extern "C" fn(*mut u64, i32) -> i32;
type PrimaryRelease = unsafe extern "C" fn(i32) -> i32;
type CtxSetCurrent = unsafe extern "C" fn(u64) -> i32;
type CtxCreate = unsafe extern "C" fn(*mut u64, u32, i32) -> i32;
type CtxDestroy = unsafe extern "C" fn(u64) -> i32;
type StreamCreate = unsafe extern "C" fn(*mut u64, u32) -> i32;
type StreamSync = unsafe extern "C" fn(u64) -> i32;
type StreamDestroy = unsafe extern "C" fn(u64) -> i32;

/// The few driver entry points a context owner uses.
struct Driver {
    retain: PrimaryRetain,
    release: PrimaryRelease,
    set_current: CtxSetCurrent,
    create: CtxCreate,
    destroy: CtxDestroy,
    stream_create: StreamCreate,
    stream_sync: StreamSync,
    stream_destroy: StreamDestroy,
    device: i32,
    _lib: Library,
}

fn check(what: &str, s: i32) {
    assert_eq!(s, 0, "{what} failed with CUresult {s}");
}

impl Driver {
    fn load() -> Self {
        // WHY: the test plays a context owner through the raw driver API, without a vendor crate.
        // SAFETY: each symbol has the cuda.h (13.0) signature named by its type; every
        // out-pointer is a live local; the library stays loaded for as long as the table.
        // NO SAFE ALTERNATIVE: the driver API is C: every entry point is an unsafe fn pointer.
        unsafe {
            let lib = Library::new("libcuda.so.1").expect("libcuda.so.1");
            check("cuInit", lib.get::<Init>(b"cuInit").expect("cuInit")(0));
            let mut device = 0;
            check(
                "cuDeviceGet",
                lib.get::<DeviceGet>(b"cuDeviceGet").expect("sym")(&raw mut device, 0),
            );
            Self {
                retain: *lib.get(b"cuDevicePrimaryCtxRetain").expect("sym"),
                release: *lib.get(b"cuDevicePrimaryCtxRelease_v2").expect("sym"),
                set_current: *lib.get(b"cuCtxSetCurrent").expect("sym"),
                create: *lib.get(b"cuCtxCreate_v2").expect("sym"),
                destroy: *lib.get(b"cuCtxDestroy_v2").expect("sym"),
                stream_create: *lib.get(b"cuStreamCreate").expect("sym"),
                stream_sync: *lib.get(b"cuStreamSynchronize").expect("sym"),
                stream_destroy: *lib.get(b"cuStreamDestroy_v2").expect("sym"),
                device,
                _lib: lib,
            }
        }
    }

    /// What a primary-context owner does on construction: retain it and make it current.
    fn own_primary(&self) -> u64 {
        let mut ctx = 0u64;
        // WHY: driver calls with live out-pointers and plain handles.
        // SAFETY: `ctx` is a live out-pointer; the retained context is live until `release`.
        // NO SAFE ALTERNATIVE: the driver API is C.
        unsafe {
            check("retain", (self.retain)(&raw mut ctx, self.device));
            check("set current", (self.set_current)(ctx));
        }
        ctx
    }

    /// What a primary-context owner's `Drop` does: release its reference.
    fn drop_primary(&self) {
        // WHY: driver calls on plain handles.
        // SAFETY: undoes exactly one `own_primary`.
        // NO SAFE ALTERNATIVE: the driver API is C.
        unsafe {
            check("set current 0", (self.set_current)(0));
            check("release", (self.release)(self.device));
        }
    }

    fn stream(&self) -> CuStream {
        let mut s = 0u64;
        // WHY: driver call with a live out-pointer.
        // SAFETY: `s` is a live out-pointer; the stream lives on the current context.
        // NO SAFE ALTERNATIVE: the driver API is C.
        check("cuStreamCreate", unsafe { (self.stream_create)(&raw mut s, 1) });
        // WHY: wraps the stream just created.
        // SAFETY: `s` is a live stream of the current context, destroyed only after its last use.
        // NO SAFE ALTERNATIVE: `CuStream::from_raw` is the only constructor from a handle.
        unsafe { CuStream::from_raw(s) }
    }
}

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
