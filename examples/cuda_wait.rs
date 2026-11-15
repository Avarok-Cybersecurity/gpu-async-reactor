// SPDX-License-Identifier: MIT OR Apache-2.0

//! A real CUDA wait, on the device's primary context (the only context a [`MappedLine`] accepts
//! streams from): a 256 MiB `cuMemsetD8Async` on a stream, then [`wait`] with each plan
//! (`block`, `poll`, `flag`, `flag+predict`) over a [`CudaPort`]; prints the wait time and how
//! each wait ended. Needs an NVIDIA driver and GPU.
//!
//! `cargo run --release --example cuda_wait --features cuda`
#![allow(unsafe_code)]

use gpu_async_reactor::backend::cuda::{CuStream, CudaPort, MappedLine};
use gpu_async_reactor::backend::wordwait::WordWait;
use gpu_async_reactor::host::std_host::StdClock;
use gpu_async_reactor::{PointTable, Slot, WaitPlan, run_inline, wait};
use libloading::Library;

type Init = unsafe extern "C" fn(u32) -> i32;
type DeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
type PrimaryRetain = unsafe extern "C" fn(*mut u64, i32) -> i32;
type CtxSetCurrent = unsafe extern "C" fn(u64) -> i32;
type StreamCreate = unsafe extern "C" fn(*mut u64, u32) -> i32;
type MemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
type MemsetAsync = unsafe extern "C" fn(u64, u8, usize, u64) -> i32;

fn check(what: &str, s: i32) {
    assert_eq!(s, 0, "{what} failed with CUresult {s}");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const BYTES: usize = 256 << 20;
    // WHY: the example retains the device's primary context (the one a line belongs to) and makes its stream and buffer through the raw driver API, to show a stream wait without a vendor crate.
    // SAFETY: the example's own driver setup: each symbol has the cuda.h signature named by its
    // type, every out-pointer is a live local, and the context reference, stream and buffer live
    // until the process exits.
    // NO SAFE ALTERNATIVE: the driver API is C: every entry point is an unsafe fn pointer.
    let (stream, buf, memset) = unsafe {
        let lib = Library::new("libcuda.so.1")?;
        check("cuInit", lib.get::<Init>(b"cuInit")?(0));
        let mut dev = 0;
        check("cuDeviceGet", lib.get::<DeviceGet>(b"cuDeviceGet")?(&raw mut dev, 0));
        let mut ctx = 0u64;
        check("cuDevicePrimaryCtxRetain", lib.get::<PrimaryRetain>(b"cuDevicePrimaryCtxRetain")?(&raw mut ctx, dev));
        check("cuCtxSetCurrent", lib.get::<CtxSetCurrent>(b"cuCtxSetCurrent")?(ctx));
        let mut s = 0u64;
        check("cuStreamCreate", lib.get::<StreamCreate>(b"cuStreamCreate")?(&raw mut s, 1));
        let mut buf = 0u64;
        check("cuMemAlloc", lib.get::<MemAlloc>(b"cuMemAlloc_v2")?(&raw mut buf, BYTES));
        let memset = *lib.get::<MemsetAsync>(b"cuMemsetD8Async")?;
        std::mem::forget(lib);
        (s, buf, memset)
    };
    // WHY: wraps the example's raw stream handle in the typed stream the reactor waits on.
    // SAFETY: `stream` was just created on the retained primary context and is never destroyed.
    // NO SAFE ALTERNATIVE: `CuStream::from_raw` is the crate's only constructor from a handle, and the handle comes from the driver.
    let cu = unsafe { CuStream::from_raw(stream) };
    let line = MappedLine::new()?;
    let word = WordWait::probe().ok();
    let clock = StdClock::new();
    let slot = Slot::new(0).expect("slot 0");
    let mut seq = 0u64;
    let mut points: PointTable<()> = PointTable::new();
    for text in [
        "block",
        "poll",
        "flag:park_us=100,backstop_us=1000",
        "flag:park_us=100,backstop_us=1000+predict:margin_us=150,min_sleep_us=20,max_sleep_us=50000,window=8",
    ] {
        let plan = WaitPlan::parse(text)?;
        for rep in 0..5 {
            // WHY: queues work on the stream so the wait has a real completion to observe.
            // SAFETY: `buf` holds BYTES bytes on the stream's context.
            // NO SAFE ALTERNATIVE: the driver call is an unsafe fn pointer.
            check("cuMemsetD8Async", unsafe { memset(buf, rep, BYTES, stream) });
            let sleep = plan.predict.and_then(|d| points.sleep_ns(&(), &d));
            let mut port = CudaPort::new(&line, cu, slot, &mut seq, word);
            let r = run_inline(wait(&plan, sleep, &mut port, &clock, &clock), "cuda_wait")??;
            if let (Some(d), Some(o)) = (plan.predict, r.outcome) {
                points.observe((), &d, o);
            }
            println!(
                "{:<5} rep {rep}: {:?} in {:>8.1} us (slept {:>7.1} us, parks {})",
                text.split([':', '+']).next().unwrap_or(text),
                r.kind,
                r.wait_ns as f64 / 1e3,
                r.slept_ns as f64 / 1e3,
                r.parks
            );
        }
    }
    Ok(())
}
