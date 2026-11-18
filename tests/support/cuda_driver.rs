// SPDX-License-Identifier: MIT OR Apache-2.0

//! The raw CUDA driver calls the GPU tests use to play a context owner (for example `cudarc`'s
//! `CudaContext`) without a vendor crate. Shared by the test binaries; each uses a subset.
#![allow(dead_code, reason = "each test binary uses a subset of the table")]

use gpu_async_reactor::backend::cuda::CuStream;
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
/// `cuStreamWaitValue64_v2(stream, addr, value, flags)`.
pub type StreamWaitValue = unsafe extern "C" fn(u64, u64, u64, u32) -> i32;

/// The few driver entry points a context owner uses.
pub struct Driver {
    pub retain: PrimaryRetain,
    pub release: PrimaryRelease,
    pub set_current: CtxSetCurrent,
    pub create: CtxCreate,
    pub destroy: CtxDestroy,
    pub stream_create: StreamCreate,
    pub stream_sync: StreamSync,
    pub stream_destroy: StreamDestroy,
    pub device: i32,
    pub wait_value: StreamWaitValue,
    _lib: Library,
}

pub fn check(what: &str, s: i32) {
    assert_eq!(s, 0, "{what} failed with CUresult {s}");
}

impl Driver {
    pub fn load() -> Self {
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
                wait_value: *lib.get(b"cuStreamWaitValue64_v2").expect("sym"),
                device,
                _lib: lib,
            }
        }
    }

    /// What a primary-context owner does on construction: retain it and make it current.
    pub fn own_primary(&self) -> u64 {
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
    pub fn drop_primary(&self) {
        // WHY: driver calls on plain handles.
        // SAFETY: undoes exactly one `own_primary`.
        // NO SAFE ALTERNATIVE: the driver API is C.
        unsafe {
            check("set current 0", (self.set_current)(0));
            check("release", (self.release)(self.device));
        }
    }

    pub fn stream(&self) -> CuStream {
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

