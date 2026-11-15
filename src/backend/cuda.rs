// SPDX-License-Identifier: MIT OR Apache-2.0

//! CUDA: a [`MappedLine`] that streams store into with `cuStreamWriteValue64`, and [`CudaPort`],
//! a [`CompletionPort`] over one stream (status by `cuStreamQuery`, the vendor wait by
//! `cuStreamSynchronize`, the park on the line through the `wordwait` backend).
//!
//! Every driver entry point is resolved by name at run time from `libcuda.so.1` (the process's
//! already-loaded driver when there is one), so the crate links nothing and a driver without
//! `cuStreamWriteValue64_v2` fails only [`MappedLine::new`].

use core::ffi::c_void;
use core::ptr::NonNull;
use core::time::Duration;

use libloading::Library;

use super::mapped::{Line, LineDriver, LineError};
use crate::core::line::{CompletionLine, Slot};
use crate::core::time::DurNs;
use crate::core::wait::CompletionPort;

/// A `CUstream` handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CuStream(u64);

impl CuStream {
    /// Wrap a raw `CUstream`.
    ///
    /// # Why
    /// the typed stream is the only handle the reactor waits on, and the driver hands streams out as bare integers.
    ///
    /// # Safety
    /// `raw` must be a live `CUstream` (or 0, the legacy default stream) of a live context for
    /// every call made with the value: the driver dereferences it, and a bogus or destroyed handle
    /// faults inside the driver (CWE-822).
    ///
    /// # No safe alternative
    /// a handle is an integer the crate cannot validate; the typed stream is built only from it, so the constructor is unsafe.
    #[must_use]
    pub const unsafe fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// The raw handle.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

type AllocHost = unsafe extern "C" fn(*mut *mut c_void, usize) -> i32;
type HostDevPtr = unsafe extern "C" fn(*mut u64, *mut c_void, u32) -> i32;
type FreeHost = unsafe extern "C" fn(*mut c_void) -> i32;
type CtxGet = unsafe extern "C" fn(*mut u64) -> i32;
type WriteValue64 = unsafe extern "C" fn(u64, u64, u64, u32) -> i32;
type CtxPush = unsafe extern "C" fn(u64) -> i32;
type CtxPop = unsafe extern "C" fn(*mut u64) -> i32;
type CtxSync = unsafe extern "C" fn() -> i32;
type StreamCtx = unsafe extern "C" fn(u64, *mut u64) -> i32;
type StreamQuery = unsafe extern "C" fn(u64) -> i32;
type StreamSync = unsafe extern "C" fn(u64) -> i32;

/// `CUDA_ERROR_NOT_READY`.
const NOT_READY: i32 = 600;

/// The CUDA driver's table and the library handle that keeps it valid.
pub(crate) struct CudaDriver {
    alloc: AllocHost,
    dev_ptr: HostDevPtr,
    free: FreeHost,
    ctx_get: CtxGet,
    write: WriteValue64,
    push: CtxPush,
    pop: CtxPop,
    sync: CtxSync,
    stream_ctx: StreamCtx,
    query: StreamQuery,
    stream_sync: StreamSync,
    _lib: Library,
}

macro_rules! sym {
    ($lib:ident, $ty:ty, $name:literal) => {
        // WHY: resolves one driver entry point by name, to call it through a typed fn pointer.
        // SAFETY: the type is the signature cuda.h (13.0) declares for this name: `CUresult` is a
        // C enum read as `i32`, `CUdeviceptr` and the pointer-sized handles are `u64`; the name is
        // NUL-free ASCII.
        // NO SAFE ALTERNATIVE: a symbol lookup asserts a foreign signature the compiler cannot check; `libloading::Library::get` is unsafe.
        *unsafe { $lib.get::<$ty>($name) }.ok()?
    };
}

impl CudaDriver {
    /// Every entry point, or `None` if the driver or any of them is missing.
    pub(crate) fn resolve() -> Option<Self> {
        // WHY: loads the vendor library at run time, so the crate links no vendor SDK and builds on a host without one.
        // SAFETY: loading the CUDA driver runs its initialisers, which have no preconditions on
        // this process; the handle is kept for as long as the table.
        // NO SAFE ALTERNATIVE: dlopen runs foreign initialisers; `libloading::Library::new` is unsafe and there is no checked loader.
        let lib = unsafe { Library::new("libcuda.so.1") }.ok()?;
        Some(Self {
            alloc: sym!(lib, AllocHost, b"cuMemAllocHost_v2"),
            dev_ptr: sym!(lib, HostDevPtr, b"cuMemHostGetDevicePointer_v2"),
            free: sym!(lib, FreeHost, b"cuMemFreeHost"),
            ctx_get: sym!(lib, CtxGet, b"cuCtxGetCurrent"),
            write: sym!(lib, WriteValue64, b"cuStreamWriteValue64_v2"),
            push: sym!(lib, CtxPush, b"cuCtxPushCurrent_v2"),
            pop: sym!(lib, CtxPop, b"cuCtxPopCurrent_v2"),
            sync: sym!(lib, CtxSync, b"cuCtxSynchronize"),
            stream_ctx: sym!(lib, StreamCtx, b"cuStreamGetCtx"),
            query: sym!(lib, StreamQuery, b"cuStreamQuery"),
            stream_sync: sym!(lib, StreamSync, b"cuStreamSynchronize"),
            _lib: lib,
        })
    }
}

fn ok(status: i32) -> Result<(), LineError> {
    if status == 0 { Ok(()) } else { Err(LineError::Status(status)) }
}

impl LineDriver for CudaDriver {
    type Stream = CuStream;

    fn domain_current(&self) -> Result<u64, LineError> {
        let mut ctx = 0u64;
        // WHY: calls a CUDA driver entry point resolved at run time (the line, stream or context operation).
        // SAFETY: `ctx` is a live out-pointer, the only memory written.
        // NO SAFE ALTERNATIVE: the driver API is C: every entry point is an unsafe fn pointer, and the vendor crates' wrappers are unsafe too.
        ok(unsafe { (self.ctx_get)(&raw mut ctx) })?;
        Ok(ctx)
    }

    fn alloc_host(&self, bytes: usize) -> Result<NonNull<c_void>, LineError> {
        let mut page: *mut c_void = core::ptr::null_mut();
        // WHY: calls a CUDA driver entry point resolved at run time (the line, stream or context operation).
        // SAFETY: `page` is a live out-pointer, the only memory the driver writes; without a
        // current context the driver returns a status instead.
        // NO SAFE ALTERNATIVE: the driver API is C: every entry point is an unsafe fn pointer, and the vendor crates' wrappers are unsafe too.
        ok(unsafe { (self.alloc)(&raw mut page, bytes) })?;
        NonNull::new(page).ok_or(LineError::Unavailable)
    }

    fn device_address(&self, host: NonNull<c_void>) -> Result<u64, LineError> {
        let mut device = 0u64;
        // WHY: calls a CUDA driver entry point resolved at run time (the line, stream or context operation).
        // SAFETY: `host` is a live page from `alloc_host`; `device` is a live out-pointer.
        // NO SAFE ALTERNATIVE: the driver API is C: every entry point is an unsafe fn pointer, and the vendor crates' wrappers are unsafe too.
        ok(unsafe { (self.dev_ptr)(&raw mut device, host.as_ptr(), 0) })?;
        Ok(device)
    }

    fn free_host(&self, host: NonNull<c_void>) {
        // WHY: calls a CUDA driver entry point resolved at run time (the line, stream or context operation).
        // SAFETY: `host` came from `alloc_host`, is freed once (the line's `Drop`), and every
        // store queued into it has landed.
        // NO SAFE ALTERNATIVE: the driver API is C: every entry point is an unsafe fn pointer, and the vendor crates' wrappers are unsafe too.
        let _ = unsafe { (self.free)(host.as_ptr()) };
    }

    fn stream_domain(&self, stream: CuStream) -> Result<u64, LineError> {
        let mut owner = 0u64;
        // WHY: calls a CUDA driver entry point resolved at run time (the line, stream or context operation).
        // SAFETY: `stream` is live by `CuStream::from_raw`'s contract; `owner` is a live
        // out-pointer, the only memory written.
        // NO SAFE ALTERNATIVE: the driver API is C: every entry point is an unsafe fn pointer, and the vendor crates' wrappers are unsafe too.
        ok(unsafe { (self.stream_ctx)(stream.0, &raw mut owner) })?;
        Ok(owner)
    }

    fn write_value(&self, stream: CuStream, addr: u64, value: u64) -> Result<(), LineError> {
        // WHY: calls a CUDA driver entry point resolved at run time (the line, stream or context operation).
        // SAFETY: `stream` is live (as above) and of the line's context (checked by the line);
        // `addr` is a word of the line's own live device-mapped page, so `Drop` settles the
        // context before the page is freed.
        // NO SAFE ALTERNATIVE: the driver API is C: every entry point is an unsafe fn pointer, and the vendor crates' wrappers are unsafe too.
        ok(unsafe { (self.write)(stream.0, addr, value, 0) })
    }

    fn settle(&self, ctx: u64) -> bool {
        // WHY: calls a CUDA driver entry point resolved at run time (the line, stream or context operation).
        // SAFETY: `ctx` is the context the line was made in; a destroyed one is refused with a
        // status. Sync waits on the pushed context; pop restores the thread's stack and writes
        // only `popped`.
        // NO SAFE ALTERNATIVE: the driver API is C: every entry point is an unsafe fn pointer, and the vendor crates' wrappers are unsafe too.
        unsafe {
            if (self.push)(ctx) != 0 {
                return false;
            }
            let synced = (self.sync)() == 0;
            let mut popped = 0u64;
            let popped_ok = (self.pop)(&raw mut popped) == 0;
            synced && popped_ok
        }
    }
}

/// A page-locked, device-mapped [`CompletionLine`] that CUDA streams store into.
pub struct MappedLine(Line<CudaDriver>);

impl MappedLine {
    /// A fresh line reading all 0, mapped for the calling thread's current context.
    ///
    /// # Errors
    /// The driver's status, [`LineError::Unavailable`] when the driver or an entry point is
    /// missing, [`LineError::NoDomain`] when no context is current.
    pub fn new() -> Result<Self, LineError> {
        Line::with_driver(CudaDriver::resolve().ok_or(LineError::Unavailable)?).map(Self)
    }

    /// The line, as the host reads it.
    #[must_use]
    pub fn line(&self) -> &CompletionLine {
        self.0.line()
    }

    /// The device address of word 0; word `i` is at `+ 8 * i` (for a stream-side wait on it).
    #[must_use]
    pub fn device_address(&self) -> u64 {
        self.0.device_address()
    }

    /// Queue a store of `value` into word `slot` on `stream`, behind the work queued there.
    ///
    /// # Errors
    /// The driver's status, or [`LineError::ForeignStream`] for a stream of another context.
    pub fn enqueue_store(&self, stream: CuStream, slot: Slot, value: u64) -> Result<(), LineError> {
        self.0.enqueue_store(stream, slot, value)
    }

    fn driver(&self) -> &CudaDriver {
        self.0.driver()
    }
}

/// A [`CompletionPort`] over one CUDA stream and a slot of a [`MappedLine`].
pub struct CudaPort<'a> {
    line: &'a MappedLine,
    stream: CuStream,
    slot: Slot,
    seq: &'a mut u64,
    wait: Option<super::wordwait::WordWait>,
}

impl<'a> CudaPort<'a> {
    /// The port for `stream`, storing into `slot` of `line`; `seq` is that slot's last sequence
    /// number (kept by the caller across waits). Without `wait` a park polls the word.
    pub fn new(
        line: &'a MappedLine,
        stream: CuStream,
        slot: Slot,
        seq: &'a mut u64,
        wait: Option<super::wordwait::WordWait>,
    ) -> Self {
        Self {
            line,
            stream,
            slot,
            seq,
            wait,
        }
    }
}

impl CompletionPort for CudaPort<'_> {
    type Error = crate::Error;

    fn query(&mut self) -> crate::Result<bool> {
        // WHY: calls a CUDA driver entry point resolved at run time (the line, stream or context operation).
        // SAFETY: the stream is live by `CuStream::from_raw`'s contract.
        // NO SAFE ALTERNATIVE: the driver API is C: every entry point is an unsafe fn pointer, and the vendor crates' wrappers are unsafe too.
        match unsafe { (self.line.driver().query)(self.stream.0) } {
            0 => Ok(true),
            NOT_READY => Ok(false),
            s => Err(crate::error!(Backend, "cuStreamQuery", s)),
        }
    }

    async fn block(&mut self) -> crate::Result<()> {
        // WHY: calls a CUDA driver entry point resolved at run time (the line, stream or context operation).
        // SAFETY: as `query`.
        // NO SAFE ALTERNATIVE: the driver API is C: every entry point is an unsafe fn pointer, and the vendor crates' wrappers are unsafe too.
        match unsafe { (self.line.driver().stream_sync)(self.stream.0) } {
            0 => Ok(()),
            s => Err(crate::error!(Backend, "cuStreamSynchronize", s)),
        }
    }

    async fn poll(&mut self) -> crate::Result<()> {
        while !self.query()? {
            core::hint::spin_loop();
        }
        Ok(())
    }

    fn arm(&mut self) -> crate::Result<()> {
        *self.seq = self.seq.wrapping_add(1);
        Ok(self.line.enqueue_store(self.stream, self.slot, *self.seq)?)
    }

    async fn park(&mut self, bound: DurNs) -> crate::Result<bool> {
        let (slot, seq) = (self.slot, *self.seq);
        let ready = |l: &CompletionLine| crate::core::line::reached(l, slot, seq);
        Ok(match &self.wait {
            Some(w) => w.park(self.line.line(), ready, Duration::from_nanos(bound.0)),
            None => ready(self.line.line()),
        })
    }
}
