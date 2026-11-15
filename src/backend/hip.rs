// SPDX-License-Identifier: MIT OR Apache-2.0

//! HIP (AMD): a [`MappedLine`] that HIP streams store into with `hipStreamWriteValue64` into
//! host-mapped, coherent memory (`hipHostMalloc(Mapped | Coherent)`), so a host thread waits on
//! it with the x86 `mwaitx`/`umwait` (or aarch64 `wfe`) park of the `wordwait` backend. The
//! domain is the HIP device: the line holds a reference to the device's primary context
//! (`hipDevicePrimaryCtxRetain`) from before the page is allocated until after it is freed, a
//! store is accepted only from a stream of that device, and `Drop` synchronizes the device before
//! freeing the page.
//!
//! Entry points are resolved by name at run time from `libamdhip64.so`; compile-checked, not yet
//! run on AMD hardware.

use core::ffi::{c_int, c_void};
use core::ptr::NonNull;

use libloading::Library;

use super::mapped::{Line, LineDriver, LineError};
use crate::core::line::{CompletionLine, Slot};

/// A `hipStream_t` handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HipStream(usize);

impl HipStream {
    /// Wrap a raw `hipStream_t`.
    ///
    /// # Why
    /// the typed stream is the only handle the reactor waits on, and the driver hands streams out as bare integers.
    ///
    /// # Safety
    /// `raw` must be a live `hipStream_t` (or null, the default stream) for every call made with
    /// the value: the runtime dereferences it.
    ///
    /// # No safe alternative
    /// a handle is an integer the crate cannot validate; the typed stream is built only from it, so the constructor is unsafe.
    #[must_use]
    pub const unsafe fn from_raw(raw: usize) -> Self {
        Self(raw)
    }

    /// The raw handle.
    #[must_use]
    pub const fn raw(self) -> usize {
        self.0
    }
}

/// `hipHostMallocMapped`.
const MAPPED: u32 = 0x2;
/// `hipHostMallocCoherent`: CPU and device see each other's writes without a flush.
const COHERENT: u32 = 0x4000_0000;

type HostMalloc = unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> i32;
type HostDevPtr = unsafe extern "C" fn(*mut *mut c_void, *mut c_void, u32) -> i32;
type HostFree = unsafe extern "C" fn(*mut c_void) -> i32;
type GetDevice = unsafe extern "C" fn(*mut c_int) -> i32;
type SetDevice = unsafe extern "C" fn(c_int) -> i32;
type DevSync = unsafe extern "C" fn() -> i32;
type StreamDevice = unsafe extern "C" fn(*mut c_void, *mut c_int) -> i32;
type PrimaryRetain = unsafe extern "C" fn(*mut *mut c_void, c_int) -> i32;
type PrimaryRelease = unsafe extern "C" fn(c_int) -> i32;
type WriteValue64 = unsafe extern "C" fn(*mut c_void, *mut c_void, u64, u32) -> i32;

/// The HIP runtime's table and the library handle that keeps it valid.
pub(crate) struct HipDriver {
    malloc: HostMalloc,
    dev_ptr: HostDevPtr,
    free: HostFree,
    get_device: GetDevice,
    set_device: SetDevice,
    dev_sync: DevSync,
    stream_device: StreamDevice,
    retain: PrimaryRetain,
    release: PrimaryRelease,
    write: WriteValue64,
    _lib: Library,
}

macro_rules! sym {
    ($lib:ident, $ty:ty, $name:literal) => {
        // WHY: resolves one driver entry point by name, to call it through a typed fn pointer.
        // SAFETY: the type is the signature hip_runtime_api.h (ROCm 6) declares for this name:
        // `hipError_t` is a C enum read as `i32`, handles and device pointers are pointers,
        // `hipDevice_t` is `int`; the name is NUL-free ASCII.
        // NO SAFE ALTERNATIVE: a symbol lookup asserts a foreign signature the compiler cannot check; `libloading::Library::get` is unsafe.
        *unsafe { $lib.get::<$ty>($name) }.ok()?
    };
}

impl HipDriver {
    /// Every entry point, or `None` if the runtime or any of them is missing.
    pub(crate) fn resolve() -> Option<Self> {
        // WHY: loads the vendor library at run time, so the crate links no vendor SDK and builds on a host without one.
        // SAFETY: loading the HIP runtime runs its initialisers, which have no preconditions on
        // this process; the handle is kept for as long as the table.
        // NO SAFE ALTERNATIVE: dlopen runs foreign initialisers; `libloading::Library::new` is unsafe and there is no checked loader.
        let lib = unsafe { Library::new("libamdhip64.so") }.ok()?;
        Some(Self {
            malloc: sym!(lib, HostMalloc, b"hipHostMalloc"),
            dev_ptr: sym!(lib, HostDevPtr, b"hipHostGetDevicePointer"),
            free: sym!(lib, HostFree, b"hipHostFree"),
            get_device: sym!(lib, GetDevice, b"hipGetDevice"),
            set_device: sym!(lib, SetDevice, b"hipSetDevice"),
            dev_sync: sym!(lib, DevSync, b"hipDeviceSynchronize"),
            stream_device: sym!(lib, StreamDevice, b"hipStreamGetDevice"),
            retain: sym!(lib, PrimaryRetain, b"hipDevicePrimaryCtxRetain"),
            release: sym!(lib, PrimaryRelease, b"hipDevicePrimaryCtxRelease"),
            write: sym!(lib, WriteValue64, b"hipStreamWriteValue64"),
            _lib: lib,
        })
    }
}

fn ok(status: i32) -> Result<(), LineError> {
    if status == 0 { Ok(()) } else { Err(LineError::Status(status)) }
}

/// A device whose primary context the line holds a reference to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HipDevice(c_int);

impl HipDriver {
    /// Run `f` with `device` current on the calling thread, then restore the thread's device.
    fn on_device<T>(&self, device: HipDevice, f: impl FnOnce() -> T) -> Result<T, LineError> {
        let mut prev: c_int = 0;
        // WHY: calls a HIP runtime entry point resolved at run time (the line, stream or context operation).
        // SAFETY: plain runtime calls on integers; `prev` is a live out-pointer.
        // NO SAFE ALTERNATIVE: the driver API is C: every entry point is an unsafe fn pointer, and the vendor crates' wrappers are unsafe too.
        unsafe {
            ok((self.get_device)(&raw mut prev))?;
            ok((self.set_device)(device.0))?;
        }
        let r = f();
        // WHY: calls a HIP runtime entry point resolved at run time (the line, stream or context operation).
        // SAFETY: restores the device read above.
        // NO SAFE ALTERNATIVE: the driver API is C: every entry point is an unsafe fn pointer, and the vendor crates' wrappers are unsafe too.
        ok(unsafe { (self.set_device)(prev) })?;
        Ok(r)
    }
}

impl LineDriver for HipDriver {
    type Stream = HipStream;
    type Domain = HipDevice;

    fn retain_current(&self) -> Result<HipDevice, LineError> {
        let mut d: c_int = -1;
        // WHY: calls a HIP runtime entry point resolved at run time (the line, stream or context operation).
        // SAFETY: `d` is a live out-pointer, the only memory written.
        // NO SAFE ALTERNATIVE: the driver API is C: every entry point is an unsafe fn pointer, and the vendor crates' wrappers are unsafe too.
        ok(unsafe { (self.get_device)(&raw mut d) })?;
        if d < 0 {
            return Err(LineError::NoDomain);
        }
        let mut ctx: *mut c_void = core::ptr::null_mut();
        // WHY: calls a HIP runtime entry point resolved at run time (the line, stream or context operation).
        // SAFETY: `d` is the current device; `ctx` is a live out-pointer. The reference taken here
        // is dropped once, by `release` (or leaked with the page).
        // NO SAFE ALTERNATIVE: the driver API is C: every entry point is an unsafe fn pointer, and the vendor crates' wrappers are unsafe too.
        ok(unsafe { (self.retain)(&raw mut ctx, d) })?;
        Ok(HipDevice(d))
    }

    fn release(&self, device: HipDevice) {
        // WHY: calls a HIP runtime entry point resolved at run time (the line, stream or context operation).
        // SAFETY: drops the one reference `retain_current` took on this device's primary context.
        // NO SAFE ALTERNATIVE: the driver API is C: every entry point is an unsafe fn pointer, and the vendor crates' wrappers are unsafe too.
        let _ = unsafe { (self.release)(device.0) };
    }

    fn alloc_host(&self, device: HipDevice, bytes: usize) -> Result<NonNull<c_void>, LineError> {
        let mut page: *mut c_void = core::ptr::null_mut();
        // WHY: calls a HIP runtime entry point resolved at run time (the line, stream or context operation).
        // SAFETY: `page` is a live out-pointer, the only memory written; the line's device is
        // current for the call.
        // NO SAFE ALTERNATIVE: the driver API is C: every entry point is an unsafe fn pointer, and the vendor crates' wrappers are unsafe too.
        ok(self.on_device(device, || unsafe {
            (self.malloc)(&raw mut page, bytes, MAPPED | COHERENT)
        })?)?;
        NonNull::new(page).ok_or(LineError::Unavailable)
    }

    fn device_address(&self, device: HipDevice, host: NonNull<c_void>) -> Result<u64, LineError> {
        let mut addr: *mut c_void = core::ptr::null_mut();
        // WHY: calls a HIP runtime entry point resolved at run time (the line, stream or context operation).
        // SAFETY: `host` is a live mapped page from `alloc_host`; `addr` is a live out-pointer.
        // NO SAFE ALTERNATIVE: the driver API is C: every entry point is an unsafe fn pointer, and the vendor crates' wrappers are unsafe too.
        ok(self.on_device(device, || unsafe {
            (self.dev_ptr)(&raw mut addr, host.as_ptr(), 0)
        })?)?;
        Ok(addr as u64)
    }

    fn free_host(&self, _device: HipDevice, host: NonNull<c_void>) {
        // WHY: calls a HIP runtime entry point resolved at run time (the line, stream or context operation).
        // SAFETY: `host` came from `alloc_host` and is freed once, after every store landed, while
        // the line still holds its device's primary context.
        // NO SAFE ALTERNATIVE: the driver API is C: every entry point is an unsafe fn pointer, and the vendor crates' wrappers are unsafe too.
        let _ = unsafe { (self.free)(host.as_ptr()) };
    }

    fn stream_in(&self, stream: HipStream, device: HipDevice) -> Result<bool, LineError> {
        let mut d: c_int = -1;
        // WHY: calls a HIP runtime entry point resolved at run time (the line, stream or context operation).
        // SAFETY: `stream` is live by `HipStream::from_raw`'s contract; `d` is a live out-pointer.
        // NO SAFE ALTERNATIVE: the driver API is C: every entry point is an unsafe fn pointer, and the vendor crates' wrappers are unsafe too.
        ok(unsafe { (self.stream_device)(stream.0 as *mut c_void, &raw mut d) })?;
        Ok(d == device.0)
    }

    fn write_value(&self, stream: HipStream, addr: u64, value: u64) -> Result<(), LineError> {
        let ptr = usize::try_from(addr).map_err(|_| LineError::Unavailable)? as *mut c_void;
        // WHY: calls a HIP runtime entry point resolved at run time (the line, stream or context operation).
        // SAFETY: `stream` is live and of the line's device (checked by the line); `ptr` is a word
        // of the line's own live mapped page, which `Drop` frees only after the device settled.
        // NO SAFE ALTERNATIVE: the driver API is C: every entry point is an unsafe fn pointer, and the vendor crates' wrappers are unsafe too.
        ok(unsafe { (self.write)(stream.0 as *mut c_void, ptr, value, 0) })
    }

    fn settle(&self, device: HipDevice) -> bool {
        // WHY: calls a HIP runtime entry point resolved at run time (the line, stream or context operation).
        // SAFETY: a plain runtime call, with the line's device current for it.
        // NO SAFE ALTERNATIVE: the driver API is C: every entry point is an unsafe fn pointer, and the vendor crates' wrappers are unsafe too.
        matches!(self.on_device(device, || unsafe { (self.dev_sync)() }), Ok(0))
    }
}

/// A page-locked, host-mapped, coherent [`CompletionLine`] that HIP streams store into.
pub struct MappedLine(Line<HipDriver>);

impl MappedLine {
    /// A fresh line reading all 0, mapped for the calling thread's current device. The line holds
    /// a reference to that device's primary context until after its page is freed.
    ///
    /// # Errors
    /// The runtime's status, [`LineError::Unavailable`] when the runtime or an entry point is
    /// missing, [`LineError::NoDomain`] when no device is current.
    pub fn new() -> Result<Self, LineError> {
        Line::with_driver(HipDriver::resolve().ok_or(LineError::Unavailable)?).map(Self)
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
    /// The runtime's status, or [`LineError::ForeignStream`] for a stream of another device.
    pub fn enqueue_store(&self, stream: HipStream, slot: Slot, value: u64) -> Result<(), LineError> {
        self.0.enqueue_store(stream, slot, value)
    }
}
