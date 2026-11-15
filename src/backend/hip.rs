// SPDX-License-Identifier: MIT OR Apache-2.0

//! HIP (AMD): a [`MappedLine`] that HIP streams store into with `hipStreamWriteValue64` into
//! host-mapped, coherent memory (`hipHostMalloc(Mapped | Coherent)`), so a host thread waits on
//! it with the x86 `mwaitx`/`umwait` (or aarch64 `wfe`) park of the `wordwait` backend. The
//! domain is the HIP device: a store is accepted only from a stream of the device the line was
//! made on, and `Drop` synchronizes that device before freeing the page.
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
            write: sym!(lib, WriteValue64, b"hipStreamWriteValue64"),
            _lib: lib,
        })
    }
}

fn ok(status: i32) -> Result<(), LineError> {
    if status == 0 { Ok(()) } else { Err(LineError::Status(status)) }
}

/// Device ids are stored biased by one, so a line's domain is never 0 ("no domain").
fn domain_of(device: c_int) -> Result<u64, LineError> {
    u64::try_from(device)
        .map(|d| d + 1)
        .map_err(|_| LineError::NoDomain)
}

impl LineDriver for HipDriver {
    type Stream = HipStream;

    fn domain_current(&self) -> Result<u64, LineError> {
        let mut d: c_int = -1;
        // WHY: calls a HIP runtime entry point resolved at run time (the line, stream or context operation).
        // SAFETY: `d` is a live out-pointer, the only memory written.
        // NO SAFE ALTERNATIVE: the driver API is C: every entry point is an unsafe fn pointer, and the vendor crates' wrappers are unsafe too.
        ok(unsafe { (self.get_device)(&raw mut d) })?;
        domain_of(d)
    }

    fn alloc_host(&self, bytes: usize) -> Result<NonNull<c_void>, LineError> {
        let mut page: *mut c_void = core::ptr::null_mut();
        // WHY: calls a HIP runtime entry point resolved at run time (the line, stream or context operation).
        // SAFETY: `page` is a live out-pointer, the only memory written.
        // NO SAFE ALTERNATIVE: the driver API is C: every entry point is an unsafe fn pointer, and the vendor crates' wrappers are unsafe too.
        ok(unsafe { (self.malloc)(&raw mut page, bytes, MAPPED | COHERENT) })?;
        NonNull::new(page).ok_or(LineError::Unavailable)
    }

    fn device_address(&self, host: NonNull<c_void>) -> Result<u64, LineError> {
        let mut device: *mut c_void = core::ptr::null_mut();
        // WHY: calls a HIP runtime entry point resolved at run time (the line, stream or context operation).
        // SAFETY: `host` is a live mapped page from `alloc_host`; `device` is a live out-pointer.
        // NO SAFE ALTERNATIVE: the driver API is C: every entry point is an unsafe fn pointer, and the vendor crates' wrappers are unsafe too.
        ok(unsafe { (self.dev_ptr)(&raw mut device, host.as_ptr(), 0) })?;
        Ok(device as u64)
    }

    fn free_host(&self, host: NonNull<c_void>) {
        // WHY: calls a HIP runtime entry point resolved at run time (the line, stream or context operation).
        // SAFETY: `host` came from `alloc_host` and is freed once, after every store landed.
        // NO SAFE ALTERNATIVE: the driver API is C: every entry point is an unsafe fn pointer, and the vendor crates' wrappers are unsafe too.
        let _ = unsafe { (self.free)(host.as_ptr()) };
    }

    fn stream_domain(&self, stream: HipStream) -> Result<u64, LineError> {
        let mut d: c_int = -1;
        // WHY: calls a HIP runtime entry point resolved at run time (the line, stream or context operation).
        // SAFETY: `stream` is live by `HipStream::from_raw`'s contract; `d` is a live out-pointer.
        // NO SAFE ALTERNATIVE: the driver API is C: every entry point is an unsafe fn pointer, and the vendor crates' wrappers are unsafe too.
        ok(unsafe { (self.stream_device)(stream.0 as *mut c_void, &raw mut d) })?;
        domain_of(d)
    }

    fn write_value(&self, stream: HipStream, addr: u64, value: u64) -> Result<(), LineError> {
        let ptr = usize::try_from(addr).map_err(|_| LineError::Unavailable)? as *mut c_void;
        // WHY: calls a HIP runtime entry point resolved at run time (the line, stream or context operation).
        // SAFETY: `stream` is live and of the line's device (checked by the line); `ptr` is a word
        // of the line's own live mapped page, which `Drop` frees only after the device settled.
        // NO SAFE ALTERNATIVE: the driver API is C: every entry point is an unsafe fn pointer, and the vendor crates' wrappers are unsafe too.
        ok(unsafe { (self.write)(stream.0 as *mut c_void, ptr, value, 0) })
    }

    fn settle(&self, domain: u64) -> bool {
        let Some(device) = domain.checked_sub(1).and_then(|d| c_int::try_from(d).ok()) else {
            return false;
        };
        let mut prev: c_int = 0;
        // WHY: calls a HIP runtime entry point resolved at run time (the line, stream or context operation).
        // SAFETY: plain runtime calls on integers; `prev` is a live out-pointer. The thread's
        // current device is restored before returning.
        // NO SAFE ALTERNATIVE: the driver API is C: every entry point is an unsafe fn pointer, and the vendor crates' wrappers are unsafe too.
        unsafe {
            if (self.get_device)(&raw mut prev) != 0 || (self.set_device)(device) != 0 {
                return false;
            }
            let synced = (self.dev_sync)() == 0;
            let restored = (self.set_device)(prev) == 0;
            synced && restored
        }
    }
}

/// A page-locked, host-mapped, coherent [`CompletionLine`] that HIP streams store into.
pub struct MappedLine(Line<HipDriver>);

impl MappedLine {
    /// A fresh line reading all 0, mapped for the calling thread's current device.
    ///
    /// # Errors
    /// The runtime's status, or [`LineError::Unavailable`] when the runtime or an entry point is
    /// missing.
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
