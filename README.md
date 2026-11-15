# gpu-async-reactor

[![crates.io](https://img.shields.io/crates/v/gpu-async-reactor.svg)](https://crates.io/crates/gpu-async-reactor)
[![docs.rs](https://img.shields.io/docsrs/gpu-async-reactor)](https://docs.rs/gpu-async-reactor)
[![CI](https://github.com/Avarok-Cybersecurity/gpu-async-reactor/actions/workflows/ci.yml/badge.svg)](https://github.com/Avarok-Cybersecurity/gpu-async-reactor/actions/workflows/ci.yml)
[![status: early](https://img.shields.io/badge/status-0.1%20early-orange)](#-status)
[![license: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#-license)
[![MSRV 1.88](https://img.shields.io/badge/MSRV-1.88-informational)](#-msrv)
[![no_std core](https://img.shields.io/badge/core-no__std%20%2B%20alloc-success)](#-design)
[![unsafe: backends only](https://img.shields.io/badge/unsafe-backends%20only-yellow)](#-safety)
[![backends: CUDA | HIP | Metal](https://img.shields.io/badge/backends-CUDA%20%7C%20HIP%20%7C%20Metal-76b900)](#-backends)

**Await GPU work as a memory descriptor.** A GPU stream stores a sequence number into a word of
host-visible memory behind its work. That word is to device completion what a file descriptor is
to I/O readiness: this crate makes it a first-class awaitable source, and supplies the wait plans,
the deadline estimator and a small executor around it, so a host thread can sleep until the GPU is
nearly done, park in hardware on the completion word, and run other ready work in between.

## 📑 Contents

- [🚀 Quick start](#-quick-start)
- [🧭 Design](#-design)
- [🔌 Backends and hosts](#-backends)
- [⚙️ Choosing an executor](#%EF%B8%8F-choosing-an-executor)
- [📊 Benchmarks](#-benchmarks)
- [🛡️ Safety](#%EF%B8%8F-safety)
- [❗ Errors](#-errors)
- [🦀 MSRV](#-msrv)
- [🚧 Status](#-status)
- [⚖️ License](#%EF%B8%8F-license)

## 🚀 Quick start

```toml
[dependencies]
gpu-async-reactor = { version = "0.1", features = ["cuda"] }
```

Wait for a CUDA stream with the plan the Metrale serve A/B picked (park on the completion word,
query the stream every millisecond as the error authority):

```rust,ignore
use gpu_async_reactor::backend::cuda::{CuStream, CudaPort, MappedLine};
use gpu_async_reactor::backend::wordwait::WordWait;
use gpu_async_reactor::host::std_host::StdClock;
use gpu_async_reactor::{Slot, WaitPlan, run_inline, wait};

let plan = WaitPlan::parse("flag:park_us=100,backstop_us=1000")?;
let line = MappedLine::new()?;                       // one 64-byte line, device-mapped, owned by
                                                     // (and retaining) the device's primary context
let stream = unsafe { CuStream::from_raw(raw) };     // your live CUstream, on the primary context
let (clock, mut seq) = (StdClock::new(), 0);
// ... enqueue work on `stream` ...
let mut port = CudaPort::new(&line, stream, Slot::new(0).unwrap(), &mut seq, WordWait::probe().ok());
let report = run_inline(wait(&plan, None, &mut port, &clock, &clock), "my wait")??;
println!("{:?} after {} ns, {} parks", report.kind, report.wait_ns, report.parks);
```

Run several waits and CPU work on one thread (no GPU needed for this one):

```text
cargo run --example fill_executor --features std,wordwait
cargo run --release --example cuda_wait --features cuda      # needs an NVIDIA GPU
```

## 🧭 Design

One crate, organised as **ports and hosts** (SBIO): the core holds every decision and does no
I/O; backends and hosts are generic parameters chosen once by the binary. There is no `dyn` in
the core.

| piece | what it is |
|---|---|
| `CompletionLine` | one 64-byte, 64-aligned line of eight `AtomicU64` completion words. On aarch64 the exclusive-reservation granule is the line, so one parked waiter watches up to eight streams. |
| `MemoryDescriptor` | word `slot` of a line reaching sequence number `target` (wrapping order). The awaitable source. |
| `WaitPlan` | how one wait is spent, parsed once from text: `block` (the vendor's blocking wait), `poll` (status queries) or `flag:park_us=…,backstop_us=…` (park on the word, with a status query every backstop as the error authority), optionally `+predict:margin_us=…,min_sleep_us=…,max_sleep_us=…,window=…`. Every field is required; nothing is defaulted. |
| `WaitBands` | which plan runs, by the rows of the current batch: `<plan>;rows>=32:<plan>`. |
| `PointTable` | the deadline estimator: per readback point, the minimum of the last `window` exact residuals; a wake that finds the work already done is censored downward, so over-sleep corrects itself geometrically. |
| `CompletionPort` + `wait` | the per-backend port (`query`, `block`, `poll`, `arm`, `park`) and the async driver that runs a plan over it. `wait` never returns `Ok` before the port's status query or blocking wait said done: the device word is a wake source, never the authority. |
| `Clock`, `Timer` | the only time sources. `ManualClock` replays any wait from virtual time. |
| `Executor` + `Reactor` | a single-threaded executor over tasks of one future type. When no task is ready, the reactor sleeps to the earliest pending deadline (a predicted completion), then parks on the completion line of the waiting descriptors, bounded by their deadlines. Completions are checked between task polls, so ready work always runs between waits, and the idle time is counted (`RunStats`). |
| `ExecutorKind` + `resolve` | the selectable executor (`naive`, `gpu-reactor`, `io-uring`) with a declared fallback chain, a coded warning, and an opt-in strict refusal. |

The wait, step by step:

```text
block?  ── yes ──▶ port.block()                                   ▶ Blocked
  │ no
query() ── done ─▶                                                 ▶ Immediate (exact residual)
  │ pending
flag? ── arm(): enqueue store(seq) behind the work
predict? ── Timer.sleep_until(start + predicted − margin); query() ── done ─▶ Late (censored)
park: flag ── park(park_us) until the word reaches seq, query() every backstop_us
      poll ── query() until done                                   ▶ Parked (exact residual)
```

## 🔌 Backends

| feature | module | provides | tested on |
|---|---|---|---|
| *(none)* | core | everything above, `no_std` + `alloc`; builds for `wasm32-unknown-unknown` and `wasm32v1-none` | unit tests, virtual clock |
| `wordwait` | `backend::wordwait` | the user-mode park: aarch64 `ldxr` + `wfet` (FEAT_WFxT) or `wfe`; x86-64 `umonitor`/`umwait` (Intel WAITPKG) or `monitorx`/`mwaitx` (AMD) | GB10 (aarch64); x86-64 compile-checked |
| `cuda` | `backend::cuda` | `MappedLine` (page-locked, device-mapped, stored into by `cuStreamWriteValue64`), `CudaPort` | GB10, CUDA 13 |
| `hip` | `backend::hip` | `MappedLine` over `hipHostMalloc(Mapped \| Coherent)` and `hipStreamWriteValue64` | compile-checked (no AMD box yet) |
| `metal` | `backend::metal` | `SharedEventPort`: an `MTLSharedEvent` (`signaledValue`, `waitUntilSignaledValue:timeoutMS:`) | compile-checked for `aarch64-apple-darwin` |

| feature | host | provides |
|---|---|---|
| `std` | `host::std_host` | `StdClock`, `StdIdle` (thread sleep, then the word-wait park or a yield loop), `Watcher` (a thread that wakes `Waker`s; usable from any runtime) |
| `tokio` | `host::tokio` | `TokioClock`: `Clock` + `Timer` over `tokio::time` (1 ms timer wheel: for consumers that are not latency-critical) |
| `io-uring` | `host::uring` | `UringIdle`: the executor's deadline sleep is one `IORING_OP_TIMEOUT` with a nanosecond timespec (no timer-slack rounding; the same ring can carry network I/O) |

Every vendor entry point is resolved by name at run time (`libcuda.so.1`, `libamdhip64.so`), so the
crate links nothing and a machine without the driver fails only the constructor. Adding hardware
means one `LineDriver` table (the CUDA and HIP lines share the ownership logic) or one
`CompletionPort`.

## ⚙️ Choosing an executor

| `ExecutorKind` | the waiting thread | needs | declared fallback |
|---|---|---|---|
| `naive` | runs the wait inline: a blocking predicted sleep, then the plan's park | nothing | none (runs anywhere) |
| `gpu-reactor` | runs the wait as an executor task; idles by sleeping to the earliest deadline, then parking on the line | a user-mode word wait | `naive` |
| `io-uring` | as `gpu-reactor`, with the deadline sleep an `IORING_OP_TIMEOUT` | io_uring and a word wait | `gpu-reactor` |

`resolve(requested, strictness, &caps)` walks the chain once at startup. With
`Strictness::Fallback` an unavailable executor yields the fallback plus a coded
`ExecutorFallback` warning naming what was asked, why it is unavailable, and what runs; with
`Strictness::Strict` it is an `ExecutorRefused` error. `Declared` is the generic shape: implement it
for any other setting with a fallback chain.

## 📊 Benchmarks

All numbers are from the Metrale Engine on an NVIDIA GB10 (Grace Arm + Blackwell, coherent
memory), CUDA 13, serving through the in-tree predecessor of this crate (identical logic). Method:
same-session, rotated boots per arm, clean box (load < 4, no other GPU work), ratios paired to the
spin baseline booted in the same rotation, 95 % CIs (Student-t on paired ratios); GPU energy from
the NVML total-energy counter read twice per cell.

**Wake latency and park cost (micro):**

| measurement | value |
|---|---|
| store → wake of a `ldxr`+`wfe` park (GB10) | 0.3-0.4 µs |
| CUDA host-function → eventfd → `poll` wake (for comparison) | 60-80 µs |
| park calls during a 10 ms wait at a 100 µs bound | ~95 (a `clrex` after each park makes it spin: 160,636; regression test in `wordwait`) |
| Metal (M3 Max): `MTLSharedEvent` wait vs `waitUntilCompleted` | wakes 36-56 µs sooner, +20-25 % waiter CPU |

**Serve A/B, Qwen3.6-35B-A3B FP8, n = 5 rotated boots, ratio to the spinning `cuStreamSynchronize`:**

| C | plan | tok/s | scheduler CPU / token | GPU J / token |
|---|---|---|---|---|
| 4 | `flag` | 1.000 [0.996, 1.004] | 1.000 | 1.000 [0.979, 1.021] |
| 8 | `flag` | 1.001 [0.997, 1.005] | 1.000 | 0.994 [0.984, 1.004] |
| 16 | `flag` | 1.000 [0.999, 1.001] | 0.999 | **0.986 [0.982, 0.990]** |
| 4 | `flag+predict` | 0.995 [0.990, 1.000] | **0.363** | 1.007 |
| 8 | `flag+predict` | 0.997 [0.993, 1.002] | **0.370** | 1.001 |
| 16 | `flag+predict` | 0.998 [0.996, 0.999] | **0.319** | 0.992 |
| 32 | `flag+predict` (n = 3) | 1.000 [0.998, 1.001] | 0.506 | 0.992 [0.985, 0.999] |
| 64 | `flag+predict` (n = 3) | 0.999 [0.996, 1.002] | 0.677 | 0.991 [0.980, 1.002] |

Reading: parking on the word holds throughput parity with a spin; adding the predicted sleep gives
back two thirds of the scheduler core at C4-C16 for a 0.2-0.5 % throughput cost, and is at parity
from C32. Waking from an interrupt or a file descriptor instead lost 0-3 % (35B) and up to 9 %
(27B, C8) and multiplied context switches 9-50x per token. CPU power of a WFE park is not on any
GB10 rail; only the GPU rail is metered. The executor A/B (`naive` vs `gpu-reactor` vs `io-uring`, same
method, C1-C64) measured them equal within noise in tok/s, CPU and J/token, so `naive` is the default.

## 🛡️ Safety

- The core module carries `#![forbid(unsafe_code)]`; the crate root denies `unsafe_code` and only
  the backend modules (and the io_uring host's submission push) allow it. Every `unsafe` block has
  a `SAFETY:` comment; every public function is safe except the two handle constructors below.
- **Handles:** `CuStream::from_raw` and `HipStream::from_raw` are `unsafe fn`: the driver
  dereferences the handle, and a bogus or destroyed one faults inside the driver (CWE-822). Build
  them only from live streams you own.
- **The mapped page outlives every other owner of its context.** A line belongs to the primary
  context of the current context's device and holds its own reference to it
  (`cuDevicePrimaryCtxRetain`; HIP: `hipDevicePrimaryCtxRetain`) from before the page is
  allocated until after it is freed. Another owner releasing the context, or destroying a
  non-primary context that was current when the line was made, cannot free the page. A store is
  accepted only from a stream of that primary context (CUDA) or device (HIP): a stream of a
  `cuCtxCreate` context, which the line cannot keep alive, is refused with `ForeignStream`. `Drop`
  synchronizes the context before freeing the page, then releases it; if that synchronization
  fails, the page and the reference are leaked, never freed under a pending store. The ordering
  is tested against a fake driver with no GPU (`backend::mapped` tests) and on a GB10
  (`tests/cuda_line_lifetime.rs`, `#[ignore]`d without a GPU).
  - **Fixed in 0.1.1 (CWE-416; advisory pending):** 0.1.0 claimed this was excluded by
    construction, and it was not: the line took no reference to its context, so safe code that
    dropped its context owner before the line (for example `cudarc`'s `CudaContext`) left the line
    reading freed memory (SIGSEGV), and a new line could be handed the same page. Upgrade to
    0.1.1.
  - **What remains assumed:** nothing resets the device while a line lives
    (`cuDevicePrimaryCtxReset`, `cudaDeviceReset`, `hipDeviceReset`). A reset frees every
    allocation of every user in the process regardless of references, so a wrapper that exposes it
    as safe is unsound on its own. `cuCtxDestroy` on a retained primary context is refused by the
    driver (measured: status 201, the line stays valid).
- **A word-wait capability cannot be forged.** `WordWait` has private fields; its only
  constructor is `WordWait::probe`, which reads the CPU (on x86-64, only CPUID leaves the CPU
  reports). 0.1.0 had public fields, so safe code could build a `WordWait` claiming an instruction
  the core lacks, and `park` would execute it (SIGILL, CWE-754).
- **The device word is never the authority.** A wait returns `Ok` only after the stream's status
  query or blocking wait said done; a lost or late store costs at most one backstop interval.
- The park reads the line only through atomics and leaves the exclusive monitor armed (a `clrex`
  raises the event and turns every park into a spin).

## ❗ Errors

One append-only `#[repr(u16)]` registry, `ErrorCode`, with a `{}` form template per code; the
arguments are kept and substituted only when the error is displayed. The code is the identity: a
host maps it into its own registry (`Error::code`, `Error::args`) and never branches on the text.

| code | name | meaning |
|---|---|---|
| `0x0001` | `InvalidPlan` | a plan or band text that does not parse (names the text and the fault) |
| `0x0002` | `ParkFallback` | warning: a park the host cannot run, and the declared fallback that runs |
| `0x0003` | `ExecutorFallback` | warning: an executor the host cannot run, and the fallback that runs |
| `0x0004` | `ExecutorRefused` | a strict executor the host cannot run |
| `0x0005` | `InvalidExecutor` | an unknown executor name |
| `0x0006` | `PendedInline` | a future pended on an inline host (a bug, never a hang) |
| `0x0007` | `Stalled` | tasks remain and nothing is registered that could wake them |
| `0x0008` | `Backend` | a vendor call failed (names the call and status) |

## 🦀 MSRV

Rust **1.88** (edition 2024, `let` chains). The MSRV is raised only in a minor release.

## 🚧 Status

`0.1.1`, early release. The core, the CUDA backend and the aarch64 word wait run in production in the
Metrale Engine on GB10; HIP, x86-64 word waits and Metal are compile-checked, awaiting hardware
runs.

## ⚖️ License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option. Unless you explicitly state otherwise, any contribution
intentionally submitted for inclusion in this crate by you, as defined in the Apache-2.0 license,
shall be dual licensed as above, without any additional terms or conditions.
