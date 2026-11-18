# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the crate follows
[Semantic Versioning](https://semver.org/).

## [0.1.2] - 2026-11-18

### Fixed
- **CWE-833, a `Watch` that outlived its `Watcher` hung (`std`).** Dropping the watcher stopped
  and joined its thread at once, so a `Watch` still pending, or polled for the first time
  afterwards, registered into a table nobody scanned and its waker never fired. The thread now
  serves every `Watch` it handed out and exits when the last one resolves or is dropped; with no
  `Watch` alive, dropping the watcher still joins it.
- **CWE-672 / CWE-670, `UringIdle` returned early after an interrupted wait (`io-uring`).** A
  signal during `submit_and_wait` left the timeout in flight, and every later sleep consumed the
  previous sleep's completion and returned at once, so an executor on it spun to each deadline.
  Each timeout now carries its own `user_data`; a sleep waits again after `EINTR` and returns
  only on its own completion, and completions of earlier timeouts are discarded. An interrupted
  wait no longer counts as a failed timeout.
- **`CudaPort::poll` spun inside an `async fn` (`cuda`).** Under the crate's `Executor` a task
  polling a stream (`Park::QueryPoll`, the tail of a flag wait, or a flag wait with no word wait)
  never pended, so it starved every other ready task, including one the GPU work waited on.

### Added
- `Pace` (`Spin` | `Yield`): what a port does between two status queries and in a park with no
  hardware wait, declared by the host. `Pace::step` is public for ports outside the crate.
- `CudaPort::paced(.., pace)`, `CudaPort::pace()` and `CudaPort::parks_in_hardware()` (the
  no-word-wait mode is now declared and reported).

### Deprecated
- `CudaPort::new`: it is `CudaPort::paced(.., Pace::Spin)`, the 0.1.1 behaviour, right only for
  an inline host. Say which host it is with `paced`.

### Notes
- `metal::SharedEventPort` is unchanged: its `block` and `park` sleep the calling thread by
  design (`waitUntilSignaledValue`), so it is an inline-host port and its poll spins.

## [0.1.1] - 2026-11-15

### Security
- **CWE-416, use-after-free from safe code (`cuda`, `hip`).** A `MappedLine` took no reference
  to the context its page was allocated in, so safe code that let that context go first (for
  example by dropping a `cudarc::CudaContext`) left the line reading freed memory (SIGSEGV), and
  a second line could be handed the same page (cross-object corruption). A line now belongs to the
  primary context of the current context's device and holds its own reference to it
  (`cuDevicePrimaryCtxRetain` / `hipDevicePrimaryCtxRetain`) until after the page is freed. The
  README's claim that this was excluded by construction was wrong and is corrected.
- **CWE-754, forgeable CPU capability (`wordwait`).** `WordWait`'s fields were public, so safe
  code could build one claiming `wfet`, `umwait` or `mwaitx` on a CPU without them and `park`
  would execute the instruction (SIGILL). The fields are private; `WordWait::probe` is the only
  constructor. The x86-64 probe now reads CPUID leaf 7 and 0x8000_0001 only when the CPU reports
  them.
- **CWE-401 / CWE-400 / CWE-833, `Watcher` (`std`).** A dropped `Watch` (a cancelled await) was
  never removed: 100,000 cancellations kept 100,001 references and the watcher thread at a full
  core. Wakers were called under the table lock, so a waker that touched the same watcher
  deadlocked it. A `Watch` now removes its registration when it resolves or is dropped, wakers
  run with no lock held, and two `Watch`es on one word are both woken (the first was lost).

### Changed
- `cuda::MappedLine` accepts stores only from streams of the device's primary context; a stream of
  a non-primary (`cuCtxCreate`) context is refused with `LineError::ForeignStream`. The
  `cuda_wait` example now uses the primary context.
- `WordWait` fields are read through `timed()` / `counter_hz()` (aarch64) and `flavor()` /
  `tsc_hz()` (x86-64). Code that built or read the fields directly must use the probe and the
  accessors; this breaks only code that relied on the unsound construction.

## [0.1.0] - 2026-11-15

### Added
- `no_std` + `alloc` core (forbids `unsafe`): `CompletionLine`, `MemoryDescriptor`, `WaitPlan`,
  `WaitBands`, `PointTable` deadline estimator, `CompletionPort` + `wait`, `Clock`/`Timer` ports,
  `ManualClock`, the FILL `Executor` + `Reactor`, `ExecutorKind` with the declared fallback chain
  and `Strictness`, and the coded `Error`/`ErrorCode` registry.
- Backends: `cuda` (`MappedLine`, `CudaPort`), `hip` (`MappedLine`), `metal`
  (`SharedEventPort`), `wordwait` (aarch64 `wfe`/`wfet`, x86-64 `umwait`/`mwaitx`).
- Hosts: `std` (`StdClock`, `StdIdle`, `Watcher`), `tokio` (`TokioClock`), `io-uring`
  (`UringIdle`).
- Extracted from the Metrale Engine's in-tree completion crate (2026-11-14), behaviour unchanged.
