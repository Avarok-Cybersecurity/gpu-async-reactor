# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the crate follows
[Semantic Versioning](https://semver.org/).

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
