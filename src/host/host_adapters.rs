// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hosts: the clocks, timers and idles a binary picks. The core never names one.
//!
//! | feature | module | what it provides |
//! |---|---|---|
//! | `std` | [`std_host`] | `StdClock`, `StdIdle` (thread sleep, then a word-wait park or a yield loop), `Watcher` |
//! | `tokio` | [`tokio`] | a [`Timer`](crate::Timer) over `tokio::time` |
//! | `io-uring` | [`uring`] | an [`Idle`](crate::Idle) whose deadline sleep is `IORING_OP_TIMEOUT` |

#[cfg(feature = "std")]
pub mod std_host;

#[cfg(feature = "tokio")]
pub mod tokio;

#[cfg(all(feature = "io-uring", target_os = "linux"))]
#[allow(unsafe_code)]
pub mod uring;
