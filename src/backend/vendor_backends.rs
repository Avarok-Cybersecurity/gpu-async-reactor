// SPDX-License-Identifier: MIT OR Apache-2.0

//! Vendor backends, each behind its cargo feature. This is the only place `unsafe` is allowed:
//! every block carries a `SAFETY:` comment, and every public item is safe to call.
//!
//! | feature | module | what it provides |
//! |---|---|---|
//! | `wordwait` | [`wordwait`] | the user-mode wait on a completion line (aarch64 `wfe`/`wfet`, x86-64 `mwaitx`/`umwait`) |
//! | `cuda` | [`cuda`] | a device-mapped line stored into by `cuStreamWriteValue64` |
//! | `hip` | [`hip`] | the same over `hipStreamWriteValue64` |
//! | `metal` | `metal` (Apple targets) | an `MTLSharedEvent` as a [`CompletionPort`](crate::CompletionPort) |

#[cfg(any(feature = "cuda", feature = "hip"))]
mod mapped;
#[cfg(any(feature = "cuda", feature = "hip"))]
pub use mapped::LineError;

#[cfg(feature = "cuda")]
#[allow(unsafe_code)]
pub mod cuda;

#[cfg(feature = "hip")]
#[allow(unsafe_code)]
pub mod hip;

#[cfg(all(feature = "metal", target_vendor = "apple"))]
pub mod metal;

#[cfg(feature = "wordwait")]
#[allow(unsafe_code)]
pub mod wordwait;
