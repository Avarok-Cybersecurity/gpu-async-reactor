// SPDX-License-Identifier: MIT OR Apache-2.0

//! The portable core: `no_std` + `alloc`, no `unsafe`, no I/O, no statics, no vendor name. Every
//! port (device, clock, timer, idle) is a generic parameter chosen by the host; nothing here
//! blocks or names a runtime.
#![forbid(unsafe_code)]

pub(crate) mod bands;
pub(crate) mod declared;
#[path = "reactor_error.rs"]
pub(crate) mod error;
pub(crate) mod estimator;
#[path = "fill_executor.rs"]
pub(crate) mod executor;
pub(crate) mod line;
pub(crate) mod manual;
#[path = "wait_plan.rs"]
pub(crate) mod plan;
pub(crate) mod reactor;
pub(crate) mod time;
pub(crate) mod wait;
