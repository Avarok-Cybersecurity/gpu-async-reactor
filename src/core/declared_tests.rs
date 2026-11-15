// SPDX-License-Identifier: MIT OR Apache-2.0

//! The executor setting: the declared chain io-uring -> gpu-reactor -> naive, a coded warning on
//! every fallback, and a refusal only when strict.

use alloc::string::String;

use super::*;

fn caps(word: bool, uring: bool) -> ExecutorCaps {
    let why = |ok: bool, w: &str| if ok { Ok(()) } else { Err(String::from(w)) };
    ExecutorCaps {
        word_wait: why(word, "no user-mode WFE"),
        io_uring: why(uring, "io_uring disabled"),
    }
}

#[test]
fn every_kind_runs_as_requested_on_a_host_that_has_everything() {
    for k in ExecutorKind::ALL {
        let r = resolve(k, Strictness::Strict, &caps(true, true)).expect("available");
        assert_eq!((r.requested, r.running, r.warning.clone()), (k, k, None));
        assert_eq!(r.label(), k.name());
        assert_eq!(ExecutorKind::parse(k.name()).expect("round-trip"), k);
    }
}

#[test]
fn a_missing_io_uring_falls_back_to_the_gpu_reactor_with_a_coded_warning() {
    let r = resolve(ExecutorKind::IoUring, Strictness::Fallback, &caps(true, false)).expect("ok");
    assert_eq!(r.running, ExecutorKind::GpuReactor);
    let w = r.warning.clone().expect("a fallback warns");
    assert_eq!(w.code(), crate::ErrorCode::ExecutorFallback);
    assert_eq!(
        alloc::format!("{w}"),
        "executor io-uring is not available on this host (io-uring: io_uring disabled), so the \
         declared fallback gpu-reactor runs"
    );
    assert_eq!(r.label(), "gpu-reactor (fallback from io-uring)");
}

#[test]
fn a_host_without_a_word_wait_walks_the_whole_chain_to_naive() {
    for k in [ExecutorKind::IoUring, ExecutorKind::GpuReactor] {
        let r = resolve(k, Strictness::Fallback, &caps(false, true)).expect("ok");
        assert_eq!(r.running, ExecutorKind::Naive, "{k:?}");
        assert!(r.warning.is_some());
    }
    let r = resolve(ExecutorKind::Naive, Strictness::Strict, &caps(false, false)).expect("naive");
    assert_eq!(r.running, ExecutorKind::Naive, "naive needs nothing, even strict");
}

#[test]
fn a_strict_setting_refuses_instead_of_falling_back() {
    let e = resolve(ExecutorKind::IoUring, Strictness::Strict, &caps(true, false))
        .expect_err("strict refuses");
    assert_eq!(e.code(), crate::ErrorCode::ExecutorRefused);
    assert_eq!(
        alloc::format!("{e}"),
        "executor io-uring is not available on this host (io_uring disabled) and the setting is \
         strict: refused"
    );
    resolve(ExecutorKind::GpuReactor, Strictness::Strict, &caps(false, true))
        .expect_err("strict gpu-reactor without a word wait refuses");
}

#[test]
fn an_unknown_executor_name_is_refused_naming_the_choices() {
    let e = ExecutorKind::parse("spin").expect_err("unknown");
    assert_eq!(e.code(), crate::ErrorCode::InvalidExecutor);
    assert_eq!(
        alloc::format!("{e}"),
        "invalid executor \"spin\": expected one of naive, gpu-reactor, io-uring"
    );
}
