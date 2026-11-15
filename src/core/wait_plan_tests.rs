// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

const DL: DeadlinePlan = DeadlinePlan {
    margin_ns: 150_000,
    min_sleep_ns: 20_000,
    max_sleep_ns: 50_000_000,
    window: 8,
};
const DL_TEXT: &str = "predict:margin_us=150,min_sleep_us=20,max_sleep_us=50000,window=8";

#[test]
fn every_measured_arm_parses_and_round_trips() {
    let flag = Park::FlagWait {
        park_ns: 100_000,
        backstop_ns: 1_000_000,
    };
    let cases = [
        (
            "block".into(),
            WaitPlan {
                predict: None,
                park: Park::Block,
            },
        ),
        (
            "poll".into(),
            WaitPlan {
                predict: None,
                park: Park::QueryPoll,
            },
        ),
        (
            format!("poll+{DL_TEXT}"),
            WaitPlan {
                predict: Some(DL),
                park: Park::QueryPoll,
            },
        ),
        (
            "flag:park_us=100,backstop_us=1000".into(),
            WaitPlan {
                predict: None,
                park: flag,
            },
        ),
        (
            format!("flag:park_us=100,backstop_us=1000+{DL_TEXT}"),
            WaitPlan {
                predict: Some(DL),
                park: flag,
            },
        ),
    ];
    for (text, want) in cases {
        let got = WaitPlan::parse(&text).unwrap_or_else(|e| panic!("{text}: {e}"));
        assert_eq!(got, want, "{text}");
        assert_eq!(got.label(), text, "label must round-trip");
    }
}

#[test]
fn field_order_is_free() {
    let p =
        WaitPlan::parse("poll+predict:window=8,max_sleep_us=50000,min_sleep_us=20,margin_us=150");
    assert_eq!(p.map(|p| p.predict), Ok(Some(DL)));
}

#[test]
fn every_malformed_text_is_refused_with_its_reason() {
    let cases = [
        ("", "park: expected"),
        ("spin", "park: expected"),
        (
            "block+predict:margin_us=150,min_sleep_us=20,max_sleep_us=50000,window=8",
            "block takes no predict",
        ),
        ("poll+deadline:margin_us=1", "expected predict:"),
        (
            "poll+predict:margin_us=150,min_sleep_us=20,max_sleep_us=50000",
            "window\" missing",
        ),
        (
            "poll+predict:margin_us=1,margin_us=1,min_sleep_us=20,max_sleep_us=50000,window=8",
            "stated twice",
        ),
        (
            "poll+predict:margin=150,min_sleep_us=20,max_sleep_us=50000,window=8",
            "unknown field",
        ),
        (
            "poll+predict:margin_us=-1,min_sleep_us=20,max_sleep_us=50000,window=8",
            "is not a u64",
        ),
        (
            "poll+predict:margin_us=150,min_sleep_us=20,max_sleep_us=50000,window=0",
            "window 0",
        ),
        (
            "poll+predict:margin_us=150,min_sleep_us=20,max_sleep_us=50000,window=17",
            "window 17",
        ),
        (
            "poll+predict:margin_us=150,min_sleep_us=60,max_sleep_us=50,window=8",
            "min_sleep_us is above",
        ),
        (
            "poll+predict:margin_us=18446744073709551615,min_sleep_us=1,max_sleep_us=2,window=8",
            "overflows",
        ),
        ("flag:park_us=0,backstop_us=1000", "park_us 0"),
        ("flag:park_us=10001,backstop_us=1000", "park_us 10001"),
        ("flag:park_us=100,backstop_us=0", "backstop_us 0"),
        ("flag:park_us=100", "backstop_us\" missing"),
        ("flag:park_us=100,backstop_us=1000,x=1", "unknown field"),
        ("poll,x", "park: expected"),
    ];
    for (text, why) in cases {
        let err = WaitPlan::parse(text).expect_err(text);
        assert_eq!(
            err.code(),
            crate::ErrorCode::InvalidPlan,
            "{text}"
        );
        let msg = alloc::format!("{err}");
        assert!(msg.contains(why), "{text}: {msg:?} lacks {why:?}");
    }
}

const FLAGDL: &str = "flag:park_us=100,backstop_us=1000+predict:margin_us=150,min_sleep_us=20,max_sleep_us=50000,window=8";

fn no_wfe() -> HostCaps {
    HostCaps {
        device_word_wait: Err("no user-mode WFE on this CPU".into()),
    }
}

#[test]
fn a_host_without_wfe_falls_back_to_poll_with_the_same_prediction_and_a_coded_warning() {
    let requested = WaitPlan::parse(FLAGDL).unwrap();
    let r = requested.resolve(&no_wfe());
    assert_eq!(
        r.plan,
        WaitPlan {
            predict: Some(DL),
            park: Park::QueryPoll
        }
    );
    let w = r.warning.clone().expect("a fallback must warn");
    assert_eq!(w.code(), crate::ErrorCode::ParkFallback);
    assert_eq!(
        alloc::format!("{w}"),
        alloc::format!(
            "completion wait {FLAGDL}: FlagWait (DeviceWordWait) is not available on this host \
             (no user-mode WFE on this CPU), so the declared fallback poll+{DL_TEXT} runs"
        )
    );
    assert_eq!(
        r.label(),
        alloc::format!("poll+{DL_TEXT} (fallback from {FLAGDL})")
    );
}

#[test]
fn a_host_with_wfe_runs_the_plan_as_requested() {
    let requested = WaitPlan::parse(FLAGDL).unwrap();
    let r = requested.resolve(&HostCaps {
        device_word_wait: Ok(()),
    });
    assert_eq!((r.plan, r.warning.is_none()), (requested, true));
    assert_eq!(r.label(), FLAGDL);
}

#[test]
fn plans_that_need_no_capability_never_fall_back() {
    for text in [
        "block".into(),
        "poll".into(),
        alloc::format!("poll+{DL_TEXT}"),
    ] {
        let p = WaitPlan::parse(&text).unwrap();
        let r = p.resolve(&no_wfe());
        assert_eq!((r.plan, r.warning.is_none()), (p, true), "{text}");
    }
}
