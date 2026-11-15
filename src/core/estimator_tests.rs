// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

const PLAN: DeadlinePlan = DeadlinePlan {
    margin_ns: 150_000,
    min_sleep_ns: 20_000,
    max_sleep_ns: 50_000_000,
    window: 8,
};

fn exact(r: u64) -> Outcome {
    Outcome::Exact { residual_ns: r }
}

#[test]
fn cold_point_does_not_sleep() {
    assert_eq!(PointEstimate::new().sleep_ns(&PLAN), None);
    assert_eq!(PointTable::<u32>::new().sleep_ns(&7, &PLAN), None);
}

#[test]
fn prediction_is_min_of_window_minus_margin() {
    let mut e = PointEstimate::new();
    for r in [1_000_000, 900_000, 1_100_000] {
        e.observe(&PLAN, exact(r));
    }
    assert_eq!(e.sleep_ns(&PLAN), Some(900_000 - 150_000));
    for _ in 0..8 {
        e.observe(&PLAN, exact(2_000_000));
    }
    assert_eq!(
        e.sleep_ns(&PLAN),
        Some(2_000_000 - 150_000),
        "the min must leave the window"
    );
}

#[test]
fn short_points_never_sleep_and_long_sleeps_are_capped() {
    for (r, want) in [
        (160_000, None),
        (100_000, None),
        (u64::MAX, Some(PLAN.max_sleep_ns)),
    ] {
        let mut e = PointEstimate::new();
        e.observe(&PLAN, exact(r));
        assert_eq!(e.sleep_ns(&PLAN), want, "residual {r}");
    }
}

#[test]
fn a_late_wake_only_lowers_the_prediction() {
    let mut e = PointEstimate::new();
    e.observe(&PLAN, exact(1_000_000));
    e.observe(&PLAN, Outcome::Late { woke_ns: 5_000_000 });
    assert_eq!(
        e.predicted_ns(),
        Some(500_000),
        "censored at min(wake, prediction) / 2"
    );
    let mut cold = PointEstimate::new();
    cold.observe(&PLAN, Outcome::Late { woke_ns: 800_000 });
    assert_eq!(cold.predicted_ns(), Some(400_000));
}

/// One simulated wait: done `r` after the start; a sleep of `s` wakes at `s + overshoot`.
fn step(e: &mut PointEstimate, r: u64, overshoot: u64) -> bool {
    match e.sleep_ns(&PLAN) {
        Some(s) if s + overshoot >= r => {
            e.observe(
                &PLAN,
                Outcome::Late {
                    woke_ns: s + overshoot,
                },
            );
            true
        }
        _ => {
            e.observe(&PLAN, exact(r));
            false
        }
    }
}

#[test]
fn overshoot_never_feeds_back_into_the_deadline() {
    // AFD run 1 drifted because a sample carried the sleep's overshoot. With a 1-5 ms overshoot
    // the prediction must never exceed the truth, and not every wait may be late.
    for alt in [false, true] {
        let r = 1_000_000;
        let mut e = PointEstimate::new();
        let mut late = 0;
        for i in 0..1000u32 {
            let o = if alt && i % 2 == 0 {
                1_000_000
            } else {
                5_000_000
            };
            late += u32::from(step(&mut e, r, o));
            assert!(
                e.predicted_ns().is_some_and(|p| p <= r),
                "prediction above truth: {e:?}"
            );
        }
        assert!(late < 1000, "every wait late");
    }
}

#[test]
fn small_overshoot_converges_to_no_late_wakes() {
    let mut e = PointEstimate::new();
    let late: u32 = (0..200)
        .map(|_| u32::from(step(&mut e, 1_000_000, 50_000)))
        .sum();
    assert_eq!(
        late, 0,
        "an overshoot under the margin must never wake late"
    );
    assert_eq!(e.sleep_ns(&PLAN), Some(850_000));
}

#[test]
fn table_keeps_points_apart() {
    let mut t = PointTable::new();
    t.observe("short", &PLAN, exact(30_000));
    t.observe("long", &PLAN, exact(4_000_000));
    assert_eq!(t.len(), 2);
    assert_eq!(t.sleep_ns(&"short", &PLAN), None);
    assert_eq!(t.sleep_ns(&"long", &PLAN), Some(3_850_000));
}
