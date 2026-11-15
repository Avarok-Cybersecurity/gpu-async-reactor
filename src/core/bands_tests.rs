// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-11-14: Band selection, the band grammar's refusals and the per-band fallback.

use alloc::string::String;

use super::{BANDS_MAX, StepRows, WaitBands};
use crate::core::plan::{HostCaps, Park, WaitPlan};

const FLAGDL: &str = "flag:park_us=100,backstop_us=1000+predict:margin_us=150,min_sleep_us=20,max_sleep_us=50000,window=8";

fn ruled() -> String {
    alloc::format!("block;rows>=32:{FLAGDL}")
}

fn block() -> WaitPlan {
    WaitPlan::parse("block").unwrap()
}

fn flagdl() -> WaitPlan {
    WaitPlan::parse(FLAGDL).unwrap()
}

#[test]
fn the_ruled_bands_block_through_31_rows_and_flag_from_32() {
    let b = WaitBands::parse(&ruled()).unwrap();
    for rows in [0, 1, 4, 8, 16, 17, 31] {
        assert_eq!(*b.select(rows), block(), "rows {rows}");
    }
    for rows in [32, 33, 64, 128, u32::MAX] {
        assert_eq!(*b.select(rows), flagdl(), "rows {rows}");
    }
}

#[test]
fn every_edge_is_inclusive_on_its_own_band_with_three_bands() {
    let b = WaitBands::parse(&alloc::format!("block;rows>=2:poll;rows>=5:{FLAGDL}")).unwrap();
    let poll = WaitPlan::parse("poll").unwrap();
    let want = [block(), block(), poll, poll, poll, flagdl(), flagdl()];
    for (rows, w) in (0u32..).zip(want) {
        assert_eq!(*b.select(rows), w, "rows {rows}");
    }
}

#[test]
fn one_plan_without_edges_is_a_single_band_for_every_row_count() {
    let b = WaitBands::parse(FLAGDL).unwrap();
    assert_eq!(b.iter().count(), 1);
    for rows in [0, 16, 32, u32::MAX] {
        assert_eq!(*b.select(rows), flagdl());
    }
}

#[test]
fn labels_round_trip() {
    for text in [
        String::from("block"),
        ruled(),
        alloc::format!("poll;rows>=1:block;rows>=9:{FLAGDL}"),
    ] {
        assert_eq!(WaitBands::parse(&text).unwrap().label(), text);
    }
}

#[test]
fn malformed_bands_are_refused_naming_the_fault() {
    let five = alloc::format!(
        "block{}",
        ";rows>=1:block;rows>=2:block;rows>=3:block;rows>=4:block"
    );
    let cases = [
        (
            alloc::format!("block;rows>=32:{FLAGDL};rows>=32:block"),
            "does not exceed",
        ),
        (String::from("block;rows>=0:poll"), "does not exceed"),
        (
            String::from("poll;rows>=8:block;rows>=4:poll"),
            "does not exceed",
        ),
        (String::from("block;32:poll"), "expected rows>=<n>:<plan>"),
        (String::from("block;rows>=x:poll"), "is not a u32"),
        (String::from("block;rows>=-1:poll"), "is not a u32"),
        (String::from("rows>=0:block"), "park: expected"),
        (String::from("block;rows>=4:bogus"), "park: expected"),
        (String::from("block;"), "expected rows>=<n>:<plan>"),
        (five, "more than"),
    ];
    for (text, want) in cases {
        let e = WaitBands::parse(&text).expect_err(&text);
        let msg = alloc::format!("{e}");
        assert!(msg.contains(want), "{text:?}: {msg}");
    }
    let four = "block;rows>=1:block;rows>=2:block;rows>=3:block";
    assert_eq!(WaitBands::parse(four).unwrap().iter().count(), BANDS_MAX);
}

#[test]
fn a_host_without_wfe_falls_back_in_the_flag_band_only() {
    let caps = HostCaps {
        device_word_wait: Err(String::from("no user-mode WFE on this CPU")),
    };
    let r = WaitBands::parse(&ruled()).unwrap().resolve(&caps);
    assert!(r.select(8).warning.is_none());
    assert_eq!(r.select(8).plan, block());
    let flag = r.select(32);
    assert!(flag.warning.is_some());
    assert_eq!(flag.plan.park, Park::QueryPoll);
    assert_eq!(flag.plan.predict, flagdl().predict);
    assert!(
        r.label().starts_with("block;rows>=32:poll+predict:"),
        "{}",
        r.label()
    );
    assert!(r.label().contains("(fallback from flag:"), "{}", r.label());
}

#[test]
fn step_rows_are_shared_by_every_clone_and_start_at_zero() {
    let a = StepRows::new();
    let b = a.clone();
    assert_eq!(b.current(), 0);
    a.publish(48);
    assert_eq!(b.current(), 48);
    b.publish(3);
    assert_eq!(a.current(), 3);
    assert_ne!(
        StepRows::new().current(),
        3,
        "a fresh cell is not linked to another"
    );
}
