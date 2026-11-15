// SPDX-License-Identifier: MIT OR Apache-2.0

//! The line's ownership rules against a fake driver (no GPU): when the domain is retained and
//! released, when the page is freed, when it is leaked, and which stores are refused.

extern crate std;

use std::boxed::Box;
use std::cell::RefCell;
use std::rc::Rc;
use std::vec::Vec;

use super::*;

/// A fake driver: pages come from a 64-aligned heap box; every call is logged.
struct Fake {
    log: Rc<RefCell<Vec<&'static str>>>,
    domain: u64,
    stream_domain: u64,
    settle_ok: bool,
    device_fails: bool,
    alloc_fails: bool,
}

#[repr(C, align(4096))]
struct Page([u8; PAGE]);

impl Fake {
    fn new(log: &Rc<RefCell<Vec<&'static str>>>) -> Self {
        Self {
            log: Rc::clone(log),
            domain: 7,
            stream_domain: 7,
            settle_ok: true,
            device_fails: false,
            alloc_fails: false,
        }
    }
    fn push(&self, s: &'static str) {
        self.log.borrow_mut().push(s);
    }
}

impl LineDriver for Fake {
    type Stream = u32;
    type Domain = u64;

    fn retain_current(&self) -> Result<u64, LineError> {
        if self.domain == 0 {
            return Err(LineError::NoDomain);
        }
        self.push("retain");
        Ok(self.domain)
    }

    fn release(&self, domain: u64) {
        assert_eq!(domain, 7, "releases the domain it retained");
        self.push("release");
    }

    fn alloc_host(&self, domain: u64, bytes: usize) -> Result<NonNull<c_void>, LineError> {
        assert_eq!((domain, bytes), (7, PAGE));
        if self.alloc_fails {
            return Err(LineError::Status(2));
        }
        self.push("alloc");
        let page = Box::into_raw(Box::new(Page([0; PAGE])));
        NonNull::new(page.cast::<c_void>()).ok_or(LineError::Unavailable)
    }

    fn device_address(&self, _domain: u64, _host: NonNull<c_void>) -> Result<u64, LineError> {
        if self.device_fails {
            Err(LineError::Status(201))
        } else {
            Ok(0x1000)
        }
    }

    fn free_host(&self, domain: u64, host: NonNull<c_void>) {
        assert_eq!(domain, 7, "frees in the line's own domain");
        self.push("free");
        // WHY: the fake driver frees the page its `alloc_host` boxed, as the real drivers free theirs.
        // SAFETY: the page came from `alloc_host`'s `Box<Page>` and the line frees it once.
        // NO SAFE ALTERNATIVE: `Box::from_raw` is the only inverse of the raw page `alloc_host` handed out.
        drop(unsafe { Box::from_raw(host.as_ptr().cast::<Page>()) });
    }

    fn stream_in(&self, _stream: u32, domain: u64) -> Result<bool, LineError> {
        Ok(self.stream_domain == domain)
    }

    fn write_value(&self, _stream: u32, addr: u64, _value: u64) -> Result<(), LineError> {
        assert_eq!(addr, 0x1000 + 8 * 3, "slot 3's device address");
        self.push("write");
        Ok(())
    }

    fn settle(&self, domain: u64) -> bool {
        assert_eq!(domain, 7, "settles the line's own domain");
        self.push("settle");
        self.settle_ok
    }
}

fn slot3() -> Slot {
    Slot::new(3).expect("slot 3")
}

#[test]
fn a_line_with_a_queued_store_settles_its_domain_before_freeing_the_page() {
    let log = Rc::new(RefCell::new(Vec::new()));
    let line = Line::with_driver(Fake::new(&log)).expect("line");
    line.enqueue_store(0, slot3(), 1).expect("store");
    drop(line);
    assert_eq!(*log.borrow(), ["retain", "alloc", "write", "settle", "free", "release"]);
}

#[test]
fn a_line_whose_domain_cannot_settle_leaks_the_page_instead_of_freeing_it() {
    let log = Rc::new(RefCell::new(Vec::new()));
    let mut fake = Fake::new(&log);
    fake.settle_ok = false;
    let line = Line::with_driver(fake).expect("line");
    line.enqueue_store(0, slot3(), 1).expect("store");
    drop(line);
    assert_eq!(
        *log.borrow(),
        ["retain", "alloc", "write", "settle"],
        "no free under a pending store, and the domain that maps it is kept"
    );
}

#[test]
fn a_line_that_never_queued_a_store_frees_without_settling() {
    let log = Rc::new(RefCell::new(Vec::new()));
    drop(Line::with_driver(Fake::new(&log)).expect("line"));
    assert_eq!(*log.borrow(), ["retain", "alloc", "free", "release"]);
}

#[test]
fn a_stream_of_another_domain_is_refused_before_any_store() {
    let log = Rc::new(RefCell::new(Vec::new()));
    let mut fake = Fake::new(&log);
    fake.stream_domain = 8;
    let line = Line::with_driver(fake).expect("line");
    assert_eq!(line.enqueue_store(0, slot3(), 1), Err(LineError::ForeignStream));
    assert_eq!(LineError::ForeignStream.status(), -4);
    drop(line);
    assert_eq!(*log.borrow(), ["retain", "alloc", "free", "release"], "no write, so no settle");
}

/// A construction failure: its name, how the fake fails, the error, and the driver calls made.
type FailCase = (&'static str, fn(&mut Fake), LineError, &'static [&'static str]);

#[test]
fn a_failed_construction_releases_what_it_took() {
    let cases: [FailCase; 3] = [
        ("no domain", |f| f.domain = 0, LineError::NoDomain, &[]),
        ("alloc fails", |f| f.alloc_fails = true, LineError::Status(2), &["retain", "release"]),
        (
            "device address fails",
            |f| f.device_fails = true,
            LineError::Status(201),
            &["retain", "alloc", "free", "release"],
        ),
    ];
    for (name, setup, want, steps) in cases {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut fake = Fake::new(&log);
        setup(&mut fake);
        let err = Line::with_driver(fake).err().expect("refused");
        assert_eq!(err, want, "{name}");
        assert_eq!(*log.borrow(), steps, "{name}: nothing leaked, nothing released twice");
    }
}

#[test]
fn a_fresh_line_reads_all_zero() {
    let log = Rc::new(RefCell::new(Vec::new()));
    let line = Line::with_driver(Fake::new(&log)).expect("line");
    for w in &line.line().words {
        assert_eq!(w.load(Ordering::Acquire), 0);
    }
}
