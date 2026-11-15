// SPDX-License-Identifier: MIT OR Apache-2.0

//! FILL on one thread, no GPU: a "device" thread stores sequence numbers into a completion line
//! at 2, 4 and 6 ms; three tasks await one slot each, and a fourth does CPU work in slices. The
//! executor runs the work while the waits are pending and idles (sleep, then a line park) only
//! when nothing is ready.
//!
//! `cargo run --example fill_executor --features std` (add `wordwait` for the hardware park).

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use gpu_async_reactor::host::std_host::{StdClock, StdIdle};
use gpu_async_reactor::{
    Clock, CompletionLine, DurNs, Executor, MemoryDescriptor, Reactor, Slot, yield_now,
};

enum Job<'r, 'l> {
    Wait(&'r Reactor<'l, StdClock>, MemoryDescriptor<'l>),
    Work(&'r Reactor<'l, StdClock>, u32),
}

async fn run(job: Job<'_, '_>) {
    match job {
        Job::Wait(r, d) => {
            let t0 = r.clock().monotonic_ns();
            r.ready(d, None).await;
            let t = r.clock().monotonic_ns().since(t0).0;
            println!("slot {} done after {:.2} ms", d.slot().index(), t as f64 / 1e6);
        }
        Job::Work(r, slices) => {
            for i in 0..slices {
                let spin = Instant::now();
                while spin.elapsed() < Duration::from_micros(300) {
                    std::hint::spin_loop();
                }
                yield_now().await;
                if i % 5 == 4 {
                    println!("work slice {i} at {:.2} ms", r.clock().monotonic_ns().0 as f64 / 1e6);
                }
            }
        }
    }
}

fn main() -> Result<(), gpu_async_reactor::Error> {
    let line = Arc::new(CompletionLine::default());
    let device = {
        let line = Arc::clone(&line);
        std::thread::spawn(move || {
            for slot in 0..3usize {
                std::thread::sleep(Duration::from_millis(2));
                line.words[slot].store(1, Ordering::Release);
            }
        })
    };
    let clock = StdClock::new();
    let reactor = Reactor::new(clock);
    let mut ex = Executor::new();
    for slot in 0..3 {
        let d = MemoryDescriptor::new(&line, Slot::new(slot).expect("slot < 8"), 1);
        ex.spawn(run(Job::Wait(&reactor, d)));
    }
    ex.spawn(run(Job::Work(&reactor, 10)));
    let mut idle = StdIdle::new(clock);
    let stats = ex.run(&reactor, &mut idle, DurNs(100_000))?;
    device.join().expect("device thread");
    println!(
        "polls {} | idles: slept {} parked {} | idle {:.2} ms | hardware park {}",
        stats.polls,
        stats.slept,
        stats.parked,
        stats.idle_ns as f64 / 1e6,
        idle.parks_in_hardware()
    );
    Ok(())
}
