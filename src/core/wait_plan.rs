// SPDX-License-Identifier: MIT OR Apache-2.0

//! The wait plan: how a host waits at a readback point, as typed fields parsed once from text
//! (resolve once, then immutable). Every field is stated in the text; nothing is defaulted.
//!
//! Text: `<park>` or `<park>+predict:<deadline fields>`, where `<park>` is `block`, `poll` or
//! `flag:park_us=<n>,backstop_us=<n>` and the deadline fields are
//! `margin_us=<n>,min_sleep_us=<n>,max_sleep_us=<n>,window=<n>`, each key exactly once.
//! `block` takes no prediction: the vendor's blocking wait has nothing to sleep in front of.
//!
//! Invariants: pure; a plan that parses is in range (see the bounds below).

use alloc::format;
use alloc::string::String;

use crate::core::error::{Error, Result};
use crate::error;

/// The largest `window` a deadline plan may state.
pub const WINDOW_MAX: usize = 16;
/// The longest single park on the device word a plan may state (10 ms).
const PARK_US_MAX: u64 = 10_000;
/// The longest gap between two status queries in a flag wait a plan may state (1 s).
const BACKSTOP_US_MAX: u64 = 1_000_000;

/// How a host waits at a readback point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WaitPlan {
    /// Sleep to the point's predicted completion minus a margin before parking.
    pub predict: Option<DeadlinePlan>,
    /// How the wait is spent once it is not (or no longer) sleeping.
    pub park: Park,
}

/// How the rest of a wait is spent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Park {
    /// The vendor's blocking wait (CUDA: `cuStreamSynchronize`).
    Block,
    /// Poll the status until it reports done.
    QueryPoll,
    /// Park on a word the device writes behind the work; query the status every `backstop_ns`.
    FlagWait {
        /// Longest single park before the waiter looks at the clock again.
        park_ns: u64,
        /// Longest time between two status queries (the error authority).
        backstop_ns: u64,
    },
}

/// The deadline fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeadlinePlan {
    /// Wake this long before the predicted completion.
    pub margin_ns: u64,
    /// A computed sleep shorter than this is skipped.
    pub min_sleep_ns: u64,
    /// No single sleep is longer than this.
    pub max_sleep_ns: u64,
    /// Exact residuals kept per readback point (`1..=WINDOW_MAX`); the prediction is their min.
    pub window: u8,
}

/// The plan text is refused for `why`.
fn refused(text: &str, why: &str) -> Error {
    error!(InvalidPlan, "completion wait plan", format!("{text:?}: {why}"))
}

/// The values of the `key=<u64>` fields of `body`, in `keys` order, each once.
fn fields<const N: usize>(text: &str, body: &str, keys: [&str; N]) -> Result<[u64; N]> {
    let mut got: [Option<u64>; N] = [None; N];
    for kv in body.split(',') {
        let (k, v) = kv
            .split_once('=')
            .ok_or_else(|| refused(text, "a field is not key=<n>"))?;
        let i = keys
            .iter()
            .position(|x| *x == k)
            .ok_or_else(|| refused(text, &format!("unknown field {k:?}; expected {keys:?}")))?;
        if got[i].is_some() {
            return Err(refused(text, &format!("field {k:?} stated twice")));
        }
        got[i] = Some(
            v.parse()
                .map_err(|_| refused(text, &format!("field {k:?}: {v:?} is not a u64")))?,
        );
    }
    let mut out = [0u64; N];
    for ((o, g), k) in out.iter_mut().zip(got).zip(keys) {
        *o = g.ok_or_else(|| {
            refused(
                text,
                &format!("field {k:?} missing (every field is required)"),
            )
        })?;
    }
    Ok(out)
}

/// `us` in nanoseconds, refused on overflow.
fn ns(text: &str, us: u64) -> Result<u64> {
    us.checked_mul(1_000)
        .ok_or_else(|| refused(text, "a duration overflows u64 nanoseconds"))
}

impl DeadlinePlan {
    fn parse(text: &str, body: &str) -> Result<Self> {
        let [m, lo, hi, w] = fields(
            text,
            body,
            ["margin_us", "min_sleep_us", "max_sleep_us", "window"],
        )?;
        let window = u8::try_from(w)
            .ok()
            .filter(|w| (1..=WINDOW_MAX).contains(&usize::from(*w)));
        let window = window
            .ok_or_else(|| refused(text, &format!("window {w}: expected 1..={WINDOW_MAX}")))?;
        if lo > hi {
            return Err(refused(text, "min_sleep_us is above max_sleep_us"));
        }
        Ok(Self {
            margin_ns: ns(text, m)?,
            min_sleep_ns: ns(text, lo)?,
            max_sleep_ns: ns(text, hi)?,
            window,
        })
    }
}

impl Park {
    fn parse(text: &str, park: &str) -> Result<Self> {
        match park {
            "block" => Ok(Self::Block),
            "poll" => Ok(Self::QueryPoll),
            p => {
                let body = p
                    .strip_prefix("flag:")
                    .ok_or_else(|| refused(text, "park: expected block, poll or flag:<fields>"))?;
                let [park_us, backstop_us] = fields(text, body, ["park_us", "backstop_us"])?;
                if !(1..=PARK_US_MAX).contains(&park_us) {
                    return Err(refused(
                        text,
                        &format!("park_us {park_us}: expected 1..={PARK_US_MAX}"),
                    ));
                }
                if !(1..=BACKSTOP_US_MAX).contains(&backstop_us) {
                    return Err(refused(
                        text,
                        &format!("backstop_us {backstop_us}: expected 1..={BACKSTOP_US_MAX}"),
                    ));
                }
                Ok(Self::FlagWait {
                    park_ns: ns(text, park_us)?,
                    backstop_ns: ns(text, backstop_us)?,
                })
            }
        }
    }
}

impl WaitPlan {
    /// Parse the lever text (module docs).
    ///
    /// # Errors
    /// `InvalidArgument` naming the text and the first thing wrong with it.
    pub fn parse(text: &str) -> Result<Self> {
        let (park, predict) = match text.split_once('+') {
            None => (text, None),
            Some((park, rest)) => {
                let body = rest
                    .strip_prefix("predict:")
                    .ok_or_else(|| refused(text, "after '+': expected predict:<fields>"))?;
                (park, Some(DeadlinePlan::parse(text, body)?))
            }
        };
        let park = Park::parse(text, park)?;
        if park == Park::Block && predict.is_some() {
            return Err(refused(text, "block takes no predict"));
        }
        Ok(Self { predict, park })
    }

    /// The plan as `parse` reads it (round-trips exactly).
    #[must_use]
    pub fn label(&self) -> String {
        let park = match self.park {
            Park::Block => String::from("block"),
            Park::QueryPoll => String::from("poll"),
            Park::FlagWait {
                park_ns,
                backstop_ns,
            } => {
                format!(
                    "flag:park_us={},backstop_us={}",
                    park_ns / 1_000,
                    backstop_ns / 1_000
                )
            }
        };
        match self.predict {
            None => park,
            Some(d) => format!(
                "{park}+predict:margin_us={},min_sleep_us={},max_sleep_us={},window={}",
                d.margin_ns / 1_000,
                d.min_sleep_ns / 1_000,
                d.max_sleep_ns / 1_000,
                d.window
            ),
        }
    }
}

/// A host capability a park needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Capability {
    /// A device-written word the host can wait on in user mode (aarch64 `ldxr` + `wfe`/`wfet`;
    /// later x86 MONITORX/UMWAIT), and a driver that stores into it behind stream work.
    DeviceWordWait,
}

/// One link of the declared fallback chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fallback {
    /// The park this link applies to (its fields kept only for matching).
    pub from: ParkKind,
    /// What the host must offer for `from` to run.
    pub needs: Capability,
    /// The park the plan resolves to without it (the plan's `predict` is kept unchanged).
    pub to: ParkKind,
}

/// A park's kind, without its fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParkKind {
    /// [`Park::Block`].
    Block,
    /// [`Park::QueryPoll`].
    QueryPoll,
    /// [`Park::FlagWait`].
    FlagWait,
}

/// the whole chain, as data. The executor never chooses;
/// [`WaitPlan::resolve`] walks this table once at boot.
/// The chain ends at `QueryPoll`, which needs nothing and runs on every host; `Block`, the
/// vendor's blocking wait, is the conservative arm a plan may name (declared, not
/// a default).
pub const FALLBACKS: [Fallback; 1] = [Fallback {
    from: ParkKind::FlagWait,
    needs: Capability::DeviceWordWait,
    to: ParkKind::QueryPoll,
}];

/// What a host offers, probed once at boot by the host (never by the core).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostCaps {
    /// `Ok` when [`Capability::DeviceWordWait`] is present, else why not.
    pub device_word_wait: core::result::Result<(), String>,
}

impl HostCaps {
    fn lacks(&self, c: Capability) -> Option<&str> {
        match c {
            Capability::DeviceWordWait => self.device_word_wait.as_ref().err().map(String::as_str),
        }
    }
}

/// The plan a host runs, and the coded warning when a fallback was taken.
#[derive(Clone, Debug)]
pub struct Resolved {
    /// The plan as requested.
    pub requested: WaitPlan,
    /// The plan that runs.
    pub plan: WaitPlan,
    /// [`ErrorCode::ParkFallback`](crate::ErrorCode::ParkFallback) when `plan != requested`.
    pub warning: Option<Error>,
}

impl Resolved {
    /// The running plan's label, naming the fallback when one was taken (boot log,
    /// `/metrics`, bench record).
    #[must_use]
    pub fn label(&self) -> String {
        if self.warning.is_some() {
            format!(
                "{} (fallback from {})",
                self.plan.label(),
                self.requested.label()
            )
        } else {
            self.plan.label()
        }
    }
}

impl Park {
    /// This park's kind.
    #[must_use]
    pub fn kind(&self) -> ParkKind {
        match self {
            Self::Block => ParkKind::Block,
            Self::QueryPoll => ParkKind::QueryPoll,
            Self::FlagWait { .. } => ParkKind::FlagWait,
        }
    }
}

impl WaitPlan {
    /// Resolve against `caps` by walking [`FALLBACKS`]; pure.
    #[must_use]
    pub fn resolve(&self, caps: &HostCaps) -> Resolved {
        let mut plan = *self;
        let mut warning = None;
        let kind = plan.park.kind();
        for link in FALLBACKS.iter().filter(|l| l.from == kind) {
            if let Some(why) = caps.lacks(link.needs) {
                let from = plan.park;
                plan.park = match link.to {
                    ParkKind::Block => Park::Block,
                    ParkKind::QueryPoll => Park::QueryPoll,
                    ParkKind::FlagWait => from,
                };
                warning = Some(error!(
                    ParkFallback,
                    self.label(),
                    format!("{:?} ({:?})", link.from, link.needs),
                    String::from(why),
                    plan.label()
                ));
            }
        }
        Resolved {
            requested: *self,
            plan,
            warning,
        }
    }
}

#[cfg(test)]
#[path = "wait_plan_tests.rs"]
mod tests;
