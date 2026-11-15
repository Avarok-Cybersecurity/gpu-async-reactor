// SPDX-License-Identifier: MIT OR Apache-2.0

//! Coded errors (CERF): one append-only `#[repr(u16)]` registry, a `{}` form template per code,
//! arguments kept as text and substituted only when the error is displayed. The code is the
//! identity; the text may change between releases, the code never does. Never branch on the text.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

macro_rules! registry {
    ($($(#[$doc:meta])* $name:ident = $code:literal => $form:literal,)*) => {
        /// The identity of an [`Error`]: append-only, a number is never reused.
        #[repr(u16)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        #[non_exhaustive]
        pub enum ErrorCode {
            $($(#[$doc])* $name = $code,)*
        }

        impl ErrorCode {
            /// Every code, in registry order.
            pub const ALL: &'static [ErrorCode] = &[$(ErrorCode::$name,)*];

            /// The form template: each `{}` takes one argument, in order.
            #[must_use]
            pub const fn form(self) -> &'static str {
                match self { $(ErrorCode::$name => $form,)* }
            }

            /// The stable number.
            #[must_use]
            pub const fn number(self) -> u16 {
                self as u16
            }
        }
    };
}

registry! {
    /// A wait-plan or band text that does not parse: (subject, the text and the fault).
    InvalidPlan = 0x0001 => "invalid argument {}: {}",
    /// A requested park the host cannot run, and the declared fallback that runs instead:
    /// (requested plan, missing capability, why, the plan that runs). A warning.
    ParkFallback = 0x0002 => "completion wait {}: {} is not available on this host ({}), so the declared fallback {} runs",
    /// A requested executor the host cannot run, and the declared fallback that runs instead:
    /// (requested, why, the executor that runs). A warning.
    ExecutorFallback = 0x0003 => "executor {} is not available on this host ({}), so the declared fallback {} runs",
    /// A requested executor the host cannot run while the setting is strict: (requested, why).
    ExecutorRefused = 0x0004 => "executor {} is not available on this host ({}) and the setting is strict: refused",
    /// An executor name that is not one of the declared kinds: (the text, the accepted names).
    InvalidExecutor = 0x0005 => "invalid executor {:?}: expected one of {}",
    /// A future pended on a host that completes every future on its first poll: (the site).
    PendedInline = 0x0006 => "{}: a future pended on an inline host",
    /// The executor has unfinished tasks and nothing to wait on: (tasks left).
    Stalled = 0x0007 => "executor stalled: {} task(s) pending with no completion source or timer registered",
    /// A backend call failed: (the call, the vendor status).
    Backend = 0x0008 => "{} failed with vendor status {}",
}

/// A coded error: an [`ErrorCode`] and its arguments.
#[derive(Clone, PartialEq, Eq)]
pub struct Error {
    code: ErrorCode,
    args: Vec<String>,
}

impl Error {
    /// An error of `code` with `args` (one per `{}` of the code's form).
    #[must_use]
    pub fn new(code: ErrorCode, args: Vec<String>) -> Self {
        Self { code, args }
    }

    /// The identity.
    #[must_use]
    pub fn code(&self) -> ErrorCode {
        self.code
    }

    /// The arguments, in form order (for a host that maps the error into its own registry).
    #[must_use]
    pub fn args(&self) -> &[String] {
        &self.args
    }
}

/// Build an [`Error`]: `error!(Code, arg, …)`; each argument is anything `Display`.
#[macro_export]
macro_rules! error {
    ($code:ident $(, $arg:expr)* $(,)?) => {
        $crate::Error::new(
            $crate::ErrorCode::$code,
            ::alloc::vec![$(::alloc::string::ToString::to_string(&$arg)),*],
        )
    };
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut args = self.args.iter();
        let mut rest = self.code.form();
        while let Some(i) = rest.find('{') {
            f.write_str(&rest[..i])?;
            let after = &rest[i..];
            let (debug, len) = if after.starts_with("{:?}") {
                (true, 4)
            } else if after.starts_with("{}") {
                (false, 2)
            } else {
                f.write_str("{")?;
                rest = &after[1..];
                continue;
            };
            match args.next() {
                Some(a) if debug => write!(f, "{a:?}")?,
                Some(a) => f.write_str(a)?,
                None => f.write_str("<missing>")?,
            }
            rest = &after[len..];
        }
        f.write_str(rest)
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:#06x} {:?}] {self}", self.code.number(), self.code)
    }
}

impl core::error::Error for Error {}

/// `Result` with this crate's [`Error`].
pub type Result<T> = core::result::Result<T, Error>;
