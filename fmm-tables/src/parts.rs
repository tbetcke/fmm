//! The parts of a table build on a caller's threads (Helmholtz CH4.0h).
//!
//! A builder that takes a [`Parts`] cuts its work into independent parts, numbered
//! 0..n, each a function of the build's inputs and its number alone, and asks the map for
//! their results in that order. [`Serial`] computes them one after the other on the calling
//! thread; a caller with a thread pool (`nd-fmm-exec`'s Helmholtz operator) computes them on
//! its threads. Which thread computes a part, and when, changes no bit: the results are put
//! together in part order, so every build is the serial build's bit for bit. This crate
//! itself runs no thread.

/// Computes the parts of a build (module documentation).
pub trait Parts: Sync {
    /// Returns `[f(0), f(1), …, f(n − 1)]`, in that order, each computed once, on any
    /// thread.
    fn map<R: Send>(&self, n: usize, f: &(dyn Fn(usize) -> R + Sync)) -> Vec<R>;
}

/// The parts one after the other on the calling thread.
#[derive(Clone, Copy, Debug, Default)]
pub struct Serial;

impl Parts for Serial {
    fn map<R: Send>(&self, n: usize, f: &(dyn Fn(usize) -> R + Sync)) -> Vec<R> {
        (0..n).map(f).collect()
    }
}
