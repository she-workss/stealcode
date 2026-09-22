//! Dedicated thread pool for the parallel kernels in this crate.
//!
//! The kernels used to run on rayon's global pool, whose default size is the
//! logical CPU count. On a 16-thread machine that made the streaming encoder
//! about 3x slower than a small pool: the serial driver thread has to run
//! between the parallel regions, and with one worker per logical CPU the
//! rayon workers spin on their latches long enough to starve it. Measured on
//! an 11 s clip with the `wav` example (Ryzen 7 8845HS, 8C/16T):
//!
//! | threads | wall   | process CPU |
//! |---------|--------|-------------|
//! | 4       | 1.76 s | -           |
//! | 5       | 1.75 s | 6.3 s       |
//! | 16      | 5.5 s  | 46.7 s      |
//!
//! The pool is therefore capped at `available_parallelism() / 4 + 1`, clamped
//! to `2..=8`. `STEALCODE_ASR_THREADS` overrides the size (e.g. `16` reproduces
//! the slow baseline for A/B checks). The global pool is deliberately left
//! untouched - callers that want it keep using it explicitly.

use std::sync::OnceLock;

use rayon::ThreadPool;

/// Lazily built pool shared by every parallel region in this crate.
fn pool() -> &'static ThreadPool {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(configured_worker_count())
            .thread_name(|i| format!("asr-worker-{i}"))
            .build()
            .expect("failed to build the ASR thread pool")
    })
}

/// Pool size: `STEALCODE_ASR_THREADS` when it parses as a positive `usize`,
/// otherwise `available_parallelism() / 4 + 1` clamped to `2..=8`.
fn configured_worker_count() -> usize {
    if let Some(n) = std::env::var("STEALCODE_ASR_THREADS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
    {
        return n.max(1);
    }
    let cpus = std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get);
    (cpus / 4 + 1).clamp(2, 8)
}

/// Number of worker threads in the dedicated pool.
#[must_use]
#[allow(dead_code, reason = "accessor used by tests and diagnostics")]
pub(crate) fn worker_count() -> usize {
    pool().current_num_threads()
}

/// Runs `f` on the dedicated pool and returns its result.
///
/// Every `.par_*` region inside `f` then executes on this pool instead of
/// rayon's global one.
pub(crate) fn install<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    pool().install(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_runs_on_a_named_worker() {
        assert!(worker_count() >= 2);
        let (value, name) = install(|| {
            (40 + 2, std::thread::current().name().map(str::to_owned))
        });
        assert_eq!(value, 42);
        let name = name.expect("pool worker thread should be named");
        assert!(
            name.starts_with("asr-worker"),
            "unexpected thread name: {name}"
        );
    }
}
