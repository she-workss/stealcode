//! Per-phase timing of `encode()` (enabled via STEALCODE_PHASE_TIMING=1).

use std::{
    sync::{Mutex, OnceLock},
    time::{Duration, Instant},
};

static ENABLED: OnceLock<bool> = OnceLock::new();
static ACC: Mutex<Vec<(&'static str, Duration)>> = Mutex::new(Vec::new());
static LAST: Mutex<Option<(Instant, &'static str)>> = Mutex::new(None);

pub(crate) fn enabled() -> bool {
    *ENABLED
        .get_or_init(|| std::env::var_os("STEALCODE_PHASE_TIMING").is_some())
}

pub(crate) fn tick(name: &'static str) {
    if !enabled() {
        return;
    }
    let Ok(mut last) = LAST.lock() else { return };
    if let Some((t0, prev)) = last.take() {
        let Ok(mut acc) = ACC.lock() else { return };
        if let Some(e) = acc.iter_mut().find(|(n, _)| *n == prev) {
            e.1 += t0.elapsed();
        } else {
            acc.push((prev, t0.elapsed()));
        }
    }
    *last = Some((Instant::now(), name));
}

pub(crate) fn report() {
    if !enabled() {
        return;
    }
    tick("total");
    let Ok(acc) = ACC.lock() else { return };
    let total: Duration = acc.iter().map(|(_, d)| *d).sum();
    eprintln!("=== PHASES encode ===");
    for (n, d) in acc.iter() {
        let pct = d.as_secs_f64() / total.as_secs_f64() * 100.0;
        eprintln!("  {n:20} {d:?} ({pct:.1}%)");
    }
}
