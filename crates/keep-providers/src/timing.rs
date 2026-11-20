//! Opt-in phase timings for finding slow paths.
//!
//! Off unless `KEEP_TIMINGS=1` is set or [`set_enabled`] is called. Only
//! phase names, call counts, item counts and durations are kept; never any
//! text. When off, a [`span`] costs one relaxed atomic load.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::{Duration, Instant};

const UNKNOWN: u8 = 0;
const OFF: u8 = 1;
const ON: u8 = 2;

static STATE: AtomicU8 = AtomicU8::new(UNKNOWN);
static STATS: Mutex<BTreeMap<&'static str, Stat>> = Mutex::new(BTreeMap::new());

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stat {
    pub calls: u64,
    pub items: u64,
    pub total: Duration,
}

pub fn enabled() -> bool {
    match STATE.load(Ordering::Relaxed) {
        ON => true,
        OFF => false,
        _ => {
            let on = std::env::var("KEEP_TIMINGS").is_ok_and(|v| !v.is_empty() && v != "0");
            STATE.store(if on { ON } else { OFF }, Ordering::Relaxed);
            on
        }
    }
}

pub fn set_enabled(on: bool) {
    STATE.store(if on { ON } else { OFF }, Ordering::Relaxed);
}

pub fn record(phase: &'static str, elapsed: Duration, items: u64) {
    if !enabled() {
        return;
    }
    let mut stats = STATS.lock().unwrap_or_else(|e| e.into_inner());
    let s = stats.entry(phase).or_default();
    s.calls += 1;
    s.items += items;
    s.total += elapsed;
}

/// Times from creation to drop and records under `phase`.
#[must_use = "a span records when it is dropped"]
pub struct Span {
    phase: &'static str,
    start: Option<Instant>,
    items: u64,
}

impl Span {
    /// Number of items (texts, rows, points) this call handled.
    pub fn items(mut self, n: usize) -> Self {
        self.items = n as u64;
        self
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            record(self.phase, start.elapsed(), self.items);
        }
    }
}

pub fn span(phase: &'static str) -> Span {
    Span {
        phase,
        start: enabled().then(Instant::now),
        items: 0,
    }
}

/// Every phase recorded since the last [`take`], sorted by name.
pub fn take() -> Vec<(&'static str, Stat)> {
    let mut stats = STATS.lock().unwrap_or_else(|e| e.into_inner());
    std::mem::take(&mut *stats).into_iter().collect()
}

/// One line per phase: calls, items, total and per-item time.
pub fn render(stats: &[(&'static str, Stat)]) -> String {
    let mut out = String::new();
    for (phase, s) in stats {
        let per_item = if s.items > 0 {
            format!(
                " ({:.3} ms/item)",
                s.total.as_secs_f64() * 1e3 / s.items as f64
            )
        } else {
            String::new()
        };
        out.push_str(&format!(
            "  {phase:<28} {:>9.3}s  calls={:<7} items={:<8}{per_item}\n",
            s.total.as_secs_f64(),
            s.calls,
            s.items
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_record_only_when_enabled() {
        set_enabled(false);
        drop(span("test.off").items(3));
        set_enabled(true);
        drop(span("test.on").items(3));
        drop(span("test.on").items(2));
        set_enabled(false);
        let stats = take();
        assert!(!stats.iter().any(|(p, _)| *p == "test.off"));
        let on = stats.iter().find(|(p, _)| *p == "test.on").unwrap().1;
        assert_eq!((on.calls, on.items), (2, 5));
        assert!(render(&stats).contains("test.on"));
    }
}
