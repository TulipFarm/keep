//! Client-side pacing for a deployment with a tokens-per-minute (TPM) and
//! requests-per-minute (RPM) quota, such as an Azure OpenAI deployment.
//!
//! Azure admits a request against the quota using its prompt plus its
//! `max_tokens`/`max_completion_tokens`, and rejects it with HTTP 429 when
//! the minute's quota is spent. Pacing the calls below the quota keeps 429s
//! rare; when one still arrives, [`RateLimiter::pause`] holds every caller
//! sharing the limiter until the provider's `Retry-After` has passed.
//!
//! The window is a sliding 60 seconds. A single request larger than the whole
//! TPM quota is admitted only into an empty window, so it can never wait
//! forever; whether the provider then accepts it is the provider's decision,
//! and a refusal is recorded like any other failure.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::time::Instant;

const WINDOW: Duration = Duration::from_secs(60);

/// What pacing cost a run, for its artifacts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RateLimitStats {
    pub tokens_per_minute: u64,
    pub requests_per_minute: u64,
    /// Requests admitted through the limiter.
    pub admitted: u64,
    /// Requests that had to wait for quota.
    pub waited: u64,
    /// Total seconds callers spent waiting for quota or a `Retry-After`.
    pub wait_secs: f64,
    /// HTTP 429 replies received by clients sharing this limiter.
    pub rejected_429: u64,
    /// Requests larger than the whole TPM quota (admitted alone).
    pub oversized: u64,
}

#[derive(Debug)]
struct State {
    window: VecDeque<(Instant, u64)>,
    tokens_in_window: u64,
    paused_until: Option<Instant>,
    stats: RateLimitStats,
}

#[derive(Debug)]
pub struct RateLimiter {
    tokens_per_minute: u64,
    requests_per_minute: u64,
    state: Mutex<State>,
}

impl RateLimiter {
    /// Zero for either limit means "no limit" on that dimension.
    pub fn new(tokens_per_minute: u64, requests_per_minute: u64) -> Self {
        Self {
            tokens_per_minute,
            requests_per_minute,
            state: Mutex::new(State {
                window: VecDeque::new(),
                tokens_in_window: 0,
                paused_until: None,
                stats: RateLimitStats {
                    tokens_per_minute,
                    requests_per_minute,
                    ..RateLimitStats::default()
                },
            }),
        }
    }

    pub fn stats(&self) -> RateLimitStats {
        self.state.lock().expect("rate limit lock").stats
    }

    /// Wait until `tokens` (prompt plus output allowance) fit the window,
    /// then charge them. Returns how long the caller waited.
    pub async fn acquire(&self, tokens: u64) -> Duration {
        let started = Instant::now();
        let mut waited = false;
        loop {
            let wait = {
                let mut s = self.state.lock().expect("rate limit lock");
                let now = Instant::now();
                while s.window.front().is_some_and(|(t, _)| now - *t >= WINDOW) {
                    let (_, n) = s.window.pop_front().expect("front exists");
                    s.tokens_in_window -= n;
                }
                let paused = s.paused_until.filter(|until| *until > now);
                let rpm_ok = self.requests_per_minute == 0
                    || (s.window.len() as u64) < self.requests_per_minute;
                let tpm_ok = self.tokens_per_minute == 0
                    || s.tokens_in_window + tokens <= self.tokens_per_minute
                    || s.window.is_empty();
                match paused {
                    None if rpm_ok && tpm_ok => {
                        if self.tokens_per_minute > 0 && tokens > self.tokens_per_minute {
                            s.stats.oversized += 1;
                        }
                        s.window.push_back((now, tokens));
                        s.tokens_in_window += tokens;
                        s.stats.admitted += 1;
                        let elapsed = now - started;
                        if waited {
                            s.stats.waited += 1;
                            s.stats.wait_secs += elapsed.as_secs_f64();
                        }
                        return elapsed;
                    }
                    Some(until) => until - now,
                    None => {
                        // Wait for the oldest entry to leave the window.
                        let oldest = s.window.front().map_or(now, |(t, _)| *t);
                        (oldest + WINDOW).saturating_duration_since(now)
                    }
                }
            };
            waited = true;
            tokio::time::sleep(wait.max(Duration::from_millis(10))).await;
        }
    }

    /// Record a 429 and hold every caller until `retry_after` has passed.
    pub fn pause(&self, retry_after: Option<Duration>) {
        let mut s = self.state.lock().expect("rate limit lock");
        s.stats.rejected_429 += 1;
        if let Some(d) = retry_after {
            let until = Instant::now() + d;
            s.paused_until = Some(s.paused_until.map_or(until, |p| p.max(until)));
        }
    }
}

/// Parse the wait an HTTP 429 asks for: `retry-after-ms` (milliseconds,
/// Azure OpenAI) first, then `retry-after` in seconds. HTTP-date values are
/// not used.
pub fn retry_after_ms(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    let get = |name: &str| {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
    };
    get("retry-after-ms")
        .map(|ms| ms.ceil() as u64)
        .or_else(|| get("retry-after").map(|s| (s * 1000.0).ceil() as u64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(start_paused = true)]
    async fn requests_wait_for_token_quota_to_leave_the_window() {
        let l = RateLimiter::new(100, 0);
        assert_eq!(l.acquire(60).await, Duration::ZERO);
        assert_eq!(l.acquire(40).await, Duration::ZERO);
        // The window is full: the next request waits a full minute.
        let waited = l.acquire(10).await;
        assert!(waited >= WINDOW && waited < WINDOW + Duration::from_secs(1));
        let s = l.stats();
        assert_eq!((s.admitted, s.waited, s.oversized), (3, 1, 0));
    }

    #[tokio::test(start_paused = true)]
    async fn an_oversized_request_is_admitted_alone_and_counted() {
        let l = RateLimiter::new(100, 0);
        assert_eq!(l.acquire(250).await, Duration::ZERO);
        let waited = l.acquire(1).await;
        assert!(waited >= WINDOW);
        assert_eq!(l.stats().oversized, 1);
    }

    #[tokio::test(start_paused = true)]
    async fn request_quota_and_retry_after_pauses_hold_callers() {
        let l = RateLimiter::new(0, 2);
        l.acquire(1).await;
        l.acquire(1).await;
        assert!(l.acquire(1).await >= WINDOW);

        let l = RateLimiter::new(0, 0);
        l.pause(Some(Duration::from_secs(7)));
        let waited = l.acquire(1).await;
        assert!(waited >= Duration::from_secs(7) && waited < Duration::from_secs(8));
        assert_eq!(l.stats().rejected_429, 1);
    }

    #[test]
    fn retry_after_prefers_milliseconds_and_ignores_dates() {
        use reqwest::header::{HeaderMap, HeaderValue};
        let mut h = HeaderMap::new();
        h.insert("retry-after", HeaderValue::from_static("3"));
        assert_eq!(retry_after_ms(&h), Some(3000));
        h.insert("retry-after-ms", HeaderValue::from_static("1250.5"));
        assert_eq!(retry_after_ms(&h), Some(1251));
        let mut d = HeaderMap::new();
        d.insert(
            "retry-after",
            HeaderValue::from_static("Wed, 21 Oct 2015 07:28:00 GMT"),
        );
        assert_eq!(retry_after_ms(&d), None);
    }
}
