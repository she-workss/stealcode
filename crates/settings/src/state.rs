use std::sync::atomic::{AtomicUsize, Ordering};

use rustc_hash::FxHashMap;
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct SessionData {
    pub conversation_id: String,
    pub messages: Vec<Value>,
}

#[derive(Debug)]
pub struct AppState {
    pub chat_sessions: FxHashMap<String, FxHashMap<String, SessionData>>,
    pub api_key_usage: FxHashMap<String, Vec<f64>>,
    pub model_usage_stats: FxHashMap<String, u64>,
    pub current_token_index: AtomicUsize,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            chat_sessions: FxHashMap::default(),
            api_key_usage: FxHashMap::default(),
            model_usage_stats: FxHashMap::default(),
            current_token_index: AtomicUsize::new(0),
        }
    }
}

impl AppState {
    /// Atomically increments the token index, wrapping modulo `pool_len`;
    /// returns `0` for an empty pool. `Relaxed` ordering suffices - we only
    /// need `fetch_add` atomicity, not any cross-thread ordering guarantee.
    pub fn increment_token_index(&self, pool_len: usize) -> usize {
        if pool_len == 0 {
            return 0;
        }
        let old = self.current_token_index.fetch_add(1, Ordering::Relaxed);
        old % pool_len
    }
}

/// Returns the current wall-clock time as fractional seconds since the Unix
/// epoch.  Returns `0.0` if the system clock is somehow before the epoch.
#[must_use]
pub fn now_secs() -> f64 {
    std::time::UNIX_EPOCH
        .elapsed()
        .unwrap_or_default()
        .as_secs_f64()
}

#[cfg(test)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn increment_token_index_round_robins() {
        let state = AppState::default();
        assert_eq!(state.increment_token_index(3), 0);
        assert_eq!(state.increment_token_index(3), 1);
        assert_eq!(state.increment_token_index(3), 2);
        assert_eq!(state.increment_token_index(3), 0); // wraps
    }

    #[test]
    fn increment_token_index_empty_pool_returns_zero() {
        let state = AppState::default();
        assert_eq!(state.increment_token_index(0), 0);
        assert_eq!(state.increment_token_index(0), 0);
    }

    #[test]
    fn increment_token_index_pool_of_one_always_zero() {
        let state = AppState::default();
        for _ in 0..5 {
            assert_eq!(state.increment_token_index(1), 0);
        }
    }

    #[test]
    fn now_secs_is_positive() {
        assert!(now_secs() > 0.0);
    }
}
