//! Token counting.
//!
//! Real runs must count tokens with the answer model's own tokenizer, because
//! the full-context fit mask and every budget depend on it. Until a tokenizer
//! is pinned, [`ApproxTokenizer`] gives a deterministic estimate that is
//! labelled as such everywhere it is used.

pub trait Tokenizer: Send + Sync {
    /// Stable identifier recorded in artifacts, e.g. `approx-v1`.
    fn id(&self) -> &str;
    /// Whether counts are exact for the pinned model.
    fn exact(&self) -> bool;
    fn count(&self, text: &str) -> usize;
}

/// A deterministic estimator: each run of letters or digits costs one token
/// per started four characters, and every other non-space character costs one.
/// It tends to land within a few tens of percent of BPE tokenizers on English
/// chat text. It is an estimate, never a substitute for the pinned tokenizer.
#[derive(Clone, Debug, Default)]
pub struct ApproxTokenizer;

impl Tokenizer for ApproxTokenizer {
    fn id(&self) -> &str {
        "approx-v1"
    }

    fn exact(&self) -> bool {
        false
    }

    fn count(&self, text: &str) -> usize {
        let mut tokens = 0;
        let mut run = 0usize;
        for c in text.chars() {
            if c.is_alphanumeric() {
                run += 1;
            } else {
                tokens += run.div_ceil(4);
                run = 0;
                if !c.is_whitespace() {
                    tokens += 1;
                }
            }
        }
        tokens + run.div_ceil(4)
    }
}

/// An OpenAI BPE encoding from `tiktoken-rs`, used as a named stand-in when
/// the answer model's own tokenizer is unavailable (for example a model
/// reached through a harness that hides it). Counts are real BPE counts for
/// that encoding but are never exact for a different model family, so
/// `exact()` is always false.
#[derive(Clone, Copy, Debug)]
pub struct TiktokenTokenizer {
    encoding: TiktokenEncoding,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TiktokenEncoding {
    O200kBase,
    Cl100kBase,
}

impl TiktokenTokenizer {
    pub fn new(encoding: TiktokenEncoding) -> Self {
        Self { encoding }
    }

    fn bpe(&self) -> &'static tiktoken_rs::CoreBPE {
        match self.encoding {
            TiktokenEncoding::O200kBase => tiktoken_rs::o200k_base_singleton(),
            TiktokenEncoding::Cl100kBase => tiktoken_rs::cl100k_base_singleton(),
        }
    }
}

impl Tokenizer for TiktokenTokenizer {
    fn id(&self) -> &str {
        match self.encoding {
            TiktokenEncoding::O200kBase => "tiktoken-o200k-base",
            TiktokenEncoding::Cl100kBase => "tiktoken-cl100k-base",
        }
    }

    fn exact(&self) -> bool {
        false
    }

    fn count(&self, text: &str) -> usize {
        self.bpe().encode_ordinary(text).len()
    }
}

/// Resolve a manifest tokenizer id to an implementation. Returns `None` for
/// ids this build cannot count with (callers fall back to [`ApproxTokenizer`]
/// and say so).
pub fn tokenizer_for(id: &str) -> Option<std::sync::Arc<dyn Tokenizer>> {
    match id {
        "approx-v1" => Some(std::sync::Arc::new(ApproxTokenizer)),
        "tiktoken-o200k-base" => Some(std::sync::Arc::new(TiktokenTokenizer::new(
            TiktokenEncoding::O200kBase,
        ))),
        "tiktoken-cl100k-base" => Some(std::sync::Arc::new(TiktokenTokenizer::new(
            TiktokenEncoding::Cl100kBase,
        ))),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiktoken_counts_real_bpe_tokens_and_stays_labelled_approximate() {
        let t = tokenizer_for("tiktoken-o200k-base").unwrap();
        assert_eq!(t.id(), "tiktoken-o200k-base");
        assert!(!t.exact());
        assert_eq!(t.count(""), 0);
        assert_eq!(t.count("hello world"), 2);
        assert!(tokenizer_for("some-model-tokenizer").is_none());
        assert_eq!(tokenizer_for("approx-v1").unwrap().id(), "approx-v1");
    }

    #[test]
    fn approx_counts_are_deterministic_and_monotonic() {
        let t = ApproxTokenizer;
        assert_eq!(t.count(""), 0);
        assert_eq!(t.count("hi"), 1);
        assert_eq!(t.count("hello"), 2);
        assert_eq!(t.count("hello, world!"), 2 + 1 + 2 + 1);
        let short = t.count("The meeting moved to Tuesday.");
        let long = t.count("The meeting moved to Tuesday. Then it moved again.");
        assert!(long > short);
        assert!(!t.exact());
    }
}
