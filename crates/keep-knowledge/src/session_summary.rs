//! One short model-written summary per session.
//!
//! Facts ([`crate::session_extract`]) answer "what is X?" well but lose
//! how things in one session hang together: who did what with whom, and
//! which events belong to the same occasion. A few sentences per session
//! keep that, and help questions that join several sessions.
//!
//! The session is rendered exactly as for fact extraction, so the content
//! hash matches; replies are cached in the same on-disk cache under their
//! own prompt version, so a summary and a fact extraction of one session
//! never collide.

use std::sync::Arc;

use keep_providers::{ChatModel, ChatRequest, Message, ProviderError, Usage};

use crate::session_extract::{
    CachedReply, ExtractionCache, SessionInput, cache_key, render_session, session_hash,
};

/// Bumped whenever the prompt or the session rendering changes.
pub const SUMMARY_PROMPT_VERSION: &str = "session-summary-v1";

/// Summaries longer than this many characters are cut.
pub const MAX_SUMMARY_CHARS: usize = 1_200;

pub const SUMMARY_PROMPT: &str = "You read one dated conversation session and write a short summary of it \
for a memory system that will later answer questions spanning many sessions.\n\
Write 2 to 4 plain sentences, at most 90 words, with no heading or list. Say who did, \
planned or felt what, naming each person, and keep the specifics: names, titles, places, \
objects, numbers and dates. Turn relative times into absolute dates from the session date. \
Attribute every point to the person it is about; never merge two people's experiences. \
Skip greetings and general advice. If the session says nothing about the people, reply: \
Nothing about the people.";

/// Bumped whenever [`DOCUMENT_SUMMARY_PROMPT`] changes.
pub const DOCUMENT_SUMMARY_PROMPT_VERSION: &str = "document-summary-v1";

/// For material that is not a conversation between people, such as a
/// document a host application ingested.
pub const DOCUMENT_SUMMARY_PROMPT: &str = "You read some source material (a document or the \
turns of a conversation) and write a short summary of it for a reader who has not seen it.\n\
Write 2 to 6 plain sentences, at most 150 words, with no heading or list. Keep the main \
points and the specifics: names, titles, places, numbers and dates. Use only what the \
material says; never add outside facts or advice. Reply with the summary only.";

/// What one session produced.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionSummary {
    pub text: String,
    pub cached: bool,
    pub usage: Usage,
    pub ai_credits: Option<f64>,
}

/// The chat request for `session`.
pub fn summary_request(session: &SessionInput, max_output_tokens: u32) -> ChatRequest {
    summary_request_with(SUMMARY_PROMPT, session, max_output_tokens)
}

/// The chat request for `session` under the system `prompt`.
pub fn summary_request_with(
    prompt: &str,
    session: &SessionInput,
    max_output_tokens: u32,
) -> ChatRequest {
    let user = render_session(session);
    let estimated = (prompt.len() + user.len()).div_ceil(4) as u64;
    ChatRequest {
        messages: vec![Message::system(prompt), Message::user(user)],
        max_output_tokens,
        temperature: 0.0,
        seed: Some(0),
        estimated_input_tokens: estimated,
    }
}

/// A reply as one line of text: whitespace collapsed, cut at
/// [`MAX_SUMMARY_CHARS`]. Empty when the session had nothing to say.
pub fn clean_reply(reply: &str) -> String {
    let text = reply.split_whitespace().collect::<Vec<_>>().join(" ");
    let text = text.trim_matches('"').trim();
    if text.is_empty()
        || text
            .trim_end_matches('.')
            .eq_ignore_ascii_case("nothing about the people")
    {
        return String::new();
    }
    let cut: String = text.chars().take(MAX_SUMMARY_CHARS).collect();
    if cut.len() < text.len() {
        format!("{cut} [...]")
    } else {
        cut
    }
}

/// Summarises whole sessions through a chat model, with an optional
/// on-disk reply cache shared with fact extraction.
pub struct SessionSummarizer {
    model: Option<Arc<dyn ChatModel>>,
    model_id: String,
    cache: Option<ExtractionCache>,
    max_output_tokens: u32,
    prompt: &'static str,
    prompt_version: &'static str,
}

impl SessionSummarizer {
    /// `model_id` keys the cache; without a model only cached replies are
    /// served.
    pub fn new(
        model: Option<Arc<dyn ChatModel>>,
        model_id: impl Into<String>,
        cache: Option<ExtractionCache>,
        max_output_tokens: u32,
    ) -> Self {
        Self {
            model,
            model_id: model_id.into(),
            cache,
            max_output_tokens,
            prompt: SUMMARY_PROMPT,
            prompt_version: SUMMARY_PROMPT_VERSION,
        }
    }

    /// Summarise with [`DOCUMENT_SUMMARY_PROMPT`] instead of the
    /// people-focused session prompt.
    pub fn for_documents(mut self) -> Self {
        self.prompt = DOCUMENT_SUMMARY_PROMPT;
        self.prompt_version = DOCUMENT_SUMMARY_PROMPT_VERSION;
        self
    }

    pub fn max_output_tokens(&self) -> u32 {
        self.max_output_tokens
    }

    pub fn has_model(&self) -> bool {
        self.model.is_some()
    }

    /// Label recorded in context methods.
    pub fn label(&self) -> String {
        format!("session:{}:{}", self.model_id, self.prompt_version)
    }

    pub fn key(&self, session: &SessionInput) -> String {
        cache_key(&self.model_id, self.prompt_version, &session_hash(session))
    }

    pub fn is_cached(&self, session: &SessionInput) -> bool {
        self.cache
            .as_ref()
            .is_some_and(|c| c.get(&self.key(session)).is_some())
    }

    /// The summary from the cache alone; `None` on a miss.
    pub fn cached(&self, session: &SessionInput) -> Option<String> {
        let hit = self.cache.as_ref()?.get(&self.key(session))?;
        Some(clean_reply(&hit.text))
    }

    /// The summary from the cache when present, else from the model (and
    /// then cached). Without a model a cache miss is an error.
    pub async fn summarize(&self, session: &SessionInput) -> Result<SessionSummary, ProviderError> {
        if let Some(text) = self.cached(session) {
            return Ok(SessionSummary {
                text,
                cached: true,
                usage: Usage::default(),
                ai_credits: None,
            });
        }
        let Some(model) = &self.model else {
            return Err(ProviderError::Config {
                message: format!(
                    "no cached summary for session {} and no model to call",
                    session.id
                ),
            });
        };
        let content_hash = session_hash(session);
        let response = model
            .complete(&summary_request_with(
                self.prompt,
                session,
                self.max_output_tokens,
            ))
            .await?;
        let ai_credits = response.billing.as_ref().map(|b| b.ai_credits);
        if let Some(cache) = &self.cache {
            cache
                .put(
                    &cache_key(&self.model_id, self.prompt_version, &content_hash),
                    &CachedReply {
                        model: self.model_id.clone(),
                        prompt_version: self.prompt_version.into(),
                        content_hash,
                        text: response.text.clone(),
                        usage: response.usage,
                        ai_credits,
                    },
                )
                .map_err(|e| ProviderError::Transport {
                    message: format!("could not write the summary cache: {e}"),
                })?;
        }
        Ok(SessionSummary {
            text: clean_reply(&response.text),
            cached: false,
            usage: response.usage,
            ai_credits,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session_extract::{SessionExtractor, SessionTurn};
    use std::path::Path;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn session() -> SessionInput {
        SessionInput {
            id: "s1".into(),
            date: None,
            turns: vec![SessionTurn {
                id: "t1".into(),
                speaker: "Ilse".into(),
                assistant: false,
                text: "I adopted a greyhound called Pepper yesterday.".into(),
            }],
        }
    }

    #[test]
    fn replies_are_cleaned_and_empty_sessions_say_nothing() {
        assert_eq!(
            clean_reply("  Ilse adopted\n a dog.  "),
            "Ilse adopted a dog."
        );
        assert_eq!(clean_reply("Nothing about the people."), "");
        let long = "a".repeat(MAX_SUMMARY_CHARS + 5);
        assert!(clean_reply(&long).ends_with(" [...]"));
    }

    #[tokio::test]
    async fn summaries_are_cached_apart_from_fact_extractions() {
        use keep_providers::mock::FnChat;
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/test-summary-cache")
            .join(std::process::id().to_string());
        let _ = std::fs::remove_dir_all(&dir);
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = calls.clone();
        let model = FnChat::new("summary", move |_| {
            seen.fetch_add(1, Ordering::SeqCst);
            Ok("Ilse adopted a greyhound called Pepper.".into())
        });
        let s = session();
        let x = SessionSummarizer::new(
            Some(Arc::new(model)),
            "mock/m@1",
            Some(ExtractionCache::new(&dir)),
            200,
        );
        assert!(!x.is_cached(&s));
        let first = x.summarize(&s).await.unwrap();
        assert!(!first.cached);
        let second = x.summarize(&s).await.unwrap();
        assert!(second.cached);
        assert_eq!(first.text, second.text);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        // Same model and session, different prompt version: no collision.
        let facts = SessionExtractor::new(None, "mock/m@1", Some(ExtractionCache::new(&dir)), 200);
        assert_ne!(facts.key(&s), x.key(&s));
        assert!(!facts.is_cached(&s));
        // Cache-only: the hit is served, a miss is refused.
        let offline =
            SessionSummarizer::new(None, "mock/m@1", Some(ExtractionCache::new(&dir)), 200);
        assert_eq!(offline.cached(&s).as_deref(), Some(first.text.as_str()));
        let mut other = s.clone();
        other.turns[0].text = "Something else.".into();
        assert!(offline.summarize(&other).await.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
