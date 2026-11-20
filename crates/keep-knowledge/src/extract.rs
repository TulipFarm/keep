//! Fact extraction.
//!
//! Extractors propose [`Claim`]s from one episode. They are deliberately
//! conservative: a claim must quote an exact span of the episode, name its
//! subject (pronouns other than the speaker's "I" are not resolved), and use
//! a known predicate. Anything else is dropped. The rule-based extractor is
//! deterministic and offline; [`ChatFactExtractor`] wraps any chat model and
//! only reaches a remote provider if the caller built that model with a
//! dispatch permit.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::{NaiveDate, TimeZone, Utc};
use keep_providers::{ChatModel, ChatRequest, Message, ProviderError};
use serde::{Deserialize, Serialize};

use crate::chunking::char_find;
use crate::model::{Episode, Time, TimeBasis};

/// Predicate used for alias claims; it adds a name, not a fact.
pub const ALIAS: &str = "known_as";

/// Predicates with at most one current value per subject. A new value
/// closes or disputes the old one instead of sitting beside it.
pub const SINGLE_VALUED: &[&str] = &["lives_in", "works_at", "prefers", "role"];

/// Every predicate an extractor may emit.
pub const PREDICATES: &[&str] = &[
    "lives_in", "works_at", "prefers", "role", "works_on", "leads", ALIAS,
];

pub fn is_single_valued(predicate: &str) -> bool {
    SINGLE_VALUED.contains(&predicate)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Claim {
    pub subject: String,
    pub subject_kind: String,
    pub predicate: String,
    pub object: String,
    /// Set when the object names an entity (a place, organisation, project).
    pub object_kind: Option<String>,
    pub valid_from: Option<Time>,
    pub basis: TimeBasis,
    /// Exact text from the episode that states the claim.
    pub quote: String,
}

#[async_trait]
pub trait FactExtractor: Send + Sync {
    /// Name and version, recorded on every piece of evidence.
    fn id(&self) -> &str;
    async fn extract(&self, episode: &Episode) -> Result<Vec<Claim>, ProviderError>;
}

/// A claim that passed validation, with its character span.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedClaim {
    pub claim: Claim,
    pub start: usize,
    pub end: usize,
}

/// Keep only claims an auditor could verify against the episode.
pub fn validate(episode: &Episode, claims: Vec<Claim>) -> Vec<CheckedClaim> {
    claims
        .into_iter()
        .filter_map(|mut c| {
            c.subject = c.subject.trim().to_string();
            c.object = c.object.trim().to_string();
            if c.subject.is_empty() || c.object.is_empty() || c.quote.trim().is_empty() {
                return None;
            }
            if !PREDICATES.contains(&c.predicate.as_str()) {
                return None;
            }
            let quote_lc = c.quote.to_lowercase();
            if !quote_lc.contains(&c.object.to_lowercase()) {
                return None;
            }
            let speaker = c.subject == episode.speaker;
            if !speaker && !quote_lc.contains(&c.subject.to_lowercase()) {
                return None;
            }
            let (start, end) = char_find(&episode.text, &c.quote)?;
            Some(CheckedClaim {
                claim: c,
                start,
                end,
            })
        })
        .collect()
}

enum ObjectMode {
    Entity(&'static str),
    Value,
    Alias,
}

const PATTERNS: &[(&str, &str, ObjectMode)] = &[
    ("lives in", "lives_in", ObjectMode::Entity("place")),
    ("live in", "lives_in", ObjectMode::Entity("place")),
    ("moved to", "lives_in", ObjectMode::Entity("place")),
    ("relocated to", "lives_in", ObjectMode::Entity("place")),
    ("works at", "works_at", ObjectMode::Entity("organization")),
    ("work at", "works_at", ObjectMode::Entity("organization")),
    ("joined", "works_at", ObjectMode::Entity("organization")),
    ("works on", "works_on", ObjectMode::Entity("project")),
    ("work on", "works_on", ObjectMode::Entity("project")),
    ("leads", "leads", ObjectMode::Entity("project")),
    ("prefers", "prefers", ObjectMode::Value),
    ("prefer", "prefers", ObjectMode::Value),
    ("known as", ALIAS, ObjectMode::Alias),
];

const SKIP_BEFORE: &[&str] = &[
    "now",
    "just",
    "recently",
    "also",
    "still",
    "have",
    "has",
    "had",
    "finally",
    "currently",
    "then",
    "really",
    "actually",
    "usually",
    "always",
    "now,",
];

const NOT_NAMES: &[&str] = &[
    "Yesterday",
    "Today",
    "Tomorrow",
    "Now",
    "Then",
    "Also",
    "And",
    "But",
    "So",
    "Last",
    "Next",
    "She",
    "He",
    "They",
    "We",
    "It",
    "The",
    "This",
    "That",
    "My",
    "Our",
    "Their",
    "His",
    "Her",
    "You",
    "Recently",
    "Since",
    "In",
    "On",
    "At",
    "When",
    "After",
    "Before",
];

const VALUE_STOPS: &[&str] = &[
    "over", "to", "because", "and", "but", "when", "since", "if", "so", "than",
];

/// Deterministic pattern extractor. It understands a handful of phrasings
/// ("lives in", "moved to", "works at", "joined", "works on", "leads",
/// "prefers", "known as") and nothing else. It never guesses: no pronoun
/// resolution, lowercase place names are ignored, and the assistant's own
/// "I" statements are skipped.
#[derive(Clone, Debug, Default)]
pub struct RuleExtractor;

impl RuleExtractor {
    pub const ID: &'static str = "rules-v1";

    pub fn extract_now(&self, episode: &Episode) -> Vec<Claim> {
        let mut out = Vec::new();
        for sentence in sentences(&episode.text) {
            let words: Vec<&str> = sentence.split_whitespace().collect();
            let lower: Vec<String> = words.iter().map(|w| w.to_lowercase()).collect();
            let mut emitted = std::collections::BTreeSet::new();
            for (phrase, predicate, mode) in PATTERNS {
                if emitted.contains(predicate) {
                    continue;
                }
                let pw: Vec<&str> = phrase.split(' ').collect();
                let Some(at) = (0..lower.len()).find(|&i| {
                    i + pw.len() <= lower.len()
                        && pw.iter().enumerate().all(|(j, p)| {
                            let w = lower[i + j].trim_end_matches([',', '.', '!', '?', ';']);
                            // the last pattern word must match exactly, so
                            // "prefer" does not fire inside "preferably".
                            w == *p
                        })
                }) else {
                    continue;
                };
                let Some((subject, subject_kind)) = subject_before(&words[..at], episode) else {
                    continue;
                };
                let rest = &words[at + pw.len()..];
                let object = match mode {
                    ObjectMode::Entity(_) | ObjectMode::Alias => proper_noun(rest),
                    ObjectMode::Value => value_phrase(rest),
                };
                let Some(object) = object else { continue };
                let (valid_from, basis) = match since_date(rest) {
                    Some(d) => (Some(d), TimeBasis::Explicit),
                    None => match episode.occurred_at {
                        Some(t) => (Some(t), TimeBasis::Observed),
                        None => (None, TimeBasis::Unknown),
                    },
                };
                out.push(Claim {
                    subject,
                    subject_kind,
                    predicate: (*predicate).into(),
                    object,
                    object_kind: match mode {
                        ObjectMode::Entity(k) => Some((*k).into()),
                        _ => None,
                    },
                    valid_from,
                    basis,
                    quote: sentence.to_string(),
                });
                emitted.insert(*predicate);
            }
        }
        out
    }
}

#[async_trait]
impl FactExtractor for RuleExtractor {
    fn id(&self) -> &str {
        Self::ID
    }
    async fn extract(&self, episode: &Episode) -> Result<Vec<Claim>, ProviderError> {
        Ok(self.extract_now(episode))
    }
}

fn sentences(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let bytes: Vec<(usize, char)> = text.char_indices().collect();
    for (k, (i, c)) in bytes.iter().enumerate() {
        let next_ws = bytes.get(k + 1).is_none_or(|(_, n)| n.is_whitespace());
        if matches!(c, '.' | '!' | '?' | '\n') && next_ws {
            let end = i + c.len_utf8();
            let s = text[start..end].trim();
            if !s.is_empty() {
                out.push(s);
            }
            start = end;
        }
    }
    let s = text[start..].trim();
    if !s.is_empty() {
        out.push(s);
    }
    out
}

fn clean(w: &str) -> &str {
    w.trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '\'')
}

fn is_name_word(w: &str) -> bool {
    let w = clean(w);
    w.chars().next().is_some_and(|c| c.is_uppercase()) && !NOT_NAMES.contains(&w)
}

fn subject_before(words: &[&str], episode: &Episode) -> Option<(String, String)> {
    let mut i = words.len();
    while i > 0 && SKIP_BEFORE.contains(&words[i - 1].to_lowercase().as_str()) {
        i -= 1;
    }
    if i == 0 {
        return None;
    }
    let last = clean(words[i - 1]);
    if last == "I" {
        if episode.speaker.eq_ignore_ascii_case("assistant") || episode.speaker.trim().is_empty() {
            return None;
        }
        return Some((episode.speaker.clone(), "person".into()));
    }
    let mut j = i;
    while j > 0 && is_name_word(words[j - 1]) && !words[j - 1].ends_with([',', '.', ';', ':']) {
        j -= 1;
    }
    // A trailing comma ends a name: "Mira Sato, now" is handled by the skip
    // list; "Mira Sato, known as Mimi" keeps "Mira Sato".
    if j == i && i > 0 && is_name_word(words[i - 1]) {
        j = i - 1;
        while j > 0 && is_name_word(words[j - 1]) && !words[j - 1].ends_with([',', '.', ';']) {
            j -= 1;
        }
    }
    if j == i {
        return None;
    }
    // "Tomas Reyes, known as Tom, joined": the subject is the full name.
    if j >= 2
        && words[j - 2].eq_ignore_ascii_case("known")
        && words[j - 1].eq_ignore_ascii_case("as")
    {
        return subject_before(&words[..j - 2], episode);
    }
    let name: Vec<&str> = words[j..i].iter().map(|w| clean(w)).collect();
    Some((name.join(" "), "person".into()))
}

fn proper_noun(rest: &[&str]) -> Option<String> {
    let mut name = Vec::new();
    for w in rest {
        if !is_name_word(w) {
            break;
        }
        name.push(clean(w));
        if w.ends_with([',', '.', '!', '?', ';', ':']) {
            break;
        }
    }
    (!name.is_empty()).then(|| name.join(" "))
}

fn value_phrase(rest: &[&str]) -> Option<String> {
    let mut out = Vec::new();
    for w in rest.iter().take(4) {
        let c = clean(w);
        if c.is_empty() || VALUE_STOPS.contains(&c.to_lowercase().as_str()) {
            break;
        }
        out.push(c);
        if w.ends_with([',', '.', '!', '?', ';', ':']) {
            break;
        }
    }
    (!out.is_empty()).then(|| out.join(" "))
}

fn since_date(rest: &[&str]) -> Option<Time> {
    let i = rest.iter().position(|w| w.eq_ignore_ascii_case("since"))?;
    let a = rest.get(i + 1).map(|w| clean(w))?;
    let year = |s: &str| s.parse::<i32>().ok().filter(|y| (1900..=2200).contains(y));
    let (y, m) = if let Some(y) = year(a) {
        (y, 1)
    } else {
        let m = month(a)?;
        (year(rest.get(i + 2).map(|w| clean(w))?)?, m)
    };
    let d = NaiveDate::from_ymd_opt(y, m, 1)?;
    Some(Utc.from_utc_datetime(&d.and_hms_opt(0, 0, 0)?))
}

fn month(s: &str) -> Option<u32> {
    const M: [&str; 12] = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ];
    let s = s.to_lowercase();
    M.iter().position(|m| *m == s).map(|i| i as u32 + 1)
}

/// Prompt sent by [`ChatFactExtractor`]. The reply must be JSON.
pub const EXTRACTION_PROMPT: &str = "You extract durable facts from one message. \
Only report facts the message states directly; never infer. Use only these \
predicates: lives_in, works_at, prefers, role, works_on, leads, known_as. \
\"quote\" must be copied exactly from the message. Use the speaker's name as \
the subject for first-person statements. Reply with JSON only: \
{\"facts\":[{\"subject\":\"\",\"predicate\":\"\",\"object\":\"\",\
\"object_is_entity\":true,\"quote\":\"\",\"valid_from\":\"YYYY-MM-DD or null\"}]}";

/// Model-backed extractor. Output passes through the same [`validate`] gate
/// as the rule extractor, so a model cannot invent evidence.
pub struct ChatFactExtractor {
    model: Arc<dyn ChatModel>,
    id: String,
    max_output_tokens: u32,
}

impl ChatFactExtractor {
    pub fn new(model: Arc<dyn ChatModel>) -> Self {
        let d = model.descriptor();
        let id = format!("chat:{}/{}@{}", d.provider, d.model, d.revision);
        Self {
            model,
            id,
            max_output_tokens: 512,
        }
    }

    pub fn request(&self, episode: &Episode) -> ChatRequest {
        let when = episode
            .occurred_at
            .map_or_else(|| "unknown".to_string(), |t| t.date_naive().to_string());
        let user = format!(
            "Speaker: {}\nDate: {when}\nMessage:\n{}",
            episode.speaker, episode.text
        );
        let estimated = (EXTRACTION_PROMPT.len() + user.len()).div_ceil(4) as u64;
        ChatRequest {
            messages: vec![Message::system(EXTRACTION_PROMPT), Message::user(user)],
            max_output_tokens: self.max_output_tokens,
            temperature: 0.0,
            seed: Some(0),
            estimated_input_tokens: estimated,
        }
    }

    pub fn parse(episode: &Episode, reply: &str) -> Result<Vec<Claim>, ProviderError> {
        #[derive(Deserialize)]
        struct Out {
            facts: Vec<Raw>,
        }
        #[derive(Deserialize)]
        struct Raw {
            subject: String,
            predicate: String,
            object: String,
            #[serde(default)]
            object_is_entity: bool,
            quote: String,
            #[serde(default)]
            valid_from: Option<String>,
        }
        let body = reply
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```");
        let out: Out = serde_json::from_str(body).map_err(|e| ProviderError::Malformed {
            message: format!("extraction reply is not the expected JSON: {e}"),
        })?;
        Ok(out
            .facts
            .into_iter()
            .map(|r| {
                let explicit = r
                    .valid_from
                    .as_deref()
                    .and_then(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d").ok())
                    .and_then(|d| d.and_hms_opt(0, 0, 0))
                    .map(|d| Utc.from_utc_datetime(&d));
                let (valid_from, basis) = match (explicit, episode.occurred_at) {
                    (Some(d), _) => (Some(d), TimeBasis::Explicit),
                    (None, Some(t)) => (Some(t), TimeBasis::Observed),
                    (None, None) => (None, TimeBasis::Unknown),
                };
                Claim {
                    subject: r.subject,
                    subject_kind: "person".into(),
                    predicate: r.predicate,
                    object: r.object,
                    object_kind: r.object_is_entity.then(|| "thing".into()),
                    valid_from,
                    basis,
                    quote: r.quote,
                }
            })
            .collect())
    }
}

#[async_trait]
impl FactExtractor for ChatFactExtractor {
    fn id(&self) -> &str {
        &self.id
    }
    async fn extract(&self, episode: &Episode) -> Result<Vec<Claim>, ProviderError> {
        let reply = self.model.complete(&self.request(episode)).await?;
        Self::parse(episode, &reply.text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use keep_providers::mock::FnChat;

    fn ep(speaker: &str, text: &str) -> Episode {
        Episode {
            corpus: "c".into(),
            id: "ep".into(),
            source_id: "s".into(),
            partition: "p".into(),
            ordinal: 0,
            speaker: speaker.into(),
            occurred_at: Some(Utc.with_ymd_and_hms(2024, 5, 1, 0, 0, 0).unwrap()),
            recorded_at: Utc.with_ymd_and_hms(2024, 5, 1, 0, 0, 0).unwrap(),
            text: text.into(),
            content_hash: String::new(),
        }
    }

    fn triples(e: &Episode) -> Vec<(String, String, String)> {
        validate(e, RuleExtractor.extract_now(e))
            .into_iter()
            .map(|c| (c.claim.subject, c.claim.predicate, c.claim.object))
            .collect()
    }

    #[test]
    fn rules_extract_named_and_first_person_facts() {
        let e = ep(
            "Mira Sato",
            "Hi there. I just moved to Porto. Tomas Reyes works at Northwind Labs, \
             and he leads Project Heron. I prefer green tea over coffee.",
        );
        assert_eq!(
            triples(&e),
            vec![
                ("Mira Sato".into(), "lives_in".into(), "Porto".into()),
                (
                    "Tomas Reyes".into(),
                    "works_at".into(),
                    "Northwind Labs".into()
                ),
                ("Mira Sato".into(), "prefers".into(), "green tea".into()),
            ]
        );
    }

    #[test]
    fn rules_refuse_to_guess() {
        // Pronoun subjects, lowercase places and the assistant's "I".
        assert!(triples(&ep("user", "She moved to Lisbon. I live in a flat.")).is_empty());
        assert!(triples(&ep("assistant", "I live in the cloud, Lisbon.")).is_empty());
    }

    #[test]
    fn explicit_since_dates_beat_observation_time() {
        let e = ep(
            "user",
            "I have lived here a while; I live in Lisbon since March 2019.",
        );
        let c = &RuleExtractor.extract_now(&e)[0];
        assert_eq!(c.basis, TimeBasis::Explicit);
        assert_eq!(c.valid_from.unwrap().date_naive().to_string(), "2019-03-01");
    }

    #[test]
    fn aliases_are_claims_too() {
        let e = ep("user", "Tomas Reyes, known as Tom, joined Northwind.");
        let t = triples(&e);
        assert_eq!(
            t[0],
            ("Tomas Reyes".into(), "works_at".into(), "Northwind".into())
        );
        assert_eq!(t[1], ("Tomas Reyes".into(), ALIAS.into(), "Tom".into()));
    }

    #[tokio::test]
    async fn chat_extractor_output_is_validated_against_the_episode() {
        let model = FnChat::new("x", |_| {
            Ok(r#"```json
{"facts":[
 {"subject":"Mira Sato","predicate":"lives_in","object":"Porto","object_is_entity":true,"quote":"I just moved to Porto."},
 {"subject":"Mira Sato","predicate":"lives_in","object":"Paris","object_is_entity":true,"quote":"I moved to Paris."},
 {"subject":"Mira Sato","predicate":"owns","object":"Porto","quote":"I just moved to Porto."}
]}
```"#
                .into())
        });
        let x = ChatFactExtractor::new(Arc::new(model));
        let e = ep("Mira Sato", "Hello. I just moved to Porto.");
        let claims = validate(&e, x.extract(&e).await.unwrap());
        assert_eq!(
            claims.len(),
            1,
            "invented quote and unknown predicate dropped"
        );
        assert_eq!(claims[0].claim.object, "Porto");
        assert_eq!((claims[0].start, claims[0].end), (7, 29));
        assert!(x.id().starts_with("chat:mock/x"));
        assert!(matches!(
            ChatFactExtractor::parse(&e, "nope"),
            Err(ProviderError::Malformed { .. })
        ));
    }
}
