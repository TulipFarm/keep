//! Model-driven fact extraction over whole sessions.
//!
//! [`crate::extract`] works one episode at a time with a closed predicate
//! list and exact-quote validation. This module reads a whole session
//! (every turn, numbered, with the session date) in one model call and
//! returns open-vocabulary facts: a subject, a snake_case attribute, a
//! value, when it became true, the turns that state it and a confidence.
//!
//! Dates are resolved deterministically: the model copies the time phrase
//! it saw ("last Tuesday", "two weeks ago") and [`resolve_when`] turns it
//! into a calendar date from the session date. The model's own absolute
//! date is used only when the phrase cannot be parsed, and the session date
//! when neither is given.
//!
//! Replies are cached on disk by [`cache_key`]: the model, the prompt
//! version and the hash of the exact session text sent, so rerunning a
//! benchmark never pays twice for the same call.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::{Datelike, Duration, Months, NaiveDate, TimeZone, Utc, Weekday};
use keep_providers::{ChatModel, ChatRequest, Message, ProviderError, Usage};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::model::{Time, TimeBasis};

/// Bumped whenever the prompt or the rendering of a session changes, so
/// cached replies from an older prompt are never reused.
pub const SESSION_PROMPT_VERSION: &str = "session-facts-v2";

/// Turns longer than this many characters are cut when sent for extraction.
pub const MAX_TURN_CHARS: usize = 4_000;
/// Assistant turns are cut harder: facts are about the people, and the
/// assistant's long answers cost tokens without adding many.
pub const MAX_ASSISTANT_CHARS: usize = 600;
/// Facts kept from one session, in reply order.
pub const MAX_FACTS: usize = 40;

pub const SESSION_PROMPT: &str = r#"You read one dated conversation session and list the facts it states about the people in it, so that a memory system can later answer questions such as "what is my current X?", "how many X have I done?", "when did I do X?" and "how long ago was X?".

Reply with JSON only: {"facts": [ ... ]}. Each fact is an object with:
- "subject": who the fact is about. For a speaker talking about themselves use the speaker label exactly as shown (for example "user" or "Caroline"). For someone else use their name, or a short description such as "user's sister".
- "attribute": a short, generic snake_case name for the property, such as job_title, home_city, commute_minutes, personal_best_5k, gym_schedule, pet, bought_item, visited_place, attended_event, read_book. Use the most generic name, so that a later update of the same property gets the same attribute.
- "value": the value, short but specific. Keep numbers, units, amounts, names and titles.
- "kind": "state" when the attribute has one current value that a later statement would replace (job title, where they live, a count or total so far, a personal best, a schedule); "event" when it is something that happened or one item among many (a purchase, a trip, an activity, a person met, a book read); "preference" for a lasting taste, habit, opinion, interest, goal or constraint, or a tool, brand or product they use, that should shape future advice or recommendations (likes Sony cameras, prefers aisle seats, is vegetarian, uses a Suica card in Tokyo, enjoys nature documentaries, avoids caffeine after 2pm, is training for a marathon).
- "when": the time phrase from the conversation that dates the fact, copied as written ("last Tuesday", "two weeks ago", "in March", "on May 3rd"), or null if none.
- "date": the calendar date that "when" refers to, as YYYY-MM-DD (or YYYY-MM), computed from the session date; null if no time was stated.
- "turns": the numbers of the [t#] turns that state the fact.
- "confidence": from 0.0 to 1.0.
- "statement": one self-contained sentence with absolute dates, for example "On 2023-05-02 the user bought a 70-200mm zoom lens."

Rules:
- Report what the speakers say about themselves, their lives, plans, possessions, people, numbers and experiences. Include changes ("I now ...", "I switched to ...") and totals ("that's my fifth ...").
- Also report preferences the user shows indirectly: what they ask help with, what they already own or use, what they enjoyed or disliked, and what they plan to do. Name the specifics (brands, titles, places, ingredients) so a later recommendation can build on them.
- Skip greetings, general knowledge, advice and hypotheticals. Skip what the assistant says unless the user confirms it as true for them.
- Prefer several small facts over one long one. At most 40 facts.
- If the session states nothing about the people, reply {"facts": []}."#;

/// One turn as sent to the extractor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SessionTurn {
    /// Episode id, used as evidence.
    pub id: String,
    pub speaker: String,
    pub assistant: bool,
    pub text: String,
}

/// A session to extract from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SessionInput {
    pub id: String,
    pub date: Option<Time>,
    pub turns: Vec<SessionTurn>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactKind {
    /// One current value per (subject, attribute); a newer one replaces it.
    State,
    /// One occurrence among many; values accumulate.
    Event,
    /// A lasting taste, habit, opinion, interest, goal or constraint; values
    /// accumulate and the profile in Memory shows them.
    Preference,
}

/// A fact as extracted, with its date resolved and turns mapped to ids.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SessionFact {
    pub subject: String,
    pub attribute: String,
    pub value: String,
    pub kind: FactKind,
    pub valid_from: Option<Time>,
    pub basis: TimeBasis,
    /// The time phrase the model copied, if any.
    pub when: Option<String>,
    /// Episode ids of the turns that state it; never empty.
    pub turns: Vec<String>,
    pub confidence: f32,
    pub statement: String,
}

/// Normalise an attribute name to snake_case ASCII-lowercase words, so
/// "Job Title" and "job_title" key the same property.
pub fn normalize_attribute(s: &str) -> String {
    let mut out = String::new();
    for w in s
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
    {
        let w = w.to_lowercase();
        if out.is_empty() && matches!(w.as_str(), "current" | "user" | "my" | "the") {
            continue;
        }
        if !out.is_empty() {
            out.push('_');
        }
        out.push_str(&w);
    }
    out
}

fn date_time(d: NaiveDate) -> Option<Time> {
    Some(Utc.from_utc_datetime(&d.and_hms_opt(0, 0, 0)?))
}

const MONTHS: [&str; 12] = [
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

pub(crate) fn month_of(w: &str) -> Option<u32> {
    let w = w.trim_end_matches('.');
    if w.len() < 3 {
        return None;
    }
    MONTHS
        .iter()
        .position(|m| m.starts_with(w))
        .map(|i| i as u32 + 1)
}

pub(crate) fn weekday_of(w: &str) -> Option<Weekday> {
    Some(match w {
        "monday" | "mon" => Weekday::Mon,
        "tuesday" | "tue" | "tues" => Weekday::Tue,
        "wednesday" | "wed" => Weekday::Wed,
        "thursday" | "thu" | "thurs" => Weekday::Thu,
        "friday" | "fri" => Weekday::Fri,
        "saturday" | "sat" => Weekday::Sat,
        "sunday" | "sun" => Weekday::Sun,
        _ => return None,
    })
}

pub(crate) fn number_of(w: &str) -> Option<i64> {
    if let Ok(n) = w.parse::<i64>() {
        return Some(n);
    }
    Some(match w {
        "a" | "an" | "one" => 1,
        "two" | "couple" => 2,
        "three" | "few" => 3,
        "four" => 4,
        "five" => 5,
        "six" => 6,
        "seven" => 7,
        "eight" => 8,
        "nine" => 9,
        "ten" => 10,
        "eleven" => 11,
        "twelve" => 12,
        _ => return None,
    })
}

pub(crate) fn day_of(w: &str) -> Option<u32> {
    let d = w
        .trim_end_matches(|c: char| c.is_alphabetic())
        .parse::<u32>()
        .ok()?;
    (1..=31).contains(&d).then_some(d)
}

pub(crate) fn year_of(w: &str) -> Option<i32> {
    w.parse::<i32>().ok().filter(|y| (1900..=2200).contains(y))
}

/// The most recent `weekday` strictly before `anchor`.
pub(crate) fn previous(anchor: NaiveDate, weekday: Weekday) -> NaiveDate {
    let mut d = anchor - Duration::days(1);
    while d.weekday() != weekday {
        d -= Duration::days(1);
    }
    d
}

/// A month and optional day mentioned without a year: the latest such date
/// not after `anchor` (people mostly talk about the past).
fn month_day(
    anchor: NaiveDate,
    month: u32,
    day: Option<u32>,
    year: Option<i32>,
) -> Option<NaiveDate> {
    let day = day.unwrap_or(1);
    match year {
        Some(y) => NaiveDate::from_ymd_opt(y, month, day),
        None => {
            let this = NaiveDate::from_ymd_opt(anchor.year(), month, day)?;
            if this <= anchor {
                Some(this)
            } else {
                NaiveDate::from_ymd_opt(anchor.year() - 1, month, day)
            }
        }
    }
}

/// Resolve a time phrase against the session date. Returns `None` for
/// anything it does not understand; it never guesses a date from a
/// phrase it cannot parse.
///
/// Understood: ISO dates (`2023-05-02`, `2023-05`), `today`, `yesterday`,
/// `the day before yesterday`, `N days/weeks/months/years ago` (digits or
/// words, `a couple of`, `a few`), `last week/month/year/weekend`,
/// `last <weekday>`, `on <weekday>` (the most recent one), `this morning`,
/// `<month> <day>[, <year>]`, `<day> <month> [<year>]`, `[in] <month>
/// [<year>]`, and `in <year>`.
pub fn resolve_when(phrase: &str, anchor: NaiveDate) -> Option<NaiveDate> {
    let lower = phrase.trim().to_lowercase();
    if let Ok(d) = NaiveDate::parse_from_str(&lower, "%Y-%m-%d") {
        return Some(d);
    }
    if let Ok(d) = NaiveDate::parse_from_str(&format!("{lower}-01"), "%Y-%m-%d") {
        return Some(d);
    }
    let words: Vec<String> = lower
        .split(|c: char| c.is_whitespace() || c == ',' || c == '/')
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| {
            !w.is_empty()
                && !matches!(
                    w.as_str(),
                    "the" | "of" | "on" | "in" | "about" | "around" | "just" | "back"
                )
        })
        .collect();
    let w: Vec<&str> = words.iter().map(String::as_str).collect();
    match w.as_slice() {
        [] => return None,
        ["today"]
        | ["tonight"]
        | ["this", "morning" | "afternoon" | "evening"]
        | ["earlier", "today"] => {
            return Some(anchor);
        }
        ["yesterday", ..] | ["last", "night"] => return Some(anchor - Duration::days(1)),
        ["day", "before", "yesterday"] => return Some(anchor - Duration::days(2)),
        ["last", "week"] => return Some(anchor - Duration::days(7)),
        ["last", "weekend"] => return Some(previous(anchor, Weekday::Sat)),
        ["last", "month"] => return anchor.checked_sub_months(Months::new(1)),
        ["last", "year"] => return anchor.checked_sub_months(Months::new(12)),
        ["last", day] | ["this", "past", day] | ["past", day] => {
            if let Some(wd) = weekday_of(day) {
                return Some(previous(anchor, wd));
            }
        }
        [day] if weekday_of(day).is_some() => {
            return Some(previous(anchor, weekday_of(day)?));
        }
        _ => {}
    }
    // N units ago.
    if let Some(at) = w.iter().position(|x| *x == "ago") {
        let unit = *w.get(at.checked_sub(1)?)?;
        let n = match at {
            1 => 1,
            _ => number_of(w[at - 2]).or_else(|| (w[at - 2] == "few").then_some(3))?,
        };
        let unit = unit.trim_end_matches('s');
        return match unit {
            "day" => Some(anchor - Duration::days(n)),
            "week" => Some(anchor - Duration::days(7 * n)),
            "month" => anchor.checked_sub_months(Months::new(n as u32)),
            "year" => anchor.checked_sub_months(Months::new(12 * n as u32)),
            _ => None,
        };
    }
    // Month names, with or without a day and year.
    let month_at = w.iter().position(|x| month_of(x).is_some());
    if let Some(i) = month_at {
        let month = month_of(w[i])?;
        let day = w
            .get(i + 1)
            .and_then(|x| day_of(x))
            .or_else(|| i.checked_sub(1).and_then(|j| day_of(w[j])));
        let year = w.iter().find_map(|x| year_of(x));
        return month_day(anchor, month, day, year);
    }
    if let [y] = w.as_slice()
        && let Some(y) = year_of(y)
    {
        return NaiveDate::from_ymd_opt(y, 1, 1);
    }
    None
}

/// The exact user message sent for `session`; its hash keys the cache.
pub fn render_session(session: &SessionInput) -> String {
    let date = session.date.map_or_else(
        || "unknown".to_string(),
        |d| format!("{} ({})", d.date_naive(), d.date_naive().weekday()),
    );
    let mut out = format!("Session date: {date}\n");
    for (i, t) in session.turns.iter().enumerate() {
        let limit = if t.assistant {
            MAX_ASSISTANT_CHARS
        } else {
            MAX_TURN_CHARS
        };
        let text = t.text.trim();
        let cut: String = text.chars().take(limit).collect();
        let more = if cut.len() < text.len() { " [...]" } else { "" };
        out.push_str(&format!("[t{}] {}: {}{}\n", i + 1, t.speaker, cut, more));
    }
    out
}

/// Hash of the session text as sent.
pub fn session_hash(session: &SessionInput) -> String {
    hex::encode(Sha256::digest(render_session(session).as_bytes()))
}

/// Cache key: (model, prompt version, session content hash).
pub fn cache_key(model: &str, prompt_version: &str, content_hash: &str) -> String {
    let mut h = Sha256::new();
    for part in [model, prompt_version, content_hash] {
        h.update(part.as_bytes());
        h.update([0u8]);
    }
    hex::encode(h.finalize())
}

/// One cached model reply, stored raw so parsing fixes never need a call.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CachedReply {
    pub model: String,
    pub prompt_version: String,
    pub content_hash: String,
    pub text: String,
    pub usage: Usage,
    #[serde(default)]
    pub ai_credits: Option<f64>,
}

/// A directory of cached replies, one JSON file per key.
#[derive(Clone, Debug)]
pub struct ExtractionCache {
    dir: PathBuf,
}

impl ExtractionCache {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn path(&self, key: &str) -> PathBuf {
        self.dir.join(&key[..2]).join(format!("{key}.json"))
    }

    pub fn get(&self, key: &str) -> Option<CachedReply> {
        let body = std::fs::read(self.path(key)).ok()?;
        serde_json::from_slice(&body).ok()
    }

    /// Write atomically: a crash never leaves a half-written reply.
    pub fn put(&self, key: &str, reply: &CachedReply) -> std::io::Result<()> {
        static NEXT_WRITE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = self.path(key);
        let dir = path.parent().expect("cache path has a parent");
        std::fs::create_dir_all(dir)?;
        let nonce = NEXT_WRITE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp = dir.join(format!(".{key}.{}.{nonce}.tmp", std::process::id()));
        std::fs::write(&tmp, serde_json::to_vec(reply).expect("reply serialises"))?;
        std::fs::rename(tmp, path)
    }
}

/// What one session produced.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionExtraction {
    pub facts: Vec<SessionFact>,
    pub cached: bool,
    pub usage: Usage,
    /// AI credits the provider reported for a fresh call.
    pub ai_credits: Option<f64>,
    /// Facts dropped because they cited no known turn or were empty.
    pub dropped: usize,
}

/// The chat request for `session`.
pub fn session_request(session: &SessionInput, max_output_tokens: u32) -> ChatRequest {
    let user = render_session(session);
    let estimated = (SESSION_PROMPT.len() + user.len()).div_ceil(4) as u64;
    ChatRequest {
        messages: vec![Message::system(SESSION_PROMPT), Message::user(user)],
        max_output_tokens,
        temperature: 0.0,
        seed: Some(0),
        estimated_input_tokens: estimated,
    }
}

/// Parse a reply into facts. Facts that cite no turn of the session, or
/// have an empty subject, attribute or value, are dropped and counted.
pub fn parse_reply(
    session: &SessionInput,
    reply: &str,
) -> Result<(Vec<SessionFact>, usize), ProviderError> {
    #[derive(Deserialize)]
    struct Out {
        #[serde(default)]
        facts: Vec<serde_json::Value>,
    }
    #[derive(Deserialize)]
    struct Raw {
        #[serde(default)]
        subject: String,
        #[serde(default)]
        attribute: String,
        #[serde(default)]
        value: serde_json::Value,
        #[serde(default)]
        kind: Option<String>,
        #[serde(default)]
        when: Option<String>,
        #[serde(default)]
        date: Option<String>,
        #[serde(default)]
        turns: Vec<serde_json::Value>,
        #[serde(default)]
        confidence: Option<f32>,
        #[serde(default)]
        statement: Option<String>,
    }
    let body = reply.trim();
    let body = match (body.find('{'), body.rfind('}')) {
        (Some(a), Some(b)) if a < b => &body[a..=b],
        _ => body,
    };
    let out: Out = serde_json::from_str(body).map_err(|e| ProviderError::Malformed {
        message: format!("session extraction reply is not the expected JSON: {e}"),
    })?;
    let anchor = session.date.map(|d| d.date_naive());
    let mut facts = Vec::new();
    let mut dropped = 0;
    for v in out.facts.into_iter() {
        let Ok(r) = serde_json::from_value::<Raw>(v) else {
            dropped += 1;
            continue;
        };
        let value = match r.value {
            serde_json::Value::String(s) => s,
            serde_json::Value::Null => String::new(),
            other => other.to_string(),
        };
        let subject = r.subject.trim().to_string();
        let attribute = normalize_attribute(&r.attribute);
        let value = value.trim().to_string();
        let mut turns: Vec<String> = Vec::new();
        for t in &r.turns {
            let n = match t {
                serde_json::Value::Number(n) => n.as_u64(),
                serde_json::Value::String(s) => {
                    s.trim().trim_start_matches(['t', 'T']).parse().ok()
                }
                _ => None,
            };
            if let Some(turn) = n
                .and_then(|n| (n as usize).checked_sub(1))
                .and_then(|i| session.turns.get(i))
                && !turns.contains(&turn.id)
            {
                turns.push(turn.id.clone());
            }
        }
        if subject.is_empty() || attribute.is_empty() || value.is_empty() || turns.is_empty() {
            dropped += 1;
            continue;
        }
        let when = r
            .when
            .filter(|w| !w.trim().is_empty() && w.trim() != "null");
        let resolved = anchor.and_then(|a| when.as_deref().and_then(|w| resolve_when(w, a)));
        let model_date = r.date.as_deref().and_then(|d| {
            let d = d.trim();
            NaiveDate::parse_from_str(d, "%Y-%m-%d")
                .or_else(|_| NaiveDate::parse_from_str(&format!("{d}-01"), "%Y-%m-%d"))
                .ok()
        });
        let (valid_from, basis) = match resolved.or(when.as_ref().and(model_date)) {
            Some(d) => (date_time(d), TimeBasis::Explicit),
            None => match session.date {
                Some(d) => (Some(d), TimeBasis::Observed),
                None => (None, TimeBasis::Unknown),
            },
        };
        let kind = match r.kind.as_deref().map(str::trim) {
            Some("state") => FactKind::State,
            Some("preference") => FactKind::Preference,
            _ => FactKind::Event,
        };
        let statement = r
            .statement
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| format!("{subject} {}: {value}", attribute.replace('_', " ")));
        facts.push(SessionFact {
            subject,
            attribute,
            value,
            kind,
            valid_from,
            basis,
            when,
            turns,
            confidence: r.confidence.unwrap_or(0.5).clamp(0.0, 1.0),
            statement,
        });
        if facts.len() == MAX_FACTS {
            break;
        }
    }
    Ok((facts, dropped))
}

/// Extracts facts from whole sessions through a chat model, with an
/// optional on-disk reply cache.
pub struct SessionExtractor {
    model: Option<Arc<dyn ChatModel>>,
    model_id: String,
    cache: Option<ExtractionCache>,
    max_output_tokens: u32,
}

impl SessionExtractor {
    /// `model_id` keys the cache; it must name the model and its revision.
    /// Without a model, only cached replies are served.
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
        }
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Whether cache misses can be filled by calling the model.
    pub fn has_model(&self) -> bool {
        self.model.is_some()
    }

    /// Facts for `session` from the cache alone; `None` on a miss.
    pub fn cached(
        &self,
        session: &SessionInput,
    ) -> Option<Result<SessionExtraction, ProviderError>> {
        let hit = self.cache.as_ref()?.get(&self.key(session))?;
        Some(
            parse_reply(session, &hit.text).map(|(facts, dropped)| SessionExtraction {
                facts,
                cached: true,
                usage: Usage::default(),
                ai_credits: None,
                dropped,
            }),
        )
    }

    /// Evidence label recorded on every fact this extractor produces.
    pub fn evidence_label(&self) -> String {
        format!("session:{}:{SESSION_PROMPT_VERSION}", self.model_id)
    }

    pub fn key(&self, session: &SessionInput) -> String {
        cache_key(
            &self.model_id,
            SESSION_PROMPT_VERSION,
            &session_hash(session),
        )
    }

    /// Whether a reply for `session` is already cached.
    pub fn is_cached(&self, session: &SessionInput) -> bool {
        self.cache
            .as_ref()
            .is_some_and(|c| c.get(&self.key(session)).is_some())
    }

    pub fn request(&self, session: &SessionInput) -> ChatRequest {
        session_request(session, self.max_output_tokens)
    }

    /// Facts for `session`: from the cache when present, else from the
    /// model (and then cached). Without a model a cache miss is an error.
    pub async fn extract(
        &self,
        session: &SessionInput,
    ) -> Result<SessionExtraction, ProviderError> {
        let content_hash = session_hash(session);
        let key = cache_key(&self.model_id, SESSION_PROMPT_VERSION, &content_hash);
        if let Some(hit) = self.cached(session) {
            return hit;
        }
        let Some(model) = &self.model else {
            return Err(ProviderError::Config {
                message: format!(
                    "no cached extraction for session {} and no extraction model to call",
                    session.id
                ),
            });
        };
        let response = model.complete(&self.request(session)).await?;
        let (facts, dropped) = parse_reply(session, &response.text)?;
        if let Some(cache) = &self.cache {
            cache
                .put(
                    &key,
                    &CachedReply {
                        model: self.model_id.clone(),
                        prompt_version: SESSION_PROMPT_VERSION.into(),
                        content_hash,
                        text: response.text.clone(),
                        usage: response.usage,
                        ai_credits: response.billing.as_ref().map(|b| b.ai_credits),
                    },
                )
                .map_err(|e| ProviderError::Transport {
                    message: format!("could not write the extraction cache: {e}"),
                })?;
        }
        Ok(SessionExtraction {
            facts,
            cached: false,
            usage: response.usage,
            ai_credits: response.billing.as_ref().map(|b| b.ai_credits),
            dropped,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concurrent_cache_writers_never_share_a_staging_file() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/test-extract-cache")
            .join(format!("concurrent-{}", std::process::id()));
        let cache = ExtractionCache::new(&dir);
        let reply = CachedReply {
            model: "mock".into(),
            prompt_version: SESSION_PROMPT_VERSION.into(),
            content_hash: "synthetic".into(),
            text: "{\"facts\":[]}".into(),
            usage: Usage::default(),
            ai_credits: None,
        };
        let gate = std::sync::Barrier::new(8);
        let successes = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        gate.wait();
                        (0..16)
                            .filter(|_| cache.put("shared", &reply).is_ok())
                            .count()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .sum::<usize>()
        });
        let intact = cache.get("shared") == Some(reply);
        std::fs::remove_dir_all(dir).unwrap();
        assert_eq!(successes, 128);
        assert!(intact);
    }

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    fn resolves_relative_phrases_from_the_session_date() {
        // 2023-05-20 is a Saturday.
        let a = d(2023, 5, 20);
        let cases = [
            ("2023-04-02", d(2023, 4, 2)),
            ("2023-04", d(2023, 4, 1)),
            ("today", a),
            ("yesterday", d(2023, 5, 19)),
            ("the day before yesterday", d(2023, 5, 18)),
            ("last Tuesday", d(2023, 5, 16)),
            ("on Saturday", d(2023, 5, 13)),
            ("last week", d(2023, 5, 13)),
            ("last weekend", d(2023, 5, 13)),
            ("two weeks ago", d(2023, 5, 6)),
            ("3 days ago", d(2023, 5, 17)),
            ("a couple of days ago", d(2023, 5, 18)),
            ("a week ago", d(2023, 5, 13)),
            ("a few days ago", d(2023, 5, 17)),
            ("a month ago", d(2023, 4, 20)),
            ("last month", d(2023, 4, 20)),
            ("May 3rd", d(2023, 5, 3)),
            ("3 May 2022", d(2022, 5, 3)),
            ("on March 14, 2021", d(2021, 3, 14)),
            ("in March", d(2023, 3, 1)),
            ("December", d(2022, 12, 1)),
            ("in 2019", d(2019, 1, 1)),
        ];
        for (phrase, want) in cases {
            assert_eq!(resolve_when(phrase, a), Some(want), "{phrase}");
        }
        for phrase in ["recently", "soon", "a while back", "next Tuesday"] {
            assert_eq!(resolve_when(phrase, a), None, "{phrase}");
        }
    }

    fn session() -> SessionInput {
        SessionInput {
            id: "s1".into(),
            date: date_time(d(2023, 5, 20)),
            turns: vec![
                SessionTurn {
                    id: "s1:1".into(),
                    speaker: "user".into(),
                    assistant: false,
                    text: "I got a 70-200mm lens last Tuesday. I'm a senior engineer now.".into(),
                },
                SessionTurn {
                    id: "s1:2".into(),
                    speaker: "assistant".into(),
                    assistant: true,
                    text: "x".repeat(2_000),
                },
            ],
        }
    }

    #[test]
    fn parses_replies_and_resolves_dates_deterministically() {
        let reply = r#"```json
{"facts": [
 {"subject": "user", "attribute": "Bought Item", "value": "70-200mm lens", "kind": "event",
  "when": "last Tuesday", "date": "2023-05-15", "turns": [1], "confidence": 0.9,
  "statement": "On 2023-05-16 the user bought a 70-200mm lens."},
 {"subject": "user", "attribute": "current job title", "value": "senior engineer", "kind": "state",
  "when": null, "date": null, "turns": ["t1"], "confidence": 0.8, "statement": ""},
 {"subject": "user", "attribute": "camera brand", "value": "Sony", "kind": "preference", "turns": [1]},
 {"subject": "user", "attribute": "pet", "value": "cat", "kind": "state", "turns": [7]},
 {"subject": "", "attribute": "x", "value": "y", "turns": [1]}
]}
```"#;
        let (facts, dropped) = parse_reply(&session(), reply).unwrap();
        assert_eq!(dropped, 2);
        assert_eq!(facts.len(), 3);
        assert_eq!(facts[2].kind, FactKind::Preference);
        // The resolver wins over the model's own (wrong) date.
        assert_eq!(facts[0].attribute, "bought_item");
        assert_eq!(facts[0].valid_from, date_time(d(2023, 5, 16)));
        assert_eq!(facts[0].basis, TimeBasis::Explicit);
        assert_eq!(facts[0].kind, FactKind::Event);
        assert_eq!(facts[0].turns, vec!["s1:1".to_string()]);
        // No time phrase: true as of the session date.
        assert_eq!(facts[1].attribute, "job_title");
        assert_eq!(facts[1].kind, FactKind::State);
        assert_eq!(facts[1].valid_from, date_time(d(2023, 5, 20)));
        assert_eq!(facts[1].basis, TimeBasis::Observed);
        assert_eq!(facts[1].statement, "user job title: senior engineer");
    }

    #[test]
    fn renders_numbered_turns_and_cuts_assistant_turns() {
        let text = render_session(&session());
        assert!(text.starts_with("Session date: 2023-05-20 (Sat)\n[t1] user: I got"));
        let t2 = text.lines().nth(2).unwrap();
        assert!(t2.starts_with("[t2] assistant: xxx"));
        assert!(t2.ends_with(" [...]"));
        assert!(t2.len() < MAX_ASSISTANT_CHARS + 40);
    }

    #[test]
    fn cache_keys_change_with_model_prompt_or_content() {
        let s = session();
        let h = session_hash(&s);
        let k = cache_key("m@1", SESSION_PROMPT_VERSION, &h);
        assert_eq!(k, cache_key("m@1", SESSION_PROMPT_VERSION, &h));
        assert_ne!(k, cache_key("m@2", SESSION_PROMPT_VERSION, &h));
        assert_ne!(k, cache_key("m@1", "session-facts-v0", &h));
        let mut edited = s.clone();
        edited.turns[0].text.push('!');
        assert_ne!(h, session_hash(&edited));
        let mut moved = s;
        moved.date = date_time(d(2023, 5, 21));
        assert_ne!(h, session_hash(&moved));
    }

    #[tokio::test]
    async fn serves_cached_replies_without_calling_the_model() {
        use keep_providers::mock::FnChat;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/test-extract-cache")
            .join(std::process::id().to_string());
        let _ = std::fs::remove_dir_all(&dir);
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = calls.clone();
        let model = FnChat::new("extract", move |_| {
            seen.fetch_add(1, Ordering::SeqCst);
            Ok(r#"{"facts":[{"subject":"user","attribute":"pet","value":"cat","kind":"state","turns":[1]}]}"#.into())
        });
        let x = SessionExtractor::new(
            Some(Arc::new(model)),
            "mock/extract@1",
            Some(ExtractionCache::new(&dir)),
            1_000,
        );
        let s = session();
        assert!(!x.is_cached(&s));
        let first = x.extract(&s).await.unwrap();
        assert!(!first.cached);
        let second = x.extract(&s).await.unwrap();
        assert!(second.cached);
        assert_eq!(first.facts, second.facts);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        // A cache-only extractor serves the hit and refuses the miss.
        let offline = SessionExtractor::new(
            None,
            "mock/extract@1",
            Some(ExtractionCache::new(&dir)),
            1_000,
        );
        assert!(offline.extract(&s).await.unwrap().cached);
        let mut other = s.clone();
        other.id = "s2".into();
        other.turns[0].text = "Something else".into();
        assert!(offline.extract(&other).await.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
