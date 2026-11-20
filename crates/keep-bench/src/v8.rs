//! Version 8: rank conversation excerpts by what a question asks.
//!
//! In a long two-person history nearly every turn by one speaker names the
//! other ("Wow, Caroline! ..."). Searching with the question as written
//! ("What did Caroline research?") then fills the excerpts with short
//! replies that match only the name. The `focus` option searches the
//! excerpts with the participant names taken out, and adds the asked
//! participant's own turns as one more fused list, so what they said
//! rises without hiding what others said about them.
//!
//! The companion `verbatim_first` option (in `knowledge_arm`) leaves out
//! facts whose source turns the excerpts already show word for word, so
//! the excerpts get that room.

use std::collections::BTreeSet;

/// Words too common to search on by themselves.
const FILLER: &[&str] = &[
    "a", "an", "and", "are", "at", "be", "did", "do", "does", "for", "from", "had", "has", "have",
    "he", "her", "his", "how", "in", "is", "it", "of", "on", "or", "she", "that", "the", "their",
    "them", "they", "to", "was", "were", "what", "when", "where", "which", "who", "whom", "whose",
    "why", "with",
];

/// The words that name `people`: every word of each name, lowercased.
fn name_words(people: &[String]) -> BTreeSet<String> {
    people
        .iter()
        .flat_map(|p| p.split(|c: char| !c.is_alphanumeric()))
        .filter(|w| w.chars().count() > 1)
        .map(str::to_lowercase)
        .collect()
}

/// A token's word without surrounding punctuation or a possessive ending.
fn core(token: &str) -> String {
    let t = token.trim_matches(|c: char| !c.is_alphanumeric());
    let t = t
        .strip_suffix("'s")
        .or_else(|| t.strip_suffix("\u{2019}s"))
        .unwrap_or(t);
    t.to_lowercase()
}

/// `question` without the participant names in it, for ranking excerpts.
/// `None` when it names no participant, or when too little would be left
/// to search on, so the caller keeps the question as written.
pub fn focus_query(question: &str, people: &[String]) -> Option<String> {
    let names = name_words(people);
    if names.is_empty() {
        return None;
    }
    let mut dropped = false;
    let kept: Vec<&str> = question
        .split_whitespace()
        .filter(|t| {
            let named = names.contains(&core(t));
            dropped |= named;
            !named
        })
        .collect();
    let content = kept
        .iter()
        .map(|t| core(t))
        .filter(|w| !w.is_empty() && !FILLER.contains(&w.as_str()))
        .count();
    (dropped && content > 0).then(|| kept.join(" "))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn people() -> Vec<String> {
        vec!["Caroline".into(), "Melanie".into()]
    }

    #[test]
    fn names_and_possessives_are_taken_out() {
        assert_eq!(
            focus_query("What is Caroline's relationship status?", &people()).as_deref(),
            Some("What is relationship status?")
        );
        assert_eq!(
            focus_query("When did Melanie and Caroline go camping?", &people()).as_deref(),
            Some("When did and go camping?")
        );
        assert_eq!(
            focus_query("What did caroline\u{2019}s friends give her?", &people()).as_deref(),
            Some("What did friends give her?")
        );
    }

    #[test]
    fn questions_without_names_or_content_are_kept() {
        // No participant named: LongMemEval's user and assistant have no names.
        assert_eq!(focus_query("What did I buy last week?", &people()), None);
        assert_eq!(focus_query("What did I buy last week?", &[]), None);
        // Only the name would be left to search on.
        assert_eq!(focus_query("Who is Caroline?", &people()), None);
        // Names inside other words stay.
        assert_eq!(focus_query("What did Carolinesque mean?", &people()), None);
    }
}
