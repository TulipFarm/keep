//! What kind of memory a question asks for.
//!
//! [`asks_about_assistant_replies`] spots questions about something the
//! assistant itself said earlier ("remind me what you suggested", "your
//! previous answer", "our last conversation"). Facts extracted from the
//! user's side of a conversation cannot answer those; the verbatim earlier
//! reply can. Matching is deterministic and word based.

const SAID: &[&str] = &[
    "said",
    "told",
    "mentioned",
    "suggested",
    "recommended",
    "gave",
    "provided",
    "listed",
    "shared",
    "wrote",
    "explained",
    "described",
    "made",
    "created",
    "proposed",
    "came",
];

const REPLY: &[&str] = &[
    "response",
    "answer",
    "suggestion",
    "suggestions",
    "recommendation",
    "recommendations",
    "list",
    "explanation",
    "reply",
    "advice",
];

const EARLIER: &[&str] = &["previous", "earlier", "last"];

fn words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '\''))
        .filter(|w| !w.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Whether the question asks about one of the assistant's own earlier
/// replies rather than about the user.
pub fn asks_about_assistant_replies(question: &str) -> bool {
    let w = words(question);
    let at = |i: usize| w.get(i).map_or("", String::as_str);
    for i in 0..w.len() {
        let (next, after) = (at(i + 1), at(i + 2));
        let hit = match at(i) {
            "you" => SAID.contains(&next),
            "your" => REPLY.contains(&next) || (EARLIER.contains(&next) && REPLY.contains(&after)),
            "our" => {
                EARLIER.contains(&next) && matches!(after, "conversation" | "chat" | "discussion")
            }
            "we" => matches!(
                next,
                "talk" | "talked" | "discuss" | "discussed" | "chat" | "chatted"
            ),
            // "Remind me what my ... was" is about the user.
            "remind" if next == "me" => !w[i + 2..].iter().any(|x| x == "i" || x == "my"),
            _ => false,
        };
        if hit {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::asks_about_assistant_replies as asks;

    #[test]
    fn spots_questions_about_earlier_replies() {
        assert!(asks(
            "Can you remind me what color the dragon was in the story?"
        ));
        assert!(asks("What was the third tip you suggested for sleep?"));
        assert!(asks("In your previous answer, which tool came first?"));
        assert!(asks(
            "Going back to our previous conversation, what was the hotel?"
        ));
        assert!(asks("Which café did we talk about near the harbour?"));
    }

    #[test]
    fn leaves_questions_about_the_user_alone() {
        assert!(!asks("How many bikes do I own?"));
        assert!(!asks("Can you remind me when my dentist visit was?"));
        assert!(!asks("What did I say about your app last week?"));
        assert!(!asks("Where did you think I should go?"));
    }
}
