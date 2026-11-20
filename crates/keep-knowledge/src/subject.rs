//! Who a question is about.
//!
//! In a conversation between several people, a question about one of them
//! ("What is Ada's favourite film?") must not be answered with something
//! another person said about themselves. [`Grounding::of`] finds the named
//! participants a question asks about, and [`Grounding::about`] tells
//! whether a fact's subject is one of them, another participant, or
//! neither, so a context can keep the two apart. Matching is deterministic
//! and word based; a question naming no participant, or every participant,
//! has no grounding.

/// The participants a question asks about, and the others.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Grounding {
    pub asked: Vec<String>,
    pub others: Vec<String>,
}

/// Whose a fact is, relative to a [`Grounding`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum About {
    /// Names one of the people asked about.
    Asked,
    /// Names another participant and none of the people asked about.
    Other,
    /// Names no participant.
    Unknown,
}

fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// The word a participant is known by: the first word of the name.
fn key(name: &str) -> Option<String> {
    words(name).into_iter().next().filter(|w| w.len() > 1)
}

fn names(text: &str, person: &str) -> bool {
    key(person).is_some_and(|k| words(text).contains(&k))
}

impl Grounding {
    /// The grounding of `question` among `participants` (speaker names).
    /// `None` when it names none of them, or all of them.
    pub fn of(question: &str, participants: &[String]) -> Option<Self> {
        let mut asked = Vec::new();
        let mut others = Vec::new();
        for p in participants {
            if key(p).is_none() || asked.contains(p) || others.contains(p) {
                continue;
            }
            if names(question, p) {
                asked.push(p.clone());
            } else {
                others.push(p.clone());
            }
        }
        (!asked.is_empty() && !others.is_empty()).then_some(Self { asked, others })
    }

    /// Whose `subject` is. A subject naming an asked person counts as
    /// theirs even if it also names someone else ("Ada's brother Ben").
    pub fn about(&self, subject: &str) -> About {
        if self.asked.iter().any(|p| names(subject, p)) {
            About::Asked
        } else if self.others.iter().any(|p| names(subject, p)) {
            About::Other
        } else {
            About::Unknown
        }
    }

    /// The people asked about, joined for display ("Ada and Ben").
    pub fn asked_label(&self) -> String {
        join(&self.asked)
    }

    pub fn others_label(&self) -> String {
        join(&self.others)
    }
}

fn join(names: &[String]) -> String {
    match names {
        [] => String::new(),
        [one] => one.clone(),
        [init @ .., last] => format!("{} and {last}", init.join(", ")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn people() -> Vec<String> {
        vec!["Ilse".into(), "Bram".into()]
    }

    #[test]
    fn finds_the_person_asked_about() {
        let g = Grounding::of("What is Ilse's favourite trilogy?", &people()).unwrap();
        assert_eq!(g.asked, vec!["Ilse".to_string()]);
        assert_eq!(g.others, vec!["Bram".to_string()]);
        assert_eq!(g.asked_label(), "Ilse");
        // Case and punctuation do not matter; substrings do not match.
        let g = Grounding::of("did BRAM, or not?", &people()).unwrap();
        assert_eq!(g.asked, vec!["Bram".to_string()]);
        assert!(Grounding::of("Who bought the Bramley apples?", &people()).is_none());
    }

    #[test]
    fn no_grounding_without_exactly_a_subset_named() {
        assert!(Grounding::of("When did I move?", &people()).is_none());
        assert!(Grounding::of("How did Ilse and Bram meet?", &people()).is_none());
        assert!(
            Grounding::of(
                "What did the user buy?",
                &["user".into(), "assistant".into()]
            )
            .is_some_and(|g| g.asked == vec!["user".to_string()])
        );
    }

    #[test]
    fn classifies_fact_subjects() {
        let g = Grounding::of("What pets does Ilse have?", &people()).unwrap();
        assert_eq!(g.about("Ilse"), About::Asked);
        assert_eq!(g.about("Ilse's mother"), About::Asked);
        assert_eq!(g.about("Bram"), About::Other);
        assert_eq!(g.about("Bram's sister and Ilse"), About::Asked);
        assert_eq!(g.about("the neighbour"), About::Unknown);
    }
}
