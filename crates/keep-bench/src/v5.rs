//! Opt-in retrieval helpers; no dataset labels or scorer data are inputs.

use std::collections::{BTreeSet, HashMap};

use crate::dataset::History;
use crate::facts::{FactGroup, FactLine};

pub fn other_fact_ids(
    lines: &[FactLine],
    grounding: &keep_knowledge::subject::Grounding,
) -> BTreeSet<String> {
    use keep_knowledge::subject::About;
    lines
        .iter()
        .filter(|line| {
            grounding.about(&line.subject) == About::Other
                && grounding.about(&line.text) != About::Asked
        })
        .map(|line| line.id.clone())
        .collect()
}

/// Bound the extra searches. Each facet retains the original named subject.
pub fn facets(question: &str, people: &[String]) -> Vec<String> {
    let mut out = vec![question.to_owned()];
    // Grounding deliberately returns None when all people are named, so
    // use whole-token names for facets rather than its contrast rule.
    let words: BTreeSet<_> = question
        .split(|c: char| !c.is_alphanumeric())
        .map(str::to_lowercase)
        .collect();
    let subjects: Vec<_> = people
        .iter()
        .filter(|p| {
            p.split_whitespace()
                .all(|w| words.contains(&w.to_lowercase()))
        })
        .cloned()
        .collect();
    for separator in [" and ", " versus ", " compared with ", "; "] {
        if let Some((a, b)) = question.split_once(separator) {
            for part in [a, b] {
                if part.split_whitespace().count() >= 2 {
                    let facet = format!("{} {}", subjects.join(" "), part.trim());
                    if !out.contains(&facet) {
                        out.push(facet);
                    }
                }
            }
            break;
        }
    }
    if subjects.len() > 1 {
        for person in &subjects {
            let mut facet = question.to_owned();
            for other in subjects.iter().filter(|p| *p != person) {
                // Replace whole tokens only, preserving names embedded in other words.
                facet = facet
                    .split_whitespace()
                    .filter(|w| {
                        !w.trim_matches(|c: char| !c.is_alphanumeric())
                            .eq_ignore_ascii_case(other)
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
            }
            if !out.contains(&facet) {
                out.push(facet);
            }
        }
    }
    out.truncate(3);
    out
}

/// Built once per history at fact ingestion, retaining the source date
/// separately from the event date attached to an extracted claim.
#[derive(Clone, Debug)]
pub struct SourceSnippet {
    position: usize,
    day: String,
    speaker: String,
    text: String,
    annotation: String,
}

pub fn source_annotations(history: &History) -> HashMap<String, SourceSnippet> {
    history
        .sessions
        .iter()
        .flat_map(|session| {
            session.turns.iter().map(|turn| {
                let annotations = session
                    .date
                    .map(|date| {
                        keep_knowledge::relative_time::in_text(&turn.text, date.date_naive())
                            .into_iter()
                            .map(|r| format!("\"{}\" = {}", r.phrase, r.resolved))
                            .collect::<Vec<_>>()
                            .join("; ")
                    })
                    .unwrap_or_default();
                (
                    turn.id.clone(),
                    SourceSnippet {
                        position: 0,
                        day: session
                            .date
                            .map(|d| d.date_naive().to_string())
                            .unwrap_or_else(|| "unknown".into()),
                        speaker: turn.role.label().into(),
                        text: turn.text.clone(),
                        annotation: annotations,
                    },
                )
            })
        })
        .enumerate()
        .map(|(position, (id, mut source))| {
            source.position = position;
            (id, source)
        })
        .collect()
}

pub fn attach_sources(
    line: &mut FactLine,
    sources: &HashMap<String, SourceSnippet>,
    provenance: bool,
    dates: bool,
) {
    let ids: BTreeSet<_> = line.turns.iter().collect();
    let mut cited: Vec<_> = ids
        .into_iter()
        .filter_map(|id| sources.get(id).map(|s| (id, s)))
        .collect();
    cited.sort_by_key(|(_, source)| source.position);
    for (id, source) in cited {
        let day = &source.day;
        if provenance {
            let snippet: String = source.text.chars().take(600).collect();
            let cut = if snippet.len() < source.text.len() {
                " [source truncated]"
            } else {
                ""
            };
            line.text.push_str(&format!(
                "\n  Source [{}; session {day}] {}: {snippet}{cut}",
                id, source.speaker
            ));
        }
        if dates && !source.annotation.is_empty() {
            line.text.push_str(&format!(
                "\n  Source-time [{id}; session {day}]: {}",
                source.annotation
            ));
        }
    }
}

/// Rank individual facts rather than spending a whole group's budget on
/// its newest entries. Duplicate restatements never displace another fact.
pub fn individual_groups(lines: Vec<FactLine>, order: &[String]) -> Vec<FactGroup> {
    let mut by_id: HashMap<_, _> = lines.into_iter().map(|l| (l.id.clone(), l)).collect();
    let mut seen = BTreeSet::new();
    order
        .iter()
        .filter_map(|id| {
            let l = by_id.remove(id)?;
            let key = (
                l.subject.to_lowercase(),
                l.predicate.clone(),
                l.value.to_lowercase(),
                l.valid_from,
                l.valid_to,
                format!("{:?}", l.status),
            );
            if !seen.insert(key) {
                return None;
            }
            Some(FactGroup {
                subject: l.subject.clone(),
                predicate: l.predicate.clone(),
                lines: vec![l],
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use keep_knowledge::model::{FactStatus, TimeBasis};

    fn line(id: &str, value: &str) -> FactLine {
        FactLine {
            id: id.into(),
            subject: "Mira".into(),
            predicate: "activity".into(),
            value: value.into(),
            text: format!("Mira {value}."),
            valid_from: None,
            valid_to: None,
            basis: TimeBasis::Observed,
            status: FactStatus::Active,
            turns: vec!["t1".into()],
            age_days: None,
        }
    }

    #[test]
    fn relevant_old_fact_is_kept_instead_of_newest_in_group() {
        let old = line("old", "studied pottery");
        let recent = line("recent", "visited a park");
        let duplicate = line("duplicate", "studied pottery");
        let groups = individual_groups(
            vec![recent, duplicate, old],
            &["old".into(), "duplicate".into(), "recent".into()],
        );
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].lines[0].id, "old");
    }

    #[test]
    fn provenance_quotes_only_cited_source_and_keeps_session_anchor() {
        use crate::dataset::{Session, Turn};
        use keep_core::knowledge::Role;
        let source = "I will visit next month.";
        let history = History {
            id: "h".into(),
            sessions: vec![Session {
                id: "s".into(),
                date_raw: None,
                date: Some("2024-12-31T12:00:00Z".parse().unwrap()),
                turns: vec![
                    Turn {
                        id: "t1".into(),
                        session_id: "s".into(),
                        role: Role::User,
                        text: source.into(),
                    },
                    Turn {
                        id: "t2".into(),
                        session_id: "s".into(),
                        role: Role::Assistant,
                        text: "unrelated answer".into(),
                    },
                ],
            }],
        };
        let mut fact = line("f", "plans a visit");
        attach_sources(&mut fact, &source_annotations(&history), true, true);
        assert!(fact.text.contains(source));
        assert!(fact.text.contains("session 2024-12-31"));
        assert!(fact.text.contains("\"next month\" = January 2025"));
        assert!(!fact.text.contains("unrelated answer"));
    }

    #[test]
    fn addressing_listener_in_a_source_does_not_change_fact_ownership() {
        use keep_knowledge::subject::Grounding;
        let grounding =
            Grounding::of("What did Lena study?", &["Mira".into(), "Lena".into()]).unwrap();
        let mut fact = line("mira-study", "studied pottery");
        let others = other_fact_ids(std::slice::from_ref(&fact), &grounding);
        fact.text
            .push_str("\n  Source [t1; session 2024-06-01] Mira: Lena, I studied pottery.");
        let group = individual_groups(vec![fact], &["mira-study".into()]);
        let block = crate::facts::assemble_grounded(
            &group,
            |group| group.lines.iter().all(|l| others.contains(&l.id)),
            "Facts about Mira, not about Lena:",
            1000,
            &keep_providers::ApproxTokenizer,
        );
        assert!(block.text.contains("Facts about Mira, not about Lena:\n-"));
    }

    #[test]
    fn facets_keep_subject_and_are_bounded_without_gold() {
        let q = "Where did Mira work and what did she study?";
        let queries = facets(q, &["Mira".into(), "Lena".into()]);
        assert_eq!(queries.len(), 3);
        assert_eq!(queries[0], q);
        assert!(queries.iter().all(|q| q.contains("Mira")));
        assert_eq!(facets("Where did Mira work?", &[]).len(), 1);
    }
}
