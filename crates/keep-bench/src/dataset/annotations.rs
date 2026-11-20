//! Known-bad gold answers and exclusions, recorded as data.
//!
//! Some public benchmarks contain wrong gold answers. The harness never edits
//! a dataset in place. Instead an annotation file lists question ids with a
//! kind, a source and a note. The `original` gold track scores every
//! question with the dataset's own answers and only records annotations; the
//! `corrected` track applies them and is always published alongside the
//! original, never instead of it. No annotations ship by default.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AnnotationKind {
    /// Gold answer believed wrong; `corrected_answer` must be given.
    WrongGold,
    /// Question is ambiguous or unscorable; excluded in the corrected track.
    Exclude,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Annotation {
    pub question_id: String,
    pub kind: AnnotationKind,
    #[serde(default)]
    pub corrected_answer: Option<String>,
    /// Where the judgement comes from: an audit URL or a review record.
    pub source: String,
    #[serde(default)]
    pub note: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AnnotationFile {
    #[serde(default, rename = "annotation")]
    pub annotations: Vec<Annotation>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GoldTrack {
    #[default]
    Original,
    Corrected,
}

/// What a gold track does to one question.
#[derive(Clone, Debug, PartialEq)]
pub enum GoldDecision {
    Score { answer: String, corrected: bool },
    Excluded { reason: String },
}

impl AnnotationFile {
    pub fn parse(text: &str) -> anyhow::Result<Self> {
        let file: AnnotationFile = toml::from_str(text)?;
        let mut seen = BTreeMap::new();
        for a in &file.annotations {
            anyhow::ensure!(
                !a.source.trim().is_empty(),
                "annotation {} has no source",
                a.question_id
            );
            if a.kind == AnnotationKind::WrongGold {
                anyhow::ensure!(
                    a.corrected_answer
                        .as_deref()
                        .is_some_and(|s| !s.trim().is_empty()),
                    "wrong-gold annotation {} needs corrected_answer",
                    a.question_id
                );
            }
            anyhow::ensure!(
                seen.insert(a.question_id.clone(), ()).is_none(),
                "duplicate annotation for {}",
                a.question_id
            );
        }
        Ok(file)
    }

    pub fn decide(&self, track: GoldTrack, question_id: &str, gold: &str) -> GoldDecision {
        let annotation = self
            .annotations
            .iter()
            .find(|a| a.question_id == question_id);
        match (track, annotation) {
            (GoldTrack::Original, _) | (GoldTrack::Corrected, None) => GoldDecision::Score {
                answer: gold.to_owned(),
                corrected: false,
            },
            (GoldTrack::Corrected, Some(a)) => match a.kind {
                AnnotationKind::Exclude => GoldDecision::Excluded {
                    reason: format!("{} ({})", a.note, a.source),
                },
                AnnotationKind::WrongGold => GoldDecision::Score {
                    answer: a.corrected_answer.clone().unwrap_or_default(),
                    corrected: true,
                },
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FILE: &str = r#"
[[annotation]]
question_id = "conv-a:q1"
kind = "wrong-gold"
corrected_answer = "Tuesday"
source = "independent review 1"

[[annotation]]
question_id = "conv-a:q2"
kind = "exclude"
source = "independent review 1"
note = "ambiguous speaker"
"#;

    #[test]
    fn original_track_ignores_annotations_and_corrected_applies_them() {
        let f = AnnotationFile::parse(FILE).unwrap();
        assert_eq!(
            f.decide(GoldTrack::Original, "conv-a:q1", "Monday"),
            GoldDecision::Score {
                answer: "Monday".into(),
                corrected: false
            }
        );
        assert_eq!(
            f.decide(GoldTrack::Corrected, "conv-a:q1", "Monday"),
            GoldDecision::Score {
                answer: "Tuesday".into(),
                corrected: true
            }
        );
        assert!(matches!(
            f.decide(GoldTrack::Corrected, "conv-a:q2", "x"),
            GoldDecision::Excluded { .. }
        ));
        assert!(AnnotationFile::parse("").unwrap().annotations.is_empty());
    }

    #[test]
    fn annotations_need_sources_and_corrections() {
        let missing_answer = "[[annotation]]\nquestion_id='q'\nkind='wrong-gold'\nsource='s'\n";
        assert!(AnnotationFile::parse(missing_answer).is_err());
        let missing_source = "[[annotation]]\nquestion_id='q'\nkind='exclude'\nsource=' '\n";
        assert!(AnnotationFile::parse(missing_source).is_err());
    }
}
