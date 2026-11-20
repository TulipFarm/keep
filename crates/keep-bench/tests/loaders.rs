mod common;

use keep_bench::dataset::{Dataset, DatasetKind};
use keep_core::knowledge::Role;

fn locomo() -> Dataset {
    Dataset::load(
        DatasetKind::Locomo,
        &common::fixtures().join("locomo-mini.json"),
    )
    .unwrap()
}

fn longmemeval() -> Dataset {
    Dataset::load(
        DatasetKind::LongmemevalS,
        &common::fixtures().join("longmemeval-mini.json"),
    )
    .unwrap()
}

#[test]
fn locomo_normalises_conversations_sessions_and_questions() {
    let ds = locomo();
    assert_eq!(ds.kind, DatasetKind::Locomo);
    assert_eq!(ds.histories.len(), 6);
    assert_eq!(ds.questions.len(), 31);

    let h = ds.history("synth-1").unwrap();
    assert_eq!(h.sessions.len(), 3);
    let s1 = &h.sessions[0];
    assert_eq!(s1.date_raw.as_deref(), Some("1:56 pm on 8 May, 2021"));
    assert_eq!(s1.date.unwrap().to_rfc3339(), "2021-05-08T13:56:00+00:00");
    assert!(s1.date < h.sessions[1].date && h.sessions[1].date < h.sessions[2].date);
    assert_eq!(s1.turns[0].id, "synth-1:D1:1");
    assert_eq!(s1.turns[0].role, Role::Speaker("Mira".into()));
    // Image captions are kept as text, since they carry answers.
    assert!(
        s1.turns[3]
            .text
            .contains("[shared an image: a photo of a small animal on a sofa]")
    );

    let q = ds.questions.iter().find(|q| q.id == "synth-1:q3").unwrap();
    assert_eq!(q.answer, "cello; Lisbon");
    assert_eq!(q.category, "cat-3");
    // "D2:4; D2:1" lists two evidence turns in one string.
    assert_eq!(q.evidence_turn_ids, ["synth-1:D2:4", "synth-1:D2:1"]);
    assert_eq!(q.evidence_session_ids, ["synth-1:session_2"]);
    // LoCoMo questions carry no question date.
    assert!(q.query.asked_at.is_none());
}

#[test]
fn locomo_adversarial_questions_become_abstention_with_distractor() {
    let ds = locomo();
    let adv: Vec<_> = ds.questions.iter().filter(|q| q.abstention).collect();
    assert_eq!(adv.len(), 6);
    for q in adv {
        assert_eq!(q.category, "adversarial");
        assert_eq!(q.distractor.as_deref(), Some("The Swift"));
        assert!(q.evidence_turn_ids.is_empty());
    }
}

#[test]
fn locomo_bad_evidence_is_reported_not_dropped_silently() {
    let ds = locomo();
    let q = ds.questions.iter().find(|q| q.id == "synth-1:q5").unwrap();
    assert!(q.evidence_turn_ids.is_empty());
    assert!(
        ds.warnings
            .iter()
            .any(|w| w.contains("synth-1:q5") && w.contains("D9:99")),
        "{:?}",
        ds.warnings
    );
}

#[test]
fn longmemeval_one_history_per_question_with_dated_query() {
    let ds = longmemeval();
    assert_eq!(ds.histories.len(), 12);
    assert_eq!(ds.questions.len(), 12);
    let q = &ds.questions[0];
    assert_eq!(q.id, "q000");
    assert_eq!(q.history_id, "q000");
    assert_eq!(q.category, "single-session-user");
    assert_eq!(
        q.query.asked_at.unwrap().to_rfc3339(),
        "2023-06-01T09:00:00+00:00"
    );

    let h = ds.history("q000").unwrap();
    assert_eq!(h.sessions.len(), 4);
    assert_eq!(h.sessions[2].id, "q000:2:sess_0_2");
    assert_eq!(h.sessions[0].turns[0].role, Role::User);
    assert_eq!(h.sessions[0].turns[1].role, Role::Assistant);
    // has_answer marks evidence turns; answer_session_ids marks sessions.
    assert_eq!(q.evidence_turn_ids, ["q000:2:sess_0_2:0"]);
    assert_eq!(q.evidence_session_ids, ["q000:2:sess_0_2"]);
}

#[test]
fn longmemeval_abstention_comes_from_the_id_suffix() {
    let ds = longmemeval();
    let abs: Vec<&str> = ds
        .questions
        .iter()
        .filter(|q| q.abstention)
        .map(|q| q.id.as_str())
        .collect();
    assert_eq!(abs, ["q003_abs", "q007_abs"]);
}

#[test]
fn baselines_see_only_the_query_never_scorer_fields() {
    // The query is the only question data a baseline receives. Its
    // serialised form must not contain the gold answer, evidence or labels.
    for ds in [locomo(), longmemeval()] {
        for q in &ds.questions {
            let v = serde_json::to_value(&q.query).unwrap();
            let keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
            assert_eq!(keys, ["asked_at", "asked_at_raw", "id", "text"]);
            let text = v.to_string();
            assert!(
                !text.contains(&format!("\"{}\"", q.answer)),
                "{} leaks its answer",
                q.id
            );
        }
    }
}

#[test]
fn source_hash_is_the_file_hash() {
    let bytes = std::fs::read(common::fixtures().join("locomo-mini.json")).unwrap();
    assert_eq!(
        locomo().source_hash,
        keep_core::ContentHash::of_bytes(&bytes)
    );
}
