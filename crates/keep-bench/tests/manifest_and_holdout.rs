mod common;

use std::collections::HashSet;

use common::{fixture_manifest, set, write_manifest};
use keep_bench::holdout;
use keep_bench::manifest::{LoadedManifest, has_errors};
use keep_bench::prepare;

const LME: &str = "longmemeval-mock.toml";

fn errors(lm: &LoadedManifest) -> Vec<String> {
    prepare::check(lm)
        .into_iter()
        .filter(|i| i.error)
        .map(|i| format!("{}: {}", i.field, i.message))
        .collect()
}

#[test]
fn fixture_manifests_are_valid() {
    for name in ["longmemeval-mock.toml", "locomo-mock.toml"] {
        let lm = common::load(name);
        assert!(errors(&lm).is_empty(), "{name}: {:?}", errors(&lm));
    }
}

#[test]
fn hash_ignores_formatting_but_tracks_every_setting() {
    let dir = tempfile::tempdir().unwrap();
    let text = fixture_manifest(LME);
    let base = write_manifest(dir.path(), &text).hash;

    // Comments, blank lines and explicit defaults do not change the hash.
    let noisy = format!("# a new comment\n\n{text}\n\n[dispatch]\nallow_model_calls = false\n");
    assert_eq!(write_manifest(dir.path(), &noisy).hash, base);

    for (key, value) in [
        ("seeds", "[1, 2, 4]"),
        ("max_output_tokens", "65"),
        ("salt", "\"another-salt\""),
        ("profile", "\"postgres-vector\""),
        ("bm25_k1", "1.5"),
    ] {
        let changed = write_manifest(dir.path(), &set(&text, key, value)).hash;
        assert_ne!(changed, base, "changing {key} must change the hash");
    }
}

#[test]
fn unknown_fields_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let text = fixture_manifest(LME).replace(
        "phase = \"development\"",
        "phase = \"development\"\nanswer_modle = \"x\"",
    );
    std::fs::write(dir.path().join("m.toml"), text).unwrap();
    let err = LoadedManifest::load(&dir.path().join("m.toml"))
        .unwrap_err()
        .to_string();
    assert!(err.contains("answer_modle"), "{err}");
}

#[test]
fn structural_rules_are_enforced() {
    let dir = tempfile::tempdir().unwrap();
    let text = fixture_manifest(LME);

    let no_full = text.replace(
        "name = \"full-context\"\nkind = \"full-context\"",
        "name = \"fc\"\nkind = \"no-memory\"",
    );
    assert!(
        errors(&write_manifest(dir.path(), &no_full))
            .iter()
            .any(|e| e.contains("full-context"))
    );

    let dup = text.replace("name = \"recent-window\"", "name = \"bm25\"");
    assert!(
        errors(&write_manifest(dir.path(), &dup))
            .iter()
            .any(|e| e.contains("bm25"))
    );

    // A confirmation run needs a frozen candidate, real providers, an exact
    // tokenizer and a fit mask; mocks are not acceptable there.
    let conf = set(&text, "phase", "\"confirmation\"");
    let e = errors(&write_manifest(dir.path(), &conf));
    for needle in ["confirmation", "fit_mask", "mock", "equal context"] {
        assert!(
            e.iter().any(|x| x.to_lowercase().contains(needle)),
            "missing {needle}: {e:?}"
        );
    }

    // An equal-context baseline (standing 1024 + evidence 256) clears it.
    let equal = format!(
        "{conf}\n[[arms]]\nname = \"bm25-equal\"\nkind = \"bm25\"\nbudget_tokens = 1280\ntop_k = 40\n"
    );
    let lm = write_manifest(dir.path(), &equal);
    assert_eq!(lm.manifest.has_equal_context_baseline(), Some(true));
    assert!(!errors(&lm).iter().any(|x| x.contains("equal context")));
}

#[test]
fn an_unsealed_holdout_is_invalid_and_cannot_run() {
    let dir = tempfile::tempdir().unwrap();
    let text: String = fixture_manifest(LME)
        .lines()
        .filter(|l| !l.starts_with("commitment"))
        .collect::<Vec<_>>()
        .join("\n");
    let lm = write_manifest(dir.path(), &text);
    assert!(errors(&lm).iter().any(|e| e.starts_with("holdout")));
    assert!(prepare::prepare(&lm, false).is_err());
    // Sealing reproduces the committed value of the fixture.
    let (commitment, held, total) = prepare::seal(&lm).unwrap();
    assert_eq!(
        Some(commitment),
        common::load(LME).manifest.holdout.commitment
    );
    assert_eq!((held, total), (4, 12));
}

#[test]
fn changing_the_holdout_definition_breaks_the_seal() {
    let dir = tempfile::tempdir().unwrap();
    let text = fixture_manifest(LME);
    for (key, value) in [("salt", "\"peeked-salt\""), ("fraction", "0.5")] {
        let lm = write_manifest(dir.path(), &set(&text, key, value));
        assert!(
            errors(&lm)
                .iter()
                .any(|e| e.starts_with("holdout.commitment")),
            "{key}"
        );
        let err = prepare::prepare(&lm, false).unwrap_err().to_string();
        assert!(err.contains("commitment mismatch"), "{err}");
    }
}

#[test]
fn a_changed_dataset_file_is_detected() {
    let dir = tempfile::tempdir().unwrap();
    let data = std::fs::read_to_string(common::fixtures().join("longmemeval-mini.json")).unwrap();
    std::fs::write(
        dir.path().join("lme.json"),
        data.replace("pottery", "ceramics"),
    )
    .unwrap();
    let text = fixture_manifest(LME).replace("\"../longmemeval-mini.json\"", "\"lme.json\"");
    let lm = write_manifest(dir.path(), &text);
    assert!(errors(&lm).iter().any(|e| e.starts_with("dataset")));
    assert!(prepare::prepare(&lm, false).is_err());
}

#[test]
fn development_runs_never_see_holdout_histories() {
    let lm = common::load(LME);
    let full = prepare::load_dataset(&lm).unwrap();
    let h = &lm.manifest.holdout;
    let held: HashSet<String> = holdout::holdout_ids(&full, &h.salt, h.fraction)
        .into_iter()
        .collect();
    assert_eq!(held.len(), 4);

    let p = prepare::prepare(&lm, false).unwrap();
    assert_eq!(p.dataset.histories.len(), 8);
    assert!(p.dataset.histories.iter().all(|x| !held.contains(&x.id)));
    assert!(
        p.dataset
            .questions
            .iter()
            .all(|q| !held.contains(&q.history_id))
    );

    // Membership depends only on salt and id, so adding data never moves an
    // existing history between splits.
    for id in &held {
        assert!(holdout::in_holdout(&h.salt, h.fraction, id));
    }

    let err = prepare::prepare(&lm, true).unwrap_err().to_string();
    assert!(err.contains("--unseal"), "{err}");
}

#[test]
fn subset_narrows_questions_after_the_split() {
    let dir = tempfile::tempdir().unwrap();
    let dev = prepare::prepare(&common::load(LME), false).unwrap();
    let picked: Vec<&str> = dev
        .dataset
        .questions
        .iter()
        .take(2)
        .map(|q| q.id.as_str())
        .collect();
    let subset = format!(
        "# two development questions\n{}\n{}\nnot-a-real-id\n",
        picked[0], picked[1]
    );
    std::fs::write(dir.path().join("subset.txt"), &subset).unwrap();
    let sha = keep_core::ContentHash::of_bytes(subset.as_bytes());
    let text = fixture_manifest(LME).replace(
        "annotations = {",
        &format!("subset = {{ path = \"subset.txt\", sha256 = \"{sha}\" }}\nannotations = {{"),
    );
    let p = prepare::prepare(&write_manifest(dir.path(), &text), false).unwrap();
    let ids: Vec<&str> = p.dataset.questions.iter().map(|q| q.id.as_str()).collect();
    assert_eq!(ids, picked);
    assert_eq!(p.dataset.histories.len(), 2);

    // An edited subset file no longer matches its pin.
    std::fs::write(dir.path().join("subset.txt"), format!("{}\n", picked[0])).unwrap();
    assert!(prepare::prepare(&write_manifest(dir.path(), &text), false).is_err());
}

#[test]
fn remote_pins_with_placeholders_block_dispatch() {
    let dir = tempfile::tempdir().unwrap();
    let text = fixture_manifest(LME).replacen(
        "provider = \"mock\"",
        "provider = \"openai-compatible\"",
        1,
    );
    let text = set(&text, "model", "\"PIN-REQUIRED\"");
    let lm = write_manifest(dir.path(), &text);
    assert!(!has_errors(&lm.manifest.validate()));
    let problems = lm.manifest.dispatch_problems();
    assert!(
        problems
            .iter()
            .any(|p| p.contains("answer model is not pinned")),
        "{problems:?}"
    );
    assert!(
        problems.iter().any(|p| p.contains("base_url_env")),
        "{problems:?}"
    );
    assert!(
        problems.iter().any(|p| p.contains("budget caps")),
        "{problems:?}"
    );
}

#[test]
fn fact_arms_need_a_matching_extraction_config() {
    let dir = tempfile::tempdir().unwrap();
    let text = fixture_manifest("facts-mock.toml");
    assert!(errors(&write_manifest(dir.path(), &text)).is_empty());

    let i = text.find("[extraction]").unwrap();
    let j = text[i..].find("\n\n").unwrap() + i;
    let no_extraction = format!("{}{}", &text[..i], &text[j + 2..]);
    let e = errors(&write_manifest(dir.path(), &no_extraction));
    assert!(
        e.iter().any(|x| x.contains("must be set together")),
        "{e:?}"
    );

    let too_big = set(&text, "facts_tokens", "256");
    let e = errors(&write_manifest(dir.path(), &too_big));
    assert!(e.iter().any(|x| x.contains("facts_tokens")), "{e:?}");

    let stale = text.replacen(
        "prompt_version = \"session-facts-v2\"\nfacts_tokens",
        "prompt_version = \"session-facts-v0\"\nfacts_tokens",
        1,
    );
    let e = errors(&write_manifest(dir.path(), &stale));
    assert!(
        e.iter().any(|x| x.contains("extraction.prompt_version")),
        "{e:?}"
    );

    let i = text.find("[[arms]]\nname = \"keep-k3\"").unwrap();
    let unused = text[..i].to_owned();
    let e = errors(&write_manifest(dir.path(), &unused));
    assert!(e.iter().any(|x| x.contains("no k3 or mk3 arm")), "{e:?}");
}

#[test]
fn fact_arm_options_are_checked_and_recorded() {
    let dir = tempfile::tempdir().unwrap();
    let text = fixture_manifest("facts-mock.toml");
    let lm = write_manifest(dir.path(), &text);
    let arm = lm
        .manifest
        .arms
        .iter()
        .find(|a| a.name == "keep-mk3-full")
        .unwrap();
    assert!(arm.temporal && arm.preferences);
    assert_eq!(arm.answer_prompt("answer-v1"), "answer-v3");
    let plain = lm
        .manifest
        .arms
        .iter()
        .find(|a| a.name == "keep-mk3")
        .unwrap();
    assert_eq!(plain.answer_prompt("answer-v1"), "answer-v1");

    let full = "path = \"mk3\"\ntemporal = true\npreferences = true\n";
    let cases = [
        (
            text.replace(
                full,
                "path = \"mk2\"\ntemporal = true\npreferences = true\n",
            ),
            "arms.keep-mk3-full.temporal",
        ),
        (
            text.replace(full, "path = \"k3\"\ntemporal = true\npreferences = true\n"),
            "arms.keep-mk3-full.preferences",
        ),
        (
            text.replace("profile_tokens = 128\n", ""),
            "arms.keep-mk3-full.preferences",
        ),
        (
            text.replace("profile_tokens = 128", "profile_tokens = 2048"),
            "extraction.profile_tokens",
        ),
        (
            text.replace("\"answer-v3\"", "\"answer-v9\""),
            "arms.keep-mk3-full.answer_prompt",
        ),
    ];
    for (bad, field) in cases {
        let e = errors(&write_manifest(dir.path(), &bad));
        assert!(e.iter().any(|x| x.contains(field)), "{field}: {e:?}");
    }
}

#[test]
fn v4_fact_options_are_checked() {
    let dir = tempfile::tempdir().unwrap();
    let text = fixture_manifest("facts-mock.toml");
    let summaries = "expected_output_tokens = 400\n\
                     summary_prompt_version = \"session-summary-v1\"\nsummary_tokens = 64\n";
    let with = |extraction: &str, arm: &str| {
        text.replacen("expected_output_tokens = 400\n", extraction, 1)
            + &format!("\n[[arms]]\nname = \"keep-mk4\"\nkind = \"keep\"\n{arm}")
    };
    let ok = with(
        summaries,
        "path = \"mk3\"\nsubjects = true\nsummaries = true\nexpand = true\n",
    );
    let lm = write_manifest(dir.path(), &ok);
    assert!(errors(&lm).is_empty(), "{:?}", errors(&lm));
    let arm = lm.manifest.arms.last().unwrap().fact_options();
    assert!(arm.subjects && arm.summaries && arm.expand);

    let cases = [
        (
            with(
                summaries,
                "path = \"mk2\"\nsubjects = true\nsummaries = true\n",
            ),
            "arms.keep-mk4.subjects",
        ),
        (
            with(
                "expected_output_tokens = 400\n",
                "path = \"mk3\"\nsummaries = true\n",
            ),
            "arms.keep-mk4.summaries",
        ),
        (
            with(summaries, "path = \"mk3\"\nexpand = true\n"),
            "no arm has summaries",
        ),
        (
            with(
                &summaries.replace("session-summary-v1", "session-summary-v0"),
                "path = \"mk3\"\nsummaries = true\n",
            ),
            "extraction.summary_prompt_version",
        ),
        (
            with(
                &summaries.replace("summary_tokens = 64\n", ""),
                "path = \"mk3\"\nsummaries = true\n",
            ),
            "must be set together",
        ),
    ];
    for (bad, field) in cases {
        let e = errors(&write_manifest(dir.path(), &bad));
        assert!(e.iter().any(|x| x.contains(field)), "{field}: {e:?}");
    }
}

#[test]
fn concurrency_override_comes_from_the_environment_and_must_be_positive() {
    use keep_bench::manifest::CONCURRENCY_ENV;
    let lm = common::load(LME);
    let m = &lm.manifest;
    let env = |v: &'static str| move |n: &str| (n == CONCURRENCY_ENV).then(|| v.to_owned());
    assert_eq!(m.dispatch_concurrency_with(|_| None).unwrap(), 1);
    assert_eq!(m.dispatch_concurrency_with(env("16")).unwrap(), 16);
    assert_eq!(m.dispatch_concurrency_with(env(" ")).unwrap(), 1);
    for bad in ["0", "-2", "4.5", "many", "1025"] {
        let e = m.dispatch_concurrency_with(env(bad)).unwrap_err();
        assert!(e.to_string().contains(CONCURRENCY_ENV), "{bad}: {e}");
    }
    // The override is not part of the manifest, so the hash cannot move.
    assert_eq!(m.hash(), lm.hash);
}
