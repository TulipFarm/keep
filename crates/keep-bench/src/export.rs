//! Export an arm's retrieval for external evaluation harnesses.
//!
//! `keep-bench export` builds each arm's context for every question (no
//! answer or judge model is called) and writes it in the per-question
//! format of Mem0's open `memory-benchmarks` harness, so that harness can
//! answer and judge it unmodified with `--evaluate-only`. `keep-bench
//! hypotheses` turns a finished run's answers into the hypothesis file of
//! the official LongMemEval evaluation script.
//!
//! Layout written under the output directory:
//!
//! ```text
//! export.json                       what was exported, hashes, commands
//! id_map.json                       Keep question id -> harness ids
//! dataset.json                      reference data for the harness
//! <arm>/predicted_<project>/*.json  one retrieval file per question
//! ```
//!
//! For LoCoMo, `dataset.json` holds the exported conversations in their
//! original order with `qa` narrowed to the exported questions, so the
//! harness's `conv{i}_q{j}` indices match. For LongMemEval it holds the
//! exported question entries without `haystack_sessions`, which neither
//! harness reads when evaluating saved retrieval.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use keep_core::ContentHash;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::artifacts::{Outcome, QuestionRecord, write_json};
use crate::dataset::{Dataset, DatasetKind};
use crate::manifest::LoadedManifest;
use crate::protocol::{self, Memory, SEGMENTATION_VERSION};
use crate::runner::{Mode, RunOptions, Selection, run_selected};

pub const EXPORT_SCHEMA: &str = "keep-export-v1";

/// One question as the harness will see it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IdMapEntry {
    pub keep_question_id: String,
    /// The harness's question id (`conv{i}_q{j}` for LoCoMo, the dataset's
    /// own id for LongMemEval).
    pub harness_question_id: String,
    /// LoCoMo: indices in the original dataset file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_conversation_idx: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_qa_idx: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArmExport {
    pub arm: String,
    pub output_dir: String,
    pub written: usize,
    /// Questions with no retrieval file because the arm could not build a
    /// context (for example full context larger than the answer window).
    /// The harness refuses to evaluate until they exist, so the gap is
    /// never silently scored.
    pub missing: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExportReport {
    pub schema: String,
    pub manifest_id: String,
    pub manifest_hash: ContentHash,
    pub dataset: String,
    pub dataset_hash: ContentHash,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protocol: Option<protocol::ProtocolRecord>,
    pub project: String,
    pub segmentation: String,
    pub single_memory: bool,
    pub consumes_holdout: bool,
    pub questions: usize,
    pub reference_file: String,
    pub reference_sha256: ContentHash,
    pub arms: Vec<ArmExport>,
    /// Shell commands that evaluate this export with the external harness.
    pub commands: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub non_canonical: Option<String>,
}

pub struct ExportRequest {
    pub out: PathBuf,
    pub project: String,
    /// Export each context as one undated memory instead of segmenting it.
    pub single_memory: bool,
    pub selection: Selection,
}

/// Retrieval results in the shape of Mem0's `format_search_results`:
/// ranked by descending score, with ids that name the arm and question.
pub fn search_results(arm: &str, question_id: &str, memories: &[Memory]) -> Vec<Value> {
    let n = memories.len().max(1) as f64;
    memories
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let mut v = json!({
                "memory": m.memory,
                "score": 1.0 - i as f64 / n,
                "id": format!("keep:{arm}:{question_id}:{i}"),
            });
            if let Some(c) = &m.created_at {
                v["created_at"] = json!(c);
            }
            v
        })
        .collect()
}

fn memories(text: &str, single: bool) -> Vec<Memory> {
    if single {
        let t = text.trim();
        return if t.is_empty() {
            Vec::new()
        } else {
            vec![Memory {
                memory: t.to_owned(),
                created_at: None,
            }]
        };
    }
    protocol::segment(text)
}

fn retrieval(arm: &str, qid: &str, text: &str, query: &str, single: bool) -> Value {
    let results = search_results(arm, qid, &memories(text, single));
    json!({
        "search_query": query,
        "total_results": results.len(),
        "search_results": results,
        "search_latency_ms": 0.0,
    })
}

fn safe_name(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

pub async fn export(lm: &LoadedManifest, req: ExportRequest) -> anyhow::Result<ExportReport> {
    anyhow::ensure!(
        !req.out.exists() || std::fs::read_dir(&req.out)?.next().is_none(),
        "{} already exists and is not empty; exports are never overwritten",
        req.out.display()
    );
    anyhow::ensure!(
        !req.project.is_empty() && safe_name(&req.project) == req.project,
        "project name must use only letters, digits, '-' and '_'"
    );
    let m = &lm.manifest;
    let selection = Selection {
        capture_context_text: true,
        ..req.selection.clone()
    };
    let run = run_selected(
        lm,
        &RunOptions {
            mode: Mode::Contexts,
            unseal: false,
            allow_model_calls: false,
            results_root: None,
            spend: None,
        },
        &selection,
    )
    .await?;
    let bytes = lm.read_pinned(&crate::manifest::PinnedFile {
        path: m.dataset.path.clone(),
        sha256: m.dataset.sha256.clone(),
    })?;
    let raw: Value = serde_json::from_slice(&bytes)?;
    let dataset = Dataset::from_bytes(m.dataset.kind, &bytes)?;
    let exported: BTreeSet<&str> = run.records.iter().map(|r| r.question_id.as_str()).collect();
    std::fs::create_dir_all(&req.out)?;

    let (reference, id_map) = match m.dataset.kind {
        DatasetKind::Locomo => locomo_reference(&raw, &exported)?,
        _ => longmemeval_reference(&raw, &exported)?,
    };
    let reference_path = req.out.join("dataset.json");
    let reference_bytes = serde_json::to_vec_pretty(&reference)?;
    std::fs::write(&reference_path, &reference_bytes)?;
    write_json(&req.out.join("id_map.json"), &id_map)?;
    let by_keep: HashMap<&str, &IdMapEntry> = id_map
        .iter()
        .map(|e| (e.keep_question_id.as_str(), e))
        .collect();
    let questions: HashMap<&str, &crate::dataset::Question> = dataset
        .questions
        .iter()
        .map(|q| (q.id.as_str(), q))
        .collect();
    let texts: HashMap<(&str, &str), &str> = run
        .context_texts
        .iter()
        .map(|c| ((c.arm.as_str(), c.question_id.as_str()), c.text.as_str()))
        .collect();

    let mut arms = Vec::new();
    for arm in m
        .arms
        .iter()
        .filter(|a| selection.arms.as_ref().is_none_or(|s| s.contains(&a.name)))
    {
        let arm_dir = req.out.join(safe_name(&arm.name));
        let dir = arm_dir.join(format!("predicted_{}", req.project));
        std::fs::create_dir_all(&dir)?;
        let mut e = ArmExport {
            arm: arm.name.clone(),
            output_dir: arm_dir.display().to_string(),
            written: 0,
            missing: BTreeMap::new(),
        };
        for r in run.records.iter().filter(|r| r.arm == arm.name) {
            let Some(text) = texts.get(&(arm.name.as_str(), r.question_id.as_str())) else {
                e.missing
                    .insert(r.question_id.clone(), outcome_reason(&r.outcome));
                continue;
            };
            let q = questions[r.question_id.as_str()];
            let ids = by_keep[r.question_id.as_str()];
            let hid = &ids.harness_question_id;
            let file = match m.dataset.kind {
                DatasetKind::Locomo => {
                    locomo_file(&raw, ids, r, q, &arm.name, text, req.single_memory)?
                }
                _ => longmemeval_file(&raw, ids, r, q, &arm.name, text, req.single_memory)?,
            };
            write_json(&dir.join(format!("{hid}.json")), &file)?;
            e.written += 1;
        }
        arms.push(e);
    }

    let commands = commands(m.dataset.kind, &req, &reference, &arms);
    let report = ExportReport {
        schema: EXPORT_SCHEMA.into(),
        manifest_id: m.id.clone(),
        manifest_hash: lm.hash.clone(),
        dataset: m.dataset.kind.as_str().into(),
        dataset_hash: m.dataset.sha256.clone(),
        protocol: m.protocol.record(),
        project: req.project.clone(),
        segmentation: if req.single_memory {
            "single-memory".into()
        } else {
            SEGMENTATION_VERSION.into()
        },
        single_memory: req.single_memory,
        consumes_holdout: m.protocol.split == protocol::SplitScope::Full,
        questions: exported.len(),
        reference_file: reference_path.display().to_string(),
        reference_sha256: ContentHash::of_bytes(&reference_bytes),
        arms,
        commands,
        non_canonical: m.non_canonical_label().map(str::to_owned),
    };
    write_json(&req.out.join("export.json"), &report)?;
    Ok(report)
}

fn outcome_reason(o: &Outcome) -> String {
    match o {
        Outcome::Unavailable { reason } | Outcome::NotRun { reason } => reason.clone(),
        Outcome::Excluded { reason } => format!("excluded: {reason}"),
        Outcome::Failed { error } => format!("failed: {error}"),
        _ => "no context was built".into(),
    }
}

/// `(conversation index, qa index)` from a Keep LoCoMo question id
/// (`{sample_id}:q{qa index}`) and the original file.
fn locomo_position(raw: &Value, keep_id: &str) -> anyhow::Result<(usize, usize)> {
    let (sample, q) = keep_id
        .rsplit_once(":q")
        .ok_or_else(|| anyhow::anyhow!("{keep_id} is not a LoCoMo question id"))?;
    let qi: usize = q.parse()?;
    let ci = raw
        .as_array()
        .and_then(|a| {
            a.iter()
                .position(|s| s["sample_id"].as_str() == Some(sample))
        })
        .ok_or_else(|| anyhow::anyhow!("{sample} is not in the dataset file"))?;
    Ok((ci, qi))
}

fn locomo_reference(
    raw: &Value,
    exported: &BTreeSet<&str>,
) -> anyhow::Result<(Value, Vec<IdMapEntry>)> {
    let mut wanted: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    let mut originals: HashMap<(usize, usize), &str> = HashMap::new();
    for id in exported {
        let (ci, qi) = locomo_position(raw, id)?;
        wanted.entry(ci).or_default().insert(qi);
        originals.insert((ci, qi), id);
    }
    let samples = raw.as_array().expect("checked by locomo_position");
    let mut out = Vec::new();
    let mut map = Vec::new();
    for (new_ci, (ci, qis)) in wanted.iter().enumerate() {
        let mut sample = samples[*ci].clone();
        let qa = sample["qa"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("conversation {ci} has no qa list"))?;
        let kept: Vec<Value> = qis.iter().map(|qi| qa[*qi].clone()).collect();
        for (new_qi, qi) in qis.iter().enumerate() {
            map.push(IdMapEntry {
                keep_question_id: originals[&(*ci, *qi)].to_owned(),
                harness_question_id: format!("conv{new_ci}_q{new_qi}"),
                original_conversation_idx: Some(*ci),
                original_qa_idx: Some(*qi),
            });
        }
        sample["qa"] = Value::Array(kept);
        out.push(sample);
    }
    Ok((Value::Array(out), map))
}

fn longmemeval_reference(
    raw: &Value,
    exported: &BTreeSet<&str>,
) -> anyhow::Result<(Value, Vec<IdMapEntry>)> {
    let entries = raw
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("LongMemEval file is not a list"))?;
    let mut out = Vec::new();
    let mut map = Vec::new();
    for e in entries {
        let Some(id) = e["question_id"].as_str() else {
            continue;
        };
        if !exported.contains(id) {
            continue;
        }
        let mut e = e.clone();
        if let Some(obj) = e.as_object_mut() {
            obj.remove("haystack_sessions");
        }
        out.push(e);
        map.push(IdMapEntry {
            keep_question_id: id.into(),
            harness_question_id: id.into(),
            original_conversation_idx: None,
            original_qa_idx: None,
        });
    }
    anyhow::ensure!(
        map.len() == exported.len(),
        "some exported questions are not in the dataset file"
    );
    Ok((Value::Array(out), map))
}

/// Python's `str()` of a JSON answer, as the harnesses apply it.
fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        other => other.to_string(),
    }
}

fn locomo_file(
    raw: &Value,
    ids: &IdMapEntry,
    r: &QuestionRecord,
    q: &crate::dataset::Question,
    arm: &str,
    text: &str,
    single: bool,
) -> anyhow::Result<Value> {
    let (ci, qi) = (
        ids.original_conversation_idx.expect("locomo map"),
        ids.original_qa_idx.expect("locomo map"),
    );
    let qa = &raw[ci]["qa"][qi];
    let code = qa["category"].as_u64().unwrap_or_default();
    let conv_idx: usize = ids
        .harness_question_id
        .strip_prefix("conv")
        .and_then(|s| s.split('_').next())
        .and_then(|s| s.parse().ok())
        .expect("harness id is conv{i}_q{j}");
    Ok(json!({
        "question_id": ids.harness_question_id,
        "conversation_idx": conv_idx,
        "category": code,
        "category_name": protocol::locomo_category_name(&r.category).unwrap_or("unknown"),
        "question": qa["question"].as_str().unwrap_or(&q.query.text),
        "ground_truth_answer": py_str(qa.get("answer").unwrap_or(&Value::Null)),
        "evidence": qa.get("evidence").cloned().unwrap_or(json!([])),
        "user_id": format!("keep_{}_conv{conv_idx}", safe_name(arm)),
        "reference_date": locomo_reference_date(&raw[ci]),
        "retrieval": retrieval(arm, &r.question_id, text, &q.query.text, single),
        "keep": {"question_id": r.question_id, "arm": arm},
    }))
}

/// Mem0's reference date: the raw date of the last session by number.
fn locomo_reference_date(sample: &Value) -> Value {
    let Some(conv) = sample["conversation"].as_object() else {
        return Value::Null;
    };
    conv.iter()
        .filter_map(|(k, v)| {
            let n: u32 = k
                .strip_prefix("session_")?
                .strip_suffix("_date_time")?
                .parse()
                .ok()?;
            Some((n, v.clone()))
        })
        .max_by_key(|(n, _)| *n)
        .map(|(_, v)| v)
        .unwrap_or(Value::Null)
}

fn longmemeval_file(
    raw: &Value,
    ids: &IdMapEntry,
    r: &QuestionRecord,
    q: &crate::dataset::Question,
    arm: &str,
    text: &str,
    single: bool,
) -> anyhow::Result<Value> {
    let entry = raw
        .as_array()
        .and_then(|a| {
            a.iter()
                .find(|e| e["question_id"].as_str() == Some(ids.harness_question_id.as_str()))
        })
        .ok_or_else(|| anyhow::anyhow!("{} not in dataset", ids.harness_question_id))?;
    Ok(json!({
        "question_id": ids.harness_question_id,
        "question_type": entry["question_type"],
        "question": entry["question"],
        "ground_truth_answer": py_str(&entry["answer"]),
        "question_date": entry.get("question_date").cloned().unwrap_or(json!("")),
        "is_abstention": ids.harness_question_id.ends_with("_abs"),
        "user_id": format!("keep_{}_{}", safe_name(arm), safe_name(&ids.harness_question_id)),
        "answer_session_ids": entry.get("answer_session_ids").cloned().unwrap_or(json!([])),
        "retrieval": retrieval(arm, &r.question_id, text, &q.query.text, single),
        "keep": {"question_id": r.question_id, "arm": arm},
    }))
}

fn commands(
    kind: DatasetKind,
    req: &ExportRequest,
    reference: &Value,
    arms: &[ArmExport],
) -> Vec<String> {
    let dataset = req.out.join("dataset.json");
    let n = reference.as_array().map_or(0, Vec::len);
    arms.iter()
        .map(|a| match kind {
            DatasetKind::Locomo => format!(
                "python -m benchmarks.locomo.run --project-name {} --evaluate-only --dataset-path {} \
                 --output-dir {} --conversations {} --categories 1,2,3,4 --top-k-cutoffs 200 \
                 --answerer-model gpt-4o-mini --judge-model gpt-4o-mini",
                req.project,
                dataset.display(),
                a.output_dir,
                (0..n).map(|i| i.to_string()).collect::<Vec<_>>().join(","),
            ),
            _ => format!(
                "python -m benchmarks.longmemeval.run --project-name {} --evaluate-only --all-questions \
                 --dataset-path {} --output-dir {} --top-k-cutoffs 200",
                req.project,
                dataset.display(),
                a.output_dir,
            ),
        })
        .collect()
}

/// One line of the official LongMemEval hypothesis file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hypothesis {
    pub question_id: String,
    pub hypothesis: String,
}

/// Hypotheses for one arm and seed of a finished run, in record order.
/// Questions without an answer get an empty hypothesis so they are scored
/// (as wrong) rather than dropped; the count is returned beside them.
pub fn hypotheses(
    records: &[QuestionRecord],
    arm: &str,
    seed: Option<u64>,
) -> anyhow::Result<(Vec<Hypothesis>, usize)> {
    let seed = match seed {
        Some(s) => s,
        None => records
            .iter()
            .find(|r| r.arm == arm)
            .map(|r| r.seed)
            .ok_or_else(|| anyhow::anyhow!("no records for arm {arm}"))?,
    };
    let mut unanswered = 0;
    let out = records
        .iter()
        .filter(|r| r.arm == arm && r.seed == seed)
        .filter(|r| !matches!(r.outcome, Outcome::Excluded { .. }))
        .map(|r| {
            let text = r.answer.as_ref().and_then(|a| a.text.clone());
            if text.is_none() {
                unanswered += 1;
            }
            Hypothesis {
                question_id: r.question_id.clone(),
                hypothesis: text.unwrap_or_default(),
            }
        })
        .collect::<Vec<_>>();
    anyhow::ensure!(!out.is_empty(), "no records for arm {arm} and seed {seed}");
    Ok((out, unanswered))
}

pub fn write_hypotheses(path: &Path, hyps: &[Hypothesis]) -> anyhow::Result<()> {
    crate::artifacts::write_jsonl(path, hyps)
}

/// Read a hypothesis file back.
pub fn read_hypotheses(path: &Path) -> anyhow::Result<Vec<Hypothesis>> {
    std::fs::read_to_string(path)?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| Ok(serde_json::from_str(l)?))
        .collect()
}
