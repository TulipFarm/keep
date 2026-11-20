//! Run summaries in JSON and Markdown.
//!
//! Accuracy is reported two ways so failures cannot hide: over every
//! scheduled question (failures, malformed verdicts and unavailable
//! questions count as incorrect) and over the questions the arm could serve.
//! The full-context row is always present; when it cannot run, it says so.

use std::collections::BTreeMap;

use keep_core::ContentHash;
use serde::{Deserialize, Serialize};

use crate::artifacts::{JudgeOutcome, Outcome, QuestionRecord};
use crate::context::{Slot, slot_tokens};
use crate::manifest::Manifest;
use crate::metrics::METRICS_VERSION;
use crate::runner::{Mode, RunMeta};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Rate {
    pub n: usize,
    pub correct: usize,
    pub accuracy: Option<f64>,
}

impl Rate {
    fn new(n: usize, correct: usize) -> Self {
        Self {
            n,
            correct,
            accuracy: (n > 0).then(|| correct as f64 / n as f64),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ArmSummary {
    pub arm: String,
    pub kind: String,
    pub diagnostic: bool,
    /// `ok`, `unavailable`, `partially-unavailable`, `not-run`, `estimated`.
    pub status: String,
    pub status_reason: Option<String>,
    pub records: usize,
    pub counts: BTreeMap<String, usize>,
    pub judge_malformed: usize,
    pub judge_failed: usize,
    /// Over all scheduled (non-excluded) records.
    pub accuracy_scheduled: Rate,
    /// Over records the arm could serve (not unavailable / not run).
    pub accuracy_available: Rate,
    #[serde(default)]
    pub accuracy_non_adversarial: Rate,
    /// Answered share of records the arm could serve.
    pub completion: Option<f64>,
    pub exact_match: Option<f64>,
    pub f1: Option<f64>,
    pub evidence_recall: Option<f64>,
    pub by_category: BTreeMap<String, Rate>,
    pub context_tokens_mean: Option<f64>,
    pub prompt_tokens_mean: Option<f64>,
    pub truncated_records: usize,
    pub answer_input_tokens: u64,
    pub answer_output_tokens: u64,
    pub judge_input_tokens: u64,
    pub judge_output_tokens: u64,
    pub cost_usd: f64,
    pub answer_latency_ms_p50: Option<u64>,
    pub answer_latency_ms_p95: Option<u64>,
    /// Keep arms only: per-record token totals of the standing and evidence
    /// allowances, and the mean of every named block.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<TokenAccounting>,
    /// Harness-billed AI credits (Copilot CLI runs), answer plus judge.
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub ai_credits: f64,
    /// Input tokens the harness added to answer calls on top of Keep's own
    /// prompt. Counted separately so Keep's context is never inflated by it.
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub answer_overhead_tokens: u64,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub judge_overhead_tokens: u64,
    /// Answer or judge calls refused because the run's credit or budget cap
    /// was reached.
    #[serde(default, skip_serializing_if = "is_zero_usize")]
    pub budget_refused: usize,
    /// Verdicts reused from an identical answer judged before (judge
    /// consistency only).
    #[serde(default, skip_serializing_if = "is_zero_usize")]
    pub judge_cached: usize,
    /// Judge calls beyond the first, cast by majority voting.
    #[serde(default, skip_serializing_if = "is_zero_usize")]
    pub judge_extra_votes: usize,
}

fn is_zero_f64(x: &f64) -> bool {
    *x == 0.0
}

fn is_zero_u64(x: &u64) -> bool {
    *x == 0
}

fn is_zero_usize(x: &usize) -> bool {
    *x == 0
}

fn budget_refusal(error: &str) -> bool {
    error.starts_with("budget exhausted")
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TokenAccounting {
    pub standing_mean: Option<f64>,
    pub standing_max: Option<usize>,
    pub evidence_mean: Option<f64>,
    pub evidence_max: Option<usize>,
    pub blocks_mean: BTreeMap<String, f64>,
}

fn token_accounting(records: &[&QuestionRecord]) -> Option<TokenAccounting> {
    let ctx: Vec<_> = records
        .iter()
        .filter_map(|r| r.context.as_ref())
        .filter(|c| !c.blocks.is_empty())
        .collect();
    if ctx.is_empty() {
        return None;
    }
    let per = |slot| -> Vec<usize> {
        ctx.iter()
            .filter_map(|c| slot_tokens(&c.blocks, slot))
            .collect()
    };
    let (standing, evidence) = (per(Slot::Standing), per(Slot::Evidence));
    let mut sums: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for b in ctx.iter().flat_map(|c| &c.blocks) {
        let e = sums.entry(b.name.clone()).or_default();
        e.0 += b.tokens;
        e.1 += 1;
    }
    Some(TokenAccounting {
        standing_mean: mean(standing.iter().map(|&x| x as f64)),
        standing_max: standing.iter().copied().max(),
        evidence_mean: mean(evidence.iter().map(|&x| x as f64)),
        evidence_max: evidence.iter().copied().max(),
        blocks_mean: sums
            .into_iter()
            .map(|(k, (t, n))| (k, t as f64 / n as f64))
            .collect(),
    })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    pub schema: String,
    pub run_id: String,
    pub mode: Mode,
    pub mock: bool,
    pub manifest_id: String,
    pub manifest_hash: ContentHash,
    pub manifest_file_hash: ContentHash,
    pub phase: String,
    pub dataset: String,
    pub dataset_hash: ContentHash,
    pub gold_track: String,
    pub storage_profile: String,
    pub questions: usize,
    pub histories: usize,
    pub seeds: Vec<u64>,
    pub metrics_version: String,
    pub tokenizer: String,
    pub tokenizer_exact: bool,
    pub fit_fraction: Option<f64>,
    pub arms: Vec<ArmSummary>,
    pub total_cost_usd: f64,
    pub embedding_input_tokens: u64,
    pub warnings: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub non_canonical: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credit_usage: Option<keep_providers::CreditUsage>,
    /// Spend charged to the run's budget ledger, failed calls included.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ledger_cost_usd: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aborted: Option<String>,
    /// Scores in the layout of the upstream harness, for parity protocols.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protocol: Option<ProtocolSummary>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProtocolSummary {
    #[serde(flatten)]
    pub record: crate::protocol::ProtocolRecord,
    pub arms: Vec<crate::protocol::ArmProtocolScore>,
}

fn percentile(sorted: &[u64], p: f64) -> Option<u64> {
    if sorted.is_empty() {
        return None;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    Some(sorted[idx])
}

fn mean(xs: impl Iterator<Item = f64>) -> Option<f64> {
    let (s, n) = xs.fold((0.0, 0usize), |(s, n), x| (s + x, n + 1));
    (n > 0).then(|| s / n as f64)
}

pub fn summarize_arm(arm: &str, records: &[&QuestionRecord]) -> ArmSummary {
    let mut s = ArmSummary {
        arm: arm.into(),
        kind: records
            .first()
            .map(|r| r.arm_kind.clone())
            .unwrap_or_default(),
        diagnostic: records.first().is_some_and(|r| r.diagnostic),
        records: records.len(),
        ..Default::default()
    };
    for r in records {
        *s.counts.entry(r.outcome.key().into()).or_default() += 1;
        match &r.judge {
            JudgeOutcome::Malformed { .. } => s.judge_malformed += 1,
            JudgeOutcome::Failed { error } => {
                s.judge_failed += 1;
                s.budget_refused += usize::from(budget_refusal(error));
            }
            _ => {}
        }
        if let Outcome::Failed { error } = &r.outcome {
            s.budget_refused += usize::from(budget_refusal(error));
        }
    }
    let count = |k: &str| s.counts.get(k).copied().unwrap_or(0);
    let scheduled: Vec<&&QuestionRecord> = records
        .iter()
        .filter(|r| !matches!(r.outcome, Outcome::Excluded { .. }))
        .collect();
    let available: Vec<&&QuestionRecord> = scheduled
        .iter()
        .copied()
        .filter(|r| {
            !matches!(
                r.outcome,
                Outcome::Unavailable { .. } | Outcome::NotRun { .. }
            )
        })
        .collect();
    let first_reason = |want: &str| {
        records.iter().find_map(|r| match &r.outcome {
            Outcome::Unavailable { reason } if want == "unavailable" => Some(reason.clone()),
            Outcome::NotRun { reason } if want == "not_run" => Some(reason.clone()),
            _ => None,
        })
    };
    let estimated = count("estimated");
    (s.status, s.status_reason) = if count("not_run") == scheduled.len() && !scheduled.is_empty() {
        ("not-run".into(), first_reason("not_run"))
    } else if count("unavailable") == scheduled.len() && !scheduled.is_empty() {
        ("unavailable".into(), first_reason("unavailable"))
    } else if estimated > 0 {
        ("estimated".into(), None)
    } else if count("unavailable") > 0 {
        ("partially-unavailable".into(), first_reason("unavailable"))
    } else {
        ("ok".into(), None)
    };
    // An arm that served nothing has no accuracy, not 0%.
    let served_nothing = matches!(s.status.as_str(), "not-run" | "unavailable");
    if estimated == 0 && !served_nothing {
        s.accuracy_scheduled = Rate::new(
            scheduled.len(),
            scheduled.iter().filter(|r| r.correct()).count(),
        );
        s.accuracy_available = Rate::new(
            available.len(),
            available.iter().filter(|r| r.correct()).count(),
        );
        let non_adversarial: Vec<_> = scheduled
            .iter()
            .filter(|r| r.category != "adversarial")
            .collect();
        s.accuracy_non_adversarial = Rate::new(
            non_adversarial.len(),
            non_adversarial.iter().filter(|r| r.correct()).count(),
        );
        s.completion =
            (!available.is_empty()).then(|| count("answered") as f64 / available.len() as f64);
        let mut cats: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        for r in &scheduled {
            let e = cats.entry(r.category.clone()).or_default();
            e.0 += 1;
            e.1 += usize::from(r.correct());
        }
        s.by_category = cats
            .into_iter()
            .map(|(k, (n, c))| (k, Rate::new(n, c)))
            .collect();
    }
    let m: Vec<_> = records.iter().filter_map(|r| r.metrics.as_ref()).collect();
    // Contexts-only records carry evidence recall but no answer.
    let answered: Vec<_> = records
        .iter()
        .filter(|r| r.answer.is_some())
        .filter_map(|r| r.metrics.as_ref())
        .collect();
    s.exact_match = mean(answered.iter().map(|x| f64::from(u8::from(x.exact_match))));
    s.f1 = mean(answered.iter().map(|x| x.f1));
    s.evidence_recall = mean(m.iter().filter_map(|x| x.evidence_recall));
    s.context_tokens_mean = mean(
        records
            .iter()
            .filter_map(|r| r.context.as_ref())
            .map(|c| c.tokens as f64),
    );
    s.prompt_tokens_mean = mean(
        records
            .iter()
            .filter_map(|r| r.prompt.as_ref())
            .map(|p| p.input_tokens as f64),
    );
    s.truncated_records = records
        .iter()
        .filter(|r| {
            r.context
                .as_ref()
                .and_then(|c| c.truncation.as_ref())
                .is_some_and(|t| t.dropped_chunks > 0)
        })
        .count();
    let mut lat = Vec::new();
    for r in records {
        if let Some(a) = &r.answer {
            s.answer_input_tokens += a.usage.input_tokens;
            s.answer_output_tokens += a.usage.output_tokens;
            s.cost_usd += a.cost_usd;
            if let Some(b) = &a.billing {
                s.ai_credits += b.ai_credits;
                s.answer_overhead_tokens += b.harness_overhead_tokens;
            }
            if a.text.is_some() {
                lat.push(a.latency_ms);
            }
        }
        if let Some(jc) = &r.judge_consistency {
            s.judge_cached += usize::from(jc.source == crate::judge::VerdictSource::Cache);
            s.judge_extra_votes += jc.extra_calls.len();
        }
        let extra = r
            .judge_consistency
            .iter()
            .flat_map(|jc| jc.extra_calls.iter());
        for j in r.judge_call.iter().chain(extra) {
            s.judge_input_tokens += j.usage.input_tokens;
            s.judge_output_tokens += j.usage.output_tokens;
            s.cost_usd += j.cost_usd;
            if let Some(b) = &j.billing {
                s.ai_credits += b.ai_credits;
                s.judge_overhead_tokens += b.harness_overhead_tokens;
            }
        }
    }
    s.tokens = token_accounting(records);
    lat.sort_unstable();
    s.answer_latency_ms_p50 = percentile(&lat, 0.5);
    s.answer_latency_ms_p95 = percentile(&lat, 0.95);
    s
}

pub fn summarize(meta: &RunMeta, manifest: &Manifest, records: &[QuestionRecord]) -> Summary {
    let arms: Vec<ArmSummary> = manifest
        .arms
        .iter()
        .map(|a| {
            let rs: Vec<&QuestionRecord> = records.iter().filter(|r| r.arm == a.name).collect();
            let mut s = summarize_arm(&a.name, &rs);
            s.kind = a.kind.label().into();
            s.diagnostic = a.kind.is_diagnostic();
            if rs.is_empty() {
                s.status = "unavailable".into();
                s.status_reason = Some("no records were produced".into());
            }
            s
        })
        .collect();
    Summary {
        schema: "keep-summary-v1".into(),
        run_id: meta.run_id.clone(),
        mode: meta.mode,
        mock: meta.mock,
        manifest_id: meta.manifest_id.clone(),
        manifest_hash: meta.manifest_hash.clone(),
        manifest_file_hash: meta.manifest_file_hash.clone(),
        phase: format!("{:?}", meta.phase).to_lowercase(),
        dataset: meta.dataset.clone(),
        dataset_hash: meta.dataset_hash.clone(),
        gold_track: meta.gold_track.clone(),
        storage_profile: meta.storage_profile.as_str().into(),
        questions: meta.questions_run,
        histories: meta.histories_run,
        seeds: meta.seeds.clone(),
        metrics_version: METRICS_VERSION.into(),
        tokenizer: meta.tokenizer.clone(),
        tokenizer_exact: meta.tokenizer_exact,
        fit_fraction: meta.fit_mask.as_ref().map(|f| f.fit_fraction),
        total_cost_usd: arms.iter().map(|a| a.cost_usd).sum(),
        arms,
        embedding_input_tokens: meta.embedding_usage.input_tokens,
        warnings: meta.warnings.clone(),
        non_canonical: meta.non_canonical.clone(),
        credit_usage: meta.credit_usage,
        ledger_cost_usd: meta.budget_usage.map(|b| b.cost_usd),
        aborted: meta.aborted.clone(),
        protocol: meta.protocol.clone().map(|record| ProtocolSummary {
            arms: if meta.mode == Mode::Execute {
                manifest
                    .arms
                    .iter()
                    .map(|a| {
                        let rs: Vec<&QuestionRecord> =
                            records.iter().filter(|r| r.arm == a.name).collect();
                        crate::protocol::score_arm(record.name, &a.name, &rs)
                    })
                    .collect()
            } else {
                Vec::new()
            },
            record,
        }),
    }
}

fn protocol_section(p: &ProtocolSummary) -> String {
    use crate::protocol::ProtocolName;
    let r = &p.record;
    let mut out = format!(
        "\n## Protocol: {}\n\nPrompts from {} at commit `{}` ({}); answer prompt `{}`, judge prompt `{}`. Split: {}.\n\n",
        r.name.as_str(),
        r.upstream_repository,
        r.upstream_commit,
        r.upstream_license,
        r.answer_prompt_version,
        r.judge_prompt_version,
        match r.split {
            crate::protocol::SplitScope::Dev =>
                "development histories only (not comparable to full-set published numbers)",
            crate::protocol::SplitScope::Full => "full set, holdout included",
        },
    );
    for f in &r.prompts {
        out.push_str(&format!("- `{}` `{}`\n", f.file, f.sha256));
    }
    if p.arms.is_empty() {
        return out;
    }
    let groups: Vec<&String> = {
        let mut g: Vec<&String> = Vec::new();
        for a in &p.arms {
            for k in a.by_group.keys() {
                if !g.contains(&k) {
                    g.push(k);
                }
            }
        }
        if r.name == ProtocolName::LongmemevalOfficial {
            let order = crate::protocol::LONGMEMEVAL_TYPES;
            g.sort_by_key(|k| order.iter().position(|t| t == k).unwrap_or(order.len()));
        }
        g
    };
    let lme = r.name == ProtocolName::LongmemevalOfficial;
    let score_name = if lme {
        "Overall accuracy"
    } else {
        "J (overall)"
    };
    out.push_str(&format!("\n| Arm | {score_name} | n |"));
    for g in &groups {
        out.push_str(&format!(" {g} |"));
    }
    if lme {
        out.push_str(" Task-averaged | Abstention |");
    }
    out.push_str(" Unscored |\n|---|---|---|");
    for _ in &groups {
        out.push_str("---|");
    }
    if lme {
        out.push_str("---|---|");
    }
    out.push_str("---|\n");
    let cell = |x: &crate::protocol::Score| format!("{} ({})", pct(x.accuracy), x.n);
    for a in &p.arms {
        out.push_str(&format!(
            "| {} | {} | {} |",
            a.arm,
            pct(a.overall.accuracy),
            a.overall.n
        ));
        for g in &groups {
            out.push_str(&format!(
                " {} |",
                a.by_group.get(*g).map(cell).unwrap_or_else(|| "—".into())
            ));
        }
        if lme {
            out.push_str(&format!(
                " {} | {} |",
                pct(a.task_averaged),
                a.abstention
                    .as_ref()
                    .map(cell)
                    .unwrap_or_else(|| "—".into())
            ));
        }
        out.push_str(&format!(" {} |\n", a.unscored));
    }
    out.push_str(
        "\nUnscored questions (failed, unavailable or an unreadable verdict) count as wrong, as upstream counts them.\n",
    );
    out
}

fn pct(x: Option<f64>) -> String {
    x.map(|v| format!("{:.1}%", v * 100.0))
        .unwrap_or_else(|| "—".into())
}

fn num(x: Option<f64>) -> String {
    x.map(|v| format!("{v:.0}")).unwrap_or_else(|| "—".into())
}

/// Per-arm harness accounting: Keep's own prompt tokens beside the hidden
/// tokens the harness added, and the credits each arm cost.
fn harness_table(s: &Summary) -> String {
    let mut out = String::from(
        "\n## Harness accounting\n\n\
         Keep prompt tokens are what Keep sent; hidden overhead is what the harness added \
         to each call on top (its own system prompt and tool definitions).\n\n\
         | Arm | Answered | Judged correct | Judge acc. (answered) | F1 | Keep prompt tokens (mean) \
         | Hidden overhead per answer call | Hidden overhead per judge call | AI credits | Failed | Budget refused |\n\
         |---|---|---|---|---|---|---|---|---|---|---|\n",
    );
    for a in &s.arms {
        let answered = a.counts.get("answered").copied().unwrap_or(0);
        let judged = a.accuracy_available.correct;
        let per = |total: u64| {
            if answered > 0 {
                format!("{:.0}", total as f64 / answered as f64)
            } else {
                "—".into()
            }
        };
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {:.2} | {} | {} |\n",
            a.arm,
            answered,
            judged,
            pct((answered > 0).then(|| judged as f64 / answered as f64)),
            pct(a.f1),
            num(a.prompt_tokens_mean),
            per(a.answer_overhead_tokens),
            per(a.judge_overhead_tokens),
            a.ai_credits,
            a.counts.get("failed").copied().unwrap_or(0),
            a.budget_refused,
        ));
    }
    if let Some(u) = &s.credit_usage {
        out.push_str(&format!(
            "\nAI credits: {:.2} reported + {:.2} held for calls that failed without a reported charge, \
             cap {:.2}; {} calls made, {} refused at the cap.\n",
            u.reported, u.unsettled, u.cap, u.calls, u.refused
        ));
    }
    for a in s
        .arms
        .iter()
        .filter(|a| a.judge_cached + a.judge_extra_votes > 0)
    {
        out.push_str(&format!(
            "\n{}: {} verdicts reused from identical answers; {} extra judge votes.",
            a.arm, a.judge_cached, a.judge_extra_votes
        ));
    }
    out
}

pub fn markdown(s: &Summary) -> String {
    let mut out = String::new();
    out.push_str(&format!("# Run {}\n\n", s.run_id));
    if let Some(label) = &s.non_canonical {
        out.push_str(&format!("> **{label}**\n\n"));
    }
    if let Some(reason) = &s.aborted {
        out.push_str(&format!(
            "> **Aborted: {reason}.** This run is incomplete and cannot be compared.\n\n"
        ));
    }
    if let Some(cost) = s.ledger_cost_usd {
        out.push_str(&format!(
            "Spend charged to this run's budget: ${cost:.4} (failed calls included).\n\n"
        ));
    }
    if s.mock {
        out.push_str(
            "> **Mock providers.** These numbers come from offline stand-ins and only show that the \
             pipeline works. They are not benchmark results.\n\n",
        );
    }
    if s.mode == Mode::Estimate {
        out.push_str(
            "> **Dry run.** No model was called; only contexts and token counts were produced.\n\n",
        );
    }
    if s.mode == Mode::Contexts {
        out.push_str(
            "> **Contexts only.** No answer or judge model was called; contexts were built with the \
             manifest's embedder and storage, and only evidence recall and token counts are scored.\n\n",
        );
    }
    out.push_str(&format!(
        "| | |\n|---|---|\n| Manifest | `{}` |\n| Manifest hash | `{}` |\n| Manifest file hash | `{}` |\n\
         | Phase | {} |\n| Dataset | {} `{}` |\n| Gold track | {} |\n| Storage profile | {} |\n\
         | Questions / histories | {} / {} |\n| Seeds | {:?} |\n| Tokenizer | {}{} |\n| Local metrics | {} |\n\
         | Full-context fit fraction | {} |\n| Total cost (USD) | {:.4} |\n\n",
        s.manifest_id,
        s.manifest_hash,
        s.manifest_file_hash,
        s.phase,
        s.dataset,
        s.dataset_hash,
        s.gold_track,
        s.storage_profile,
        s.questions,
        s.histories,
        s.seeds,
        s.tokenizer,
        if s.tokenizer_exact { "" } else { " (estimate)" },
        s.metrics_version,
        s.fit_fraction.map(|f| format!("{:.1}%", f * 100.0)).unwrap_or_else(|| "no frozen fit mask".into()),
        s.total_cost_usd,
    ));
    out.push_str(
        "| Arm | Kind | Status | Acc. (scheduled) | Acc. (available) | Completion | EM | F1 | Evidence recall \
         | Ctx tokens | Truncated | Failed | Unavailable | Judge malformed | Cost USD | p50 ms | p95 ms |\n",
    );
    out.push_str("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    for a in &s.arms {
        let c = |k: &str| a.counts.get(k).copied().unwrap_or(0);
        out.push_str(&format!(
            "| {}{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {:.4} | {} | {} |\n",
            a.arm,
            if a.diagnostic { " (diagnostic)" } else { "" },
            a.kind,
            a.status,
            pct(a.accuracy_scheduled.accuracy),
            pct(a.accuracy_available.accuracy),
            pct(a.completion),
            pct(a.exact_match),
            pct(a.f1),
            pct(a.evidence_recall),
            num(a.context_tokens_mean),
            a.truncated_records,
            c("failed"),
            c("unavailable"),
            a.judge_malformed,
            a.cost_usd,
            a.answer_latency_ms_p50.map(|v| v.to_string()).unwrap_or_else(|| "—".into()),
            a.answer_latency_ms_p95.map(|v| v.to_string()).unwrap_or_else(|| "—".into()),
        ));
    }
    if let Some(p) = &s.protocol {
        out.push_str(&protocol_section(p));
    }
    if s.credit_usage.is_some() || s.arms.iter().any(|a| a.ai_credits > 0.0) {
        out.push_str(&harness_table(s));
    }
    let reasons: Vec<&ArmSummary> = s
        .arms
        .iter()
        .filter(|a| a.status_reason.is_some())
        .collect();
    if !reasons.is_empty() {
        out.push_str("\n## Unavailable or not run\n\n");
        for a in reasons {
            out.push_str(&format!(
                "- **{}** ({}): {}\n",
                a.arm,
                a.status,
                a.status_reason.as_deref().unwrap_or("")
            ));
        }
    }
    let keep: Vec<(&ArmSummary, &TokenAccounting)> = s
        .arms
        .iter()
        .filter_map(|a| a.tokens.as_ref().map(|t| (a, t)))
        .collect();
    if !keep.is_empty() {
        out.push_str(
            "\n## Keep context tokens\n\nStanding blocks share `standing_tokens`; evidence blocks share \
             `evidence_tokens`. Means and maxima are per question.\n\n\
             | Arm | Acc. (scheduled) | F1 | Standing mean | Standing max | Evidence mean | Evidence max | Blocks (mean) |\n\
             |---|---|---|---|---|---|---|---|\n",
        );
        for (a, t) in keep {
            let blocks: Vec<String> = t
                .blocks_mean
                .iter()
                .map(|(k, v)| format!("{k} {v:.0}"))
                .collect();
            out.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} |\n",
                a.arm,
                pct(a.accuracy_scheduled.accuracy),
                pct(a.f1),
                num(t.standing_mean),
                num(t.standing_max.map(|x| x as f64)),
                num(t.evidence_mean),
                num(t.evidence_max.map(|x| x as f64)),
                blocks.join(", "),
            ));
        }
    }
    let cats: std::collections::BTreeSet<&String> =
        s.arms.iter().flat_map(|a| a.by_category.keys()).collect();
    if !cats.is_empty() {
        if s.dataset == "locomo" {
            out.push_str("\n## Non-adversarial accuracy (scheduled)\n\n| Arm | Accuracy | Questions |\n|---|---:|---:|\n");
            for a in &s.arms {
                out.push_str(&format!(
                    "| {} | {} | {} |\n",
                    a.arm,
                    pct(a.accuracy_non_adversarial.accuracy),
                    a.accuracy_non_adversarial.n
                ));
            }
            out.push_str("\nExcludes only the adversarial category; failures remain in the denominator. This is not a published-protocol score.\n");
        }
        out.push_str("\n## Accuracy by category (scheduled)\n\n| Arm |");
        for c in &cats {
            out.push_str(&format!(" {c} |"));
        }
        out.push_str("\n|---|");
        out.push_str(&"---|".repeat(cats.len()));
        out.push('\n');
        for a in &s.arms {
            out.push_str(&format!("| {} |", a.arm));
            for c in &cats {
                let r = a.by_category.get(*c);
                out.push_str(&format!(
                    " {} |",
                    r.map(|r| format!("{} (n={})", pct(r.accuracy), r.n))
                        .unwrap_or_else(|| "—".into())
                ));
            }
            out.push('\n');
        }
    }
    if !s.warnings.is_empty() {
        out.push_str("\n## Warnings\n\n");
        for w in &s.warnings {
            out.push_str(&format!("- {w}\n"));
        }
    }
    out
}
