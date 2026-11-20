//! Judge calibration.
//!
//! An LLM judge can be lenient: it may accept an on-topic but wrong answer,
//! a date that is off by a year, a hedge, or an answer that contradicts the
//! gold. Before its verdicts are trusted, the judge is scored against items
//! whose correct verdict is known: synthetic probes built from development
//! questions, and human-labelled items supplied as JSONL. The report gives
//! agreement, Cohen's kappa and the false-accept rate per probe type.

use std::collections::BTreeMap;

use keep_providers::{ApproxTokenizer, ChatModel, ChatRequest, Tokenizer};
use serde::{Deserialize, Serialize};

use crate::dataset::Dataset;
use crate::metrics;
use crate::prompt;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationItem {
    pub id: String,
    pub question: String,
    pub gold: String,
    pub candidate: String,
    #[serde(default)]
    pub abstention: bool,
    /// The verdict a careful grader should give.
    pub expected_correct: bool,
    /// Probe family, e.g. `correct`, `wrong-on-topic`, `date-shift`,
    /// `abstain-on-answerable`, `contradiction`, `distractor`, or `human`.
    pub probe: String,
}

/// Shift the first four-digit year in `text` by one. `None` if there is no
/// year to shift.
fn shift_year(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    for i in 0..bytes.len().saturating_sub(3) {
        let boundary_before = i == 0 || !bytes[i - 1].is_ascii_digit();
        let boundary_after = i + 4 == bytes.len() || !bytes[i + 4].is_ascii_digit();
        // ASCII digits are always char boundaries, so slicing is safe.
        if boundary_before && boundary_after && bytes[i..i + 4].iter().all(u8::is_ascii_digit) {
            let year: u32 = text[i..i + 4].parse().ok()?;
            if (1900..=2100).contains(&year) {
                return Some(format!("{}{}{}", &text[..i], year + 1, &text[i + 4..]));
            }
        }
    }
    None
}

/// Build probes from a dataset's questions (call this on the development
/// split only). At most `per_type` probes of each type are produced.
pub fn probes(dataset: &Dataset, per_type: usize) -> Vec<CalibrationItem> {
    let mut out: Vec<CalibrationItem> = Vec::new();
    let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut push = |out: &mut Vec<CalibrationItem>, probe: &'static str, item: CalibrationItem| {
        let n = counts.entry(probe).or_default();
        if *n < per_type {
            *n += 1;
            out.push(item);
        }
    };
    for q in &dataset.questions {
        let mk = |probe: &str, candidate: String, expected: bool| CalibrationItem {
            id: format!("{}#{probe}", q.id),
            question: q.query.text.clone(),
            gold: q.answer.clone(),
            candidate,
            abstention: q.abstention,
            expected_correct: expected,
            probe: probe.into(),
        };
        if q.abstention {
            push(
                &mut out,
                "correct-abstention",
                mk("correct-abstention", "I don't know.".into(), true),
            );
            if let Some(d) = &q.distractor {
                push(&mut out, "distractor", mk("distractor", d.clone(), false));
            }
            continue;
        }
        push(&mut out, "correct", mk("correct", q.answer.clone(), true));
        push(
            &mut out,
            "abstain-on-answerable",
            mk("abstain-on-answerable", "I don't know.".into(), false),
        );
        push(
            &mut out,
            "contradiction",
            mk(
                "contradiction",
                format!("No, it was not {}.", q.answer),
                false,
            ),
        );
        if let Some(shifted) = shift_year(&q.answer) {
            push(&mut out, "date-shift", mk("date-shift", shifted, false));
        }
        // Another answer from the same history: on topic, but wrong.
        if let Some(other) = dataset.questions.iter().find(|o| {
            o.history_id == q.history_id
                && o.id != q.id
                && !o.abstention
                && metrics::f1(&o.answer, &q.answer) < 0.2
                && !metrics::normalize(&o.answer).is_empty()
        }) {
            push(
                &mut out,
                "wrong-on-topic",
                mk("wrong-on-topic", other.answer.clone(), false),
            );
        }
    }
    out
}

pub fn read_items(text: &str) -> anyhow::Result<Vec<CalibrationItem>> {
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| serde_json::from_str(l).map_err(|e| anyhow::anyhow!("line {}: {e}", i + 1)))
        .collect()
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ProbeStats {
    pub n: usize,
    pub agree: usize,
    pub malformed: usize,
    pub failed: usize,
    /// Expected incorrect, judged correct.
    pub false_accepts: usize,
    /// Expected correct, judged incorrect.
    pub false_rejects: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationReport {
    pub judge: keep_providers::ProviderDescriptor,
    pub prompt_version: String,
    pub items: usize,
    pub agreement: f64,
    pub cohen_kappa: Option<f64>,
    pub false_accept_rate: Option<f64>,
    pub false_reject_rate: Option<f64>,
    pub by_probe: BTreeMap<String, ProbeStats>,
    pub mock: bool,
}

/// Cohen's kappa for two binary raters given as (expected, judged) pairs.
pub fn cohen_kappa(pairs: &[(bool, bool)]) -> Option<f64> {
    if pairs.is_empty() {
        return None;
    }
    let n = pairs.len() as f64;
    let po = pairs.iter().filter(|(a, b)| a == b).count() as f64 / n;
    let pa = pairs.iter().filter(|(a, _)| *a).count() as f64 / n;
    let pb = pairs.iter().filter(|(_, b)| *b).count() as f64 / n;
    let pe = pa * pb + (1.0 - pa) * (1.0 - pb);
    if (1.0 - pe).abs() < 1e-12 {
        return None;
    }
    Some((po - pe) / (1.0 - pe))
}

pub async fn calibrate(
    judge: &dyn ChatModel,
    items: &[CalibrationItem],
    max_output_tokens: u32,
) -> CalibrationReport {
    let tok = ApproxTokenizer;
    let mut by_probe: BTreeMap<String, ProbeStats> = BTreeMap::new();
    let mut pairs = Vec::new();
    let (mut fa, mut neg, mut fr, mut pos) = (0, 0, 0, 0);
    for item in items {
        let messages =
            prompt::judge_messages(&item.question, &item.gold, &item.candidate, item.abstention);
        let req = ChatRequest {
            estimated_input_tokens: messages.iter().map(|m| tok.count(&m.content) as u64).sum(),
            messages,
            max_output_tokens,
            temperature: 0.0,
            seed: Some(0),
        };
        let s = by_probe.entry(item.probe.clone()).or_default();
        s.n += 1;
        // Unparseable or failed verdicts count as disagreement and as
        // whatever error that implies, so they cannot flatter the judge.
        let verdict = match judge.complete(&req).await {
            Ok(r) => match prompt::parse_verdict(&r.text) {
                Some(v) => Some(v),
                None => {
                    s.malformed += 1;
                    None
                }
            },
            Err(_) => {
                s.failed += 1;
                None
            }
        };
        let judged = verdict.unwrap_or(!item.expected_correct);
        if judged == item.expected_correct {
            s.agree += 1;
        }
        if item.expected_correct {
            pos += 1;
            if !judged {
                fr += 1;
                s.false_rejects += 1;
            }
        } else {
            neg += 1;
            if judged {
                fa += 1;
                s.false_accepts += 1;
            }
        }
        pairs.push((item.expected_correct, judged));
    }
    let agree = pairs.iter().filter(|(a, b)| a == b).count();
    CalibrationReport {
        judge: judge.descriptor().clone(),
        prompt_version: prompt::JUDGE_PROMPT_VERSION.into(),
        items: items.len(),
        agreement: if items.is_empty() {
            0.0
        } else {
            agree as f64 / items.len() as f64
        },
        cohen_kappa: cohen_kappa(&pairs),
        false_accept_rate: (neg > 0).then(|| fa as f64 / neg as f64),
        false_reject_rate: (pos > 0).then(|| fr as f64 / pos as f64),
        by_probe,
        mock: judge.descriptor().mock,
    }
}

pub fn markdown(r: &CalibrationReport) -> String {
    let mut out = format!(
        "# Judge calibration\n\nJudge {} `{}` ({}), prompt {}. {} items.\n\n",
        r.judge.provider, r.judge.model, r.judge.revision, r.prompt_version, r.items
    );
    if r.mock {
        out.push_str("> **Mock judge.** This checks the calibration pipeline only.\n\n");
    }
    out.push_str(&format!(
        "Agreement {:.1}%, Cohen's kappa {}, false-accept rate {}, false-reject rate {}.\n\n",
        r.agreement * 100.0,
        r.cohen_kappa
            .map(|k| format!("{k:.3}"))
            .unwrap_or_else(|| "n/a".into()),
        r.false_accept_rate
            .map(|k| format!("{:.1}%", k * 100.0))
            .unwrap_or_else(|| "n/a".into()),
        r.false_reject_rate
            .map(|k| format!("{:.1}%", k * 100.0))
            .unwrap_or_else(|| "n/a".into()),
    ));
    out.push_str("| Probe | n | Agree | False accepts | False rejects | Malformed | Failed |\n|---|---|---|---|---|---|---|\n");
    for (k, s) in &r.by_probe {
        out.push_str(&format!(
            "| {k} | {} | {} | {} | {} | {} | {} |\n",
            s.n, s.agree, s.false_accepts, s.false_rejects, s.malformed, s.failed
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use keep_providers::mock::FnChat;

    #[test]
    fn year_shift_only_touches_plausible_years() {
        assert_eq!(shift_year("7 May 2023").as_deref(), Some("7 May 2024"));
        assert_eq!(shift_year("12345"), None);
        assert_eq!(shift_year("room 0042"), None);
        assert_eq!(shift_year("no year"), None);
    }

    #[test]
    fn kappa_matches_hand_computation() {
        // 10 items: 4 TT, 1 TF, 1 FT, 4 FF -> po=0.8, pe=0.5, kappa=0.6.
        let mut v = vec![(true, true); 4];
        v.push((true, false));
        v.push((false, true));
        v.extend(vec![(false, false); 4]);
        assert!((cohen_kappa(&v).unwrap() - 0.6).abs() < 1e-12);
        assert_eq!(cohen_kappa(&[(true, true)]), None);
    }

    #[tokio::test]
    async fn an_always_correct_judge_shows_up_as_false_accepts() {
        let item = |probe: &str, expected: bool| CalibrationItem {
            id: probe.into(),
            question: "q".into(),
            gold: "g".into(),
            candidate: "c".into(),
            abstention: false,
            expected_correct: expected,
            probe: probe.into(),
        };
        let items = vec![
            item("correct", true),
            item("date-shift", false),
            item("contradiction", false),
        ];
        let lenient = FnChat::new("lenient", |_| Ok("CORRECT".into()));
        let r = calibrate(&lenient, &items, 8).await;
        assert_eq!(r.false_accept_rate, Some(1.0));
        assert_eq!(r.by_probe["date-shift"].false_accepts, 1);
        assert!((r.agreement - 1.0 / 3.0).abs() < 1e-12);

        let garbled = FnChat::new("garbled", |_| Ok("maybe".into()));
        let r = calibrate(&garbled, &items, 8).await;
        assert_eq!(r.agreement, 0.0);
        assert_eq!(r.by_probe["correct"].malformed, 1);
    }
}
