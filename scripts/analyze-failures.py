#!/usr/bin/env python3
"""Review completed benchmark records, never read or rewrite dataset labels.

All-arms-wrong is a review queue, not proof of a bad label. Incomplete,
failed, duplicate or unjudged comparisons cannot enter that queue.
"""
import argparse
import collections
import json


def analyze(records, subset=None):
    groups = collections.defaultdict(list)
    arms = set()
    for r in records:
        if subset is not None and r["question_id"] not in subset:
            continue
        messages = (r.get("prompt") or {}).get("messages") or []
        message = messages[-1].get("content", "") if messages else ""
        question = (message.rsplit("\nQuestion:", 1)[-1].split("\nAnswer:", 1)[0].strip()
                    if "\nQuestion:" in message else None)
        # answer-v6 replies carry working notes; review the scored answer.
        final = r.get("final_answer")
        r = {k: r.get(k) for k in (
            "question_id", "seed", "arm", "category", "gold", "outcome",
            "judge", "answer", "metrics")}
        r["review_question"] = question
        if final is not None:
            r["answer"] = {**(r["answer"] or {}), "text": final}
        arms.add(r["arm"])
        groups[(r["question_id"], r["seed"])].append(r)
    suspects = []
    incomplete = 0
    incomplete_failures = []
    disagreements = []
    category = collections.Counter()
    failures = collections.defaultdict(collections.Counter)
    for (qid, seed), rows in sorted(groups.items()):
        by_arm = {r["arm"]: r for r in rows}
        complete = (len(rows) == len(by_arm) == len(arms)
                    and all(r.get("outcome", {}).get("state") == "answered"
                            and r.get("judge", {}).get("state") == "judged"
                            and isinstance(r["judge"].get("correct"), bool)
                            for r in rows))
        if not complete:
            incomplete += 1
            judged = [r for r in rows if isinstance(r.get("judge", {}).get("correct"), bool)]
            if judged and all(not r["judge"]["correct"] for r in judged):
                incomplete_failures.append({
                    "question_id": qid, "seed": seed,
                    "unjudged_or_missing": sorted(arms - {r["arm"] for r in judged}),
                })
        identical = collections.defaultdict(list)
        for r in rows:
            answer = (r.get("answer") or {}).get("text", "").strip()
            correct = r.get("judge", {}).get("correct")
            if answer and isinstance(correct, bool):
                identical[" ".join(answer.casefold().split())].append(
                    {"arm": r["arm"], "correct": correct})
        for answer, votes in identical.items():
            if len({vote["correct"] for vote in votes}) > 1:
                disagreements.append({
                    "question_id": qid, "seed": seed, "answer": answer, "verdicts": votes,
                })
        for r in rows:
            if r.get("judge", {}).get("correct") is not False:
                continue
            f = failures[r["arm"]]
            f["wrong"] += 1
            f[r["category"]] += 1
            m = r.get("metrics") or {}
            f["abstained"] += bool(m.get("abstained"))
            f["claimed_full_evidence_recall"] += m.get("evidence_recall") == 1
            f["partial_evidence_recall"] += 0 < (m.get("evidence_recall") or 0) < 1
            f["zero_evidence_recall"] += m.get("evidence_recall") == 0
            text = (r.get("answer") or {}).get("text", "")
            f["answer_over_40_words"] += len(text.split()) > 40
            f["exact_match_but_judge_wrong"] += bool(m.get("exact_match"))
        if complete and all(not r["judge"]["correct"] for r in rows):
            first = rows[0]
            category[first["category"]] += 1
            suspects.append({
                "question_id": qid, "seed": seed, "category": first["category"],
                "question": first["review_question"], "gold": first["gold"],
                "answers": {a: r["answer"]["text"] for a, r in sorted(by_arm.items())},
            })
    return {
        "warning": "Review candidates only, not verified bad labels. Do not tune against this queue.",
        "arms": sorted(arms), "question_seed_pairs": len(groups),
        "incomplete_pairs_excluded": incomplete,
        "incomplete_all_judged_arms_wrong": incomplete_failures,
        "identical_answer_judge_disagreements": disagreements,
        "all_arms_wrong_count": len(suspects), "all_arms_wrong_by_category": dict(category),
        "failure_signals_by_arm": {a: dict(v) for a, v in sorted(failures.items())},
        "all_arms_wrong": suspects,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", help="One run's records.jsonl (read only)")
    parser.add_argument("--subset", help="Optional existing development question-id file")
    args = parser.parse_args()
    subset = None
    if args.subset:
        with open(args.subset) as f:
            subset = {line.strip() for line in f if line.strip() and not line.startswith("#")}
    with open(args.records) as f:
        result = analyze((json.loads(line) for line in f if line.strip()), subset)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
