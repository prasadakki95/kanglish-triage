"""Evaluation harness for Kanglish Review & Support Triage.

Usage:
    python eval/evaluate.py --gold data/seed_examples.jsonl --preds eval/sample_preds.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from schema import (
    CATEGORIES,
    SENTIMENTS,
    URGENCIES,
    extract_json_object,
    validate_prediction,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_num}: {exc}") from exc
    return records


def compute_macro_f1(y_true: list[str], y_pred: list[str | None], labels: tuple[str, ...]) -> float:
    """Compute macro-averaged F1 across known label classes without external deps."""
    f1_scores: list[float] = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        # Only average over classes that actually appear in ground truth or predictions
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def evaluate_predictions(
    gold_records: list[dict[str, Any]],
    pred_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score predictions against gold records and return metrics + error examples."""
    preds_by_id = {rec["id"]: rec for rec in pred_records}

    total = len(gold_records)
    if total == 0:
        raise ValueError("Gold dataset is empty.")

    strict_json_count = 0
    valid_schema_count = 0
    all_labels_match_count = 0

    y_true_sent: list[str] = []
    y_pred_sent: list[str | None] = []
    y_true_cat: list[str] = []
    y_pred_cat: list[str | None] = []
    y_true_urg: list[str] = []
    y_pred_urg: list[str | None] = []

    error_counts: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []

    for gold in gold_records:
        ex_id = gold["id"]
        gold_label = gold["label"]
        pred_entry = preds_by_id.get(ex_id)

        if pred_entry is None:
            parsed, is_strict = None, False
            error_counts["missing_prediction"] += 1
            errors = ["missing_prediction"]
        elif "raw_output" in pred_entry and isinstance(pred_entry["raw_output"], str):
            parsed, is_strict = extract_json_object(pred_entry["raw_output"])
            is_valid, errors = validate_prediction(parsed)
            if not is_valid:
                for err in errors:
                    error_counts[err] += 1
        elif "prediction" in pred_entry and isinstance(pred_entry["prediction"], dict):
            parsed, is_strict = pred_entry["prediction"], True
            is_valid, errors = validate_prediction(parsed)
            if not is_valid:
                for err in errors:
                    error_counts[err] += 1
        else:
            parsed, is_strict = None, False
            errors = ["invalid_prediction_record"]
            error_counts["invalid_prediction_record"] += 1

        is_valid = len(errors) == 0
        if is_strict and is_valid:
            strict_json_count += 1
        if is_valid:
            valid_schema_count += 1

        p_sent = parsed.get("sentiment") if (parsed and is_valid) else None
        p_cat = parsed.get("category") if (parsed and is_valid) else None
        p_urg = parsed.get("urgency") if (parsed and is_valid) else None

        y_true_sent.append(gold_label["sentiment"])
        y_pred_sent.append(p_sent)
        y_true_cat.append(gold_label["category"])
        y_pred_cat.append(p_cat)
        y_true_urg.append(gold_label["urgency"])
        y_pred_urg.append(p_urg)

        exact_match = (
            is_valid
            and p_sent == gold_label["sentiment"]
            and p_cat == gold_label["category"]
            and p_urg == gold_label["urgency"]
        )
        if exact_match:
            all_labels_match_count += 1
        else:
            failures.append(
                {
                    "id": ex_id,
                    "text": gold["text"],
                    "gold": gold_label,
                    "pred": parsed,
                    "errors": errors,
                }
            )

    sent_acc = sum(t == p for t, p in zip(y_true_sent, y_pred_sent)) / total
    cat_acc = sum(t == p for t, p in zip(y_true_cat, y_pred_cat)) / total
    urg_acc = sum(t == p for t, p in zip(y_true_urg, y_pred_urg)) / total

    metrics = {
        "total_examples": total,
        "valid_schema_rate": round(valid_schema_count / total, 4),
        "strict_json_rate": round(strict_json_count / total, 4),
        "all_labels_exact_match": round(all_labels_match_count / total, 4),
        "sentiment_accuracy": round(sent_acc, 4),
        "sentiment_macro_f1": round(compute_macro_f1(y_true_sent, y_pred_sent, SENTIMENTS), 4),
        "category_accuracy": round(cat_acc, 4),
        "category_macro_f1": round(compute_macro_f1(y_true_cat, y_pred_cat, CATEGORIES), 4),
        "urgency_accuracy": round(urg_acc, 4),
        "urgency_macro_f1": round(compute_macro_f1(y_true_urg, y_pred_urg, URGENCIES), 4),
        "schema_error_counts": dict(error_counts),
    }
    return {"metrics": metrics, "failures": failures}


def print_report(report: dict[str, Any], max_failures: int = 5) -> None:
    m = report["metrics"]
    print("\n=== Kanglish Triage Evaluation Report ===")
    print(f"Total examples         : {m['total_examples']}")
    print(f"Valid schema rate      : {m['valid_schema_rate']:.1%}")
    print(f"Strict JSON rate       : {m['strict_json_rate']:.1%}")
    print(f"All-labels exact match : {m['all_labels_exact_match']:.1%}")
    print("-" * 41)
    print(f"Sentiment  Acc / F1    : {m['sentiment_accuracy']:.1%} / {m['sentiment_macro_f1']:.3f}")
    print(f"Category   Acc / F1    : {m['category_accuracy']:.1%} / {m['category_macro_f1']:.3f}")
    print(f"Urgency    Acc / F1    : {m['urgency_accuracy']:.1%} / {m['urgency_macro_f1']:.3f}")

    if m["schema_error_counts"]:
        print("-" * 41)
        print("Schema errors:", m["schema_error_counts"])

    failures = report["failures"]
    if failures and max_failures > 0:
        print(f"\n--- Sample Failures (showing {min(len(failures), max_failures)} of {len(failures)}) ---")
        for item in failures[:max_failures]:
            print(f"[{item['id']}] {item['text']}")
            print(f"  Gold  : {item['gold']}")
            print(f"  Pred  : {item['pred']}")
            if item["errors"]:
                print(f"  Errors: {item['errors']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Kanglish triage predictions.")
    parser.add_argument("--gold", type=Path, required=True, help="Path to gold JSONL file.")
    parser.add_argument("--preds", type=Path, required=True, help="Path to predictions JSONL file.")
    parser.add_argument("--out", type=Path, default=None, help="Optional path to save JSON report.")
    parser.add_argument("--show-failures", type=int, default=5, help="Number of failure examples to print.")
    args = parser.parse_args()

    gold_records = load_jsonl(args.gold)
    pred_records = load_jsonl(args.preds)
    report = evaluate_predictions(gold_records, pred_records)
    print_report(report, max_failures=args.show_failures)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved detailed report to {args.out}")


if __name__ == "__main__":
    main()
