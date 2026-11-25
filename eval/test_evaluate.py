"""Unit tests for the Kanglish triage schema and evaluation harness."""

import json
import sys
import unittest
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from evaluate import evaluate_predictions, load_jsonl  # noqa: E402
from schema import extract_json_object, validate_prediction  # noqa: E402


class TestKanglishEvalHarness(unittest.TestCase):
    def test_seed_dataset_validates(self) -> None:
        seed_path = EVAL_DIR.parent / "data" / "seed_examples.jsonl"
        records = load_jsonl(seed_path)
        self.assertGreaterEqual(len(records), 20)
        for rec in records:
            ok, errs = validate_prediction(rec["label"])
            self.assertTrue(ok, f"Seed {rec['id']} failed validation: {errs}")

    def test_extract_fenced_and_strict_json(self) -> None:
        raw_strict = '{"sentiment":"positive","category":"food_quality","urgency":"low","english_summary":"Great food."}'
        parsed, strict = extract_json_object(raw_strict)
        self.assertTrue(strict)
        self.assertIsNotNone(parsed)

        raw_fenced = f"Sure! Here is the JSON:\n```json\n{raw_strict}\n```"
        parsed_f, strict_f = extract_json_object(raw_fenced)
        self.assertFalse(strict_f)
        self.assertEqual(parsed_f, parsed)

    def test_evaluate_predictions_perfect_and_broken(self) -> None:
        gold = [
            {
                "id": "1",
                "text": "Dosa sakath agide",
                "label": {
                    "sentiment": "positive",
                    "category": "food_quality",
                    "urgency": "low",
                    "english_summary": "Dosa was great.",
                },
            },
            {
                "id": "2",
                "text": "Delivery tumba late aytu",
                "label": {
                    "sentiment": "negative",
                    "category": "delivery_service",
                    "urgency": "high",
                    "english_summary": "Delivery was very late.",
                },
            },
        ]
        preds = [
            {"id": "1", "raw_output": json.dumps(gold[0]["label"])},
            {"id": "2", "raw_output": "not valid json at all"},
        ]
        report = evaluate_predictions(gold, preds)
        m = report["metrics"]
        self.assertEqual(m["total_examples"], 2)
        self.assertEqual(m["valid_schema_rate"], 0.5)
        self.assertEqual(m["all_labels_exact_match"], 0.5)
        self.assertIn("unparseable_json", m["schema_error_counts"])


if __name__ == "__main__":
    unittest.main()
