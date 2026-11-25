# Kanglish Review & Support Triage (`kanglish-triage`)

A portfolio-grade fine-tuning & evaluation project for classifying **code-mixed Kanglish (Kannada in English script + English)** reviews and support messages across Bengaluru and North Karnataka food, cafe, delivery, and travel/ride domains.

## Project Structure
- `data/seed_examples.jsonl` — Initial 24-example curated seed dataset (Bengaluru + North Karnataka).
- `eval/schema.py` — Canonical JSON schema, system prompt, and output validator.
- `eval/evaluate.py` — Reproducible evaluation harness (JSON validity, strict JSON rate, accuracy, macro-F1, and failure inspection).
- `eval/test_evaluate.py` — Unit tests for schema validation and the evaluation harness.

## Quick Start (Verify Eval Harness)
```bash
python -m unittest eval/test_evaluate.py -v
```
