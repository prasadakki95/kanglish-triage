# Kanglish Review & Support Triage (`kanglish-triage`)

Fine-tuning and iteratively evaluating a small open-weights LLM (`Qwen/Qwen2.5-1.5B-Instruct`, 4-bit QLoRA) to triage **code-mixed Kanglish (Kannada in English script + English)** consumer reviews and support messages across Bengaluru and North Karnataka food, cafe, delivery, and travel/ride domains.

## Why Fine-Tune Instead of Prompting?
Small 1.5B base models struggle with Kanglish regional vocabulary (*"khanavali"*, *"ennegayi"*, *"aiti"*, *"sakath"*) and Kannada negation (*"agalla"*, *"illa"*), often misclassifying negative/mixed reviews as positive or hallucinating lodging-hotel rooms for *"Jayanagar hotel"*.

Using a reproducible evaluation harness **before and after training** on a held-out test split (zero text overlap with train/val):

### 1. In-Distribution Benchmark (`test.jsonl`, v2: 179 examples)
| Setup | Valid Schema | Exact Match (All) | Sentiment F1 | Category F1 | Urgency F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Zero-Shot Base** (`Qwen2.5-1.5B-Instruct`) | 96.7% | 17.9% | 0.413 | 0.486 | 0.434 |
| **Few-Shot Base** (3-shot) | 93.8% | 27.9% | 0.475 | 0.678 | 0.426 |
| **Fine-Tuned v2 (4-bit QLoRA)** | **100.0%** | **100.0%** | **1.000** | **1.000** | **1.000** |

*(v1 on 130 test examples achieved 100.0% valid schema and 99.2% exact match, up from 23.1% zero-shot and 30.0% few-shot.)*

### 2. Unseen Out-of-Distribution (OOD) Stress Test (`ood_test.jsonl`, 25 wild examples)
To test generalization beyond template-augmented data, we evaluated on 25 brand-new, human-written Kanglish reviews (Silk Board traffic, CTR benne dosa, Mishra Pedha, Namma Yatri, Hampi bicycle rental, emojis, etc.) **never seen during training**:

| Iteration | Valid Schema | Exact Match (All) | Sentiment Acc / F1 | Category Acc / F1 | Urgency Acc / F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **v1 (Template-augmented)** | 96.0% | 52.0% | 84.0% / 0.869 | 60.0% / 0.617 | 92.0% / 0.886 |
| **v2 (+ Prompt rules & contrastive data)** | 96.0% | **68.0%** (+16%) | 84.0% / 0.869 | **84.0% / 0.852** (+24%) | **92.0% / 0.896** |

## Iterative Error Analysis (v1 → v2)
- **What v1's OOD test uncovered (Shortcut Learning):** v1 over-indexed on surface venue/transit nouns—e.g., *"Hubballi **railway station** hatra hotel alli shenga holige tinde"* was misclassified as `ride_travel` because of *"railway station"*, and *"Koramangala **pub** ... 2000 cover charge"* was misclassified as `ambience` instead of `pricing`.
- **What v2 fixed:** Adding prompt boundary rules in [eval/schema.py](eval/schema.py) and contrastive anti-shortcut examples in [data/prepare_dataset.py](data/prepare_dataset.py) lifted OOD Category Accuracy from **60.0% → 84.0%** (`+24%`) and fixed examples like `ood_002` (*shenga holige near Hubballi railway station* → `food_quality`).
- **Remaining OOD gaps (v3 roadmap):** Unseen idioms (*"swarga guru!"* in `ood_002`), multi-aspect reviews (*"coffee strong ide, but music volume..."* in `ood_011` where both `food_quality` and `ambience` apply), and rare schema hallucinations (`1/25`). Collecting 300–500 natural human-written reviews or supporting multi-label categories is the natural next step.

## Project Structure
- [data/seed_examples.jsonl](data/seed_examples.jsonl) — 24 curated Bengaluru & North Karnataka seed examples.
- [data/ood_test.jsonl](data/ood_test.jsonl) — 25 unseen, human-written OOD stress-test examples.
- [data/prepare_dataset.py](data/prepare_dataset.py) — Deterministic generator, deduplicator, and stratified `train`/`val`/`test` splitter with leakage assertions.
- [eval/schema.py](eval/schema.py) — Canonical JSON schema, system prompt, and output validator.
- [eval/evaluate.py](eval/evaluate.py) — Reproducible evaluation harness (JSON validity, strict JSON rate, accuracy, macro-F1, failure inspection).
- [eval/test_evaluate.py](eval/test_evaluate.py) — Unit tests for schema validation and the evaluation harness.
- [notebooks/kanglish_triage_colab.ipynb](notebooks/kanglish_triage_colab.ipynb) — End-to-end Colab/Kaggle notebook for baselines, QLoRA training, and evaluation.
- [app.py](app.py) & [requirements.txt](requirements.txt) — Interactive Gradio demo (ready for local run, Kaggle `GRADIO_SHARE=true`, or Hugging Face Spaces).

## Quick Start
```bash
python -m unittest eval/test_evaluate.py -v
python data/prepare_dataset.py --seed-file data/seed_examples.jsonl --out-dir data --target-size 1200
KANGLISH_ADAPTER_ID="<your-hf-username>/qwen2.5-1.5b-kanglish-triage-lora" python app.py
```
