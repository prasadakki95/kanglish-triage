"""Interactive Gradio demo for Kanglish Review & Support Triage.

Run locally or on Hugging Face Spaces:
    python app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "eval"))

from schema import SYSTEM_PROMPT, extract_json_object, validate_prediction  # noqa: E402

BASE_MODEL_ID = os.getenv("KANGLISH_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
LOCAL_ADAPTER = ROOT_DIR / "outputs" / "best-lora-adapter"
DEFAULT_ADAPTER = str(LOCAL_ADAPTER) if LOCAL_ADAPTER.exists() else os.getenv(
    "KANGLISH_ADAPTER_ID",
    "prasadakki95/qwen2.5-1.5b-kanglish-triage-lora",
)

_TOKENIZER = None
_MODEL = None


def load_model():
    global _TOKENIZER, _MODEL
    if _MODEL is not None and _TOKENIZER is not None:
        return _TOKENIZER, _MODEL

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    _TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if _TOKENIZER.pad_token is None:
        _TOKENIZER.pad_token = _TOKENIZER.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    _MODEL = PeftModel.from_pretrained(base, DEFAULT_ADAPTER)
    _MODEL.eval()
    return _TOKENIZER, _MODEL


@torch.inference_mode()
def triage_review(text: str) -> tuple[dict, str]:
    if not text or not text.strip():
        return {"error": "Please enter a Kanglish or English review."}, "Empty input"

    tokenizer, model = load_model()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text.strip()},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=160,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    parsed, is_strict = extract_json_object(raw_output)
    is_valid, errors = validate_prediction(parsed)

    if is_valid and parsed is not None:
        status = "Valid schema (strict JSON)" if is_strict else "Valid schema (extracted JSON)"
        return parsed, status

    return {"raw_output": raw_output, "errors": errors}, f"Invalid schema: {', '.join(errors)}"


EXAMPLES = [
    ["Indiranagar cafe ambience sakath agide, but filter coffee rate tumba jasti aytu guru."],
    ["Hubballi railway station hatra iruva hotel alli shenga holige mattu tuppa tinde, swarga guru!"],
    ["Order madi 1 hour aytu, delivery boy phone pick madta illa. Tumba late aytu."],
    ["Koramangala pub alli stag entry ge 2000 cover charge! Adakke 2 beer kuda baralla, robbery guru."],
    ["Namma Yatri auto anna time ge bandru, extra kuda kelilla, smooth ride."],
]

demo = gr.Interface(
    fn=triage_review,
    inputs=gr.Textbox(
        label="Kanglish / English Review or Support Message",
        placeholder="e.g. Hubballi khanavali jolada rotti bhari aiti, full maja bantu!",
        lines=3,
    ),
    outputs=[
        gr.JSON(label="Structured Triage Output"),
        gr.Textbox(label="Schema Validation"),
    ],
    title="Kanglish Review & Support Triage",
    description=(
        "Fine-tuned `Qwen2.5-1.5B-Instruct` (4-bit QLoRA) for code-mixed Kanglish "
        "(Bengaluru & North Karnataka) reviews across food, cafes, delivery, pricing, and travel/rides. "
        f"Adapter: `{DEFAULT_ADAPTER}`"
    ),
    examples=EXAMPLES,
)

if __name__ == "__main__":
    demo.launch(share=os.getenv("GRADIO_SHARE", "false").lower() == "true")
