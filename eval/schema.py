"""Canonical schema and validation helpers for Kanglish Review & Support Triage."""

from __future__ import annotations

import json
import re
from typing import Any

SENTIMENTS = ("positive", "negative", "neutral", "mixed")
CATEGORIES = (
    "food_quality",
    "delivery_service",
    "ambience",
    "pricing",
    "ride_travel",
    "other",
)
URGENCIES = ("low", "medium", "high")

REQUIRED_FIELDS = ("sentiment", "category", "urgency", "english_summary")

SYSTEM_PROMPT = (
    "You are a precise triage assistant for Bengaluru and North Karnataka "
    "consumer reviews (Kanglish, Kannada-in-English-script, and English). "
    "Given a user message, output ONLY a valid JSON object with keys:\n"
    '- "sentiment": one of ["positive", "negative", "neutral", "mixed"]\n'
    '- "category": one of ["food_quality", "delivery_service", "ambience", "pricing", "ride_travel", "other"]\n'
    '- "urgency": one of ["low", "medium", "high"]\n'
    '- "english_summary": concise one-sentence English summary\n'
    "Category rules: use \"food_quality\" for taste, freshness, or expired food (even in parcels or near stations); "
    "use \"pricing\" for bills, cover charges, surge fares, discounts, or refunds (even at pubs or cabs); "
    "use \"delivery_service\" for delivery speed, packaging, or wrong/missing items; "
    "use \"ambience\" for venue vibe, seating, noise, Wi-Fi, or parking; "
    "use \"ride_travel\" for cabs, autos, buses, trains, or tourist spots; "
    "use \"other\" for app bugs, account/OTP issues, or customer support/staff behavior.\n"
    "Do not include any extra text or markdown."
)


def extract_json_object(raw_text: str) -> tuple[dict[str, Any] | None, bool]:
    """Parse JSON from raw model output, optionally stripping markdown fences.

    Returns (parsed_dict_or_none, was_strict_json_without_fences).
    """
    text = raw_text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed, True
    except json.JSONDecodeError:
        pass

    # Fallback: strip ```json ... ``` fences or grab first {...} block
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        brace_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        candidate = brace_match.group(0) if brace_match else None

    if candidate is not None:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, False
        except json.JSONDecodeError:
            pass

    return None, False


def validate_prediction(obj: dict[str, Any] | None) -> tuple[bool, list[str]]:
    """Validate that a parsed JSON object conforms to the Kanglish triage schema."""
    if obj is None:
        return False, ["unparseable_json"]

    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in obj:
            errors.append(f"missing_field:{field}")

    if "sentiment" in obj and obj["sentiment"] not in SENTIMENTS:
        errors.append(f"invalid_sentiment:{obj['sentiment']}")
    if "category" in obj and obj["category"] not in CATEGORIES:
        errors.append(f"invalid_category:{obj['category']}")
    if "urgency" in obj and obj["urgency"] not in URGENCIES:
        errors.append(f"invalid_urgency:{obj['urgency']}")
    if "english_summary" in obj:
        summary = obj["english_summary"]
        if not isinstance(summary, str) or not summary.strip():
            errors.append("invalid_english_summary")

    return len(errors) == 0, errors
