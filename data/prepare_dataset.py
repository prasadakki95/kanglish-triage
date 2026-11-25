"""Generate, curate, deduplicate, and split the Kanglish Triage dataset.

Combines curated seed examples with diverse, template-driven Bengaluru and
North Karnataka Kanglish examples across food, cafes, delivery, pricing,
travel/rides, and app support, then writes stratified train/val/test splits.

Usage:
    python data/prepare_dataset.py --seed data/seed_examples.jsonl --out-dir data --target-size 900
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "eval"))

from schema import validate_prediction  # noqa: E402


BENGALURU_AREAS = [
    "Indiranagar", "Koramangala", "HSR Layout", "Jayanagar",
    "Malleshwaram", "Whitefield", "MG Road", "Church Street", "BTM Layout",
]
NORTH_KARNATAKA_CITIES = [
    "Hubballi", "Dharwad", "Belagavi", "Kalaburagi",
    "Vijayapura", "Hampi", "Gokarna", "Bagalkot", "Gadag",
]

FOOD_ITEMS_BLR = [
    "masala dosa", "filter coffee", "biryani", "idli vada", "bisi bele bath",
    "veg roll", "paneer tikka", "cold brew", "chocolate waffle", "ragi mudde",
]
FOOD_ITEMS_NK = [
    "jolada rotti", "ennegayi", "girmit", "mirchi bajji", "dharwad pedha",
    "shenga chutney", "kunda", "mandakki", "kadak rotti", "uddina vada",
]

# Each template tuple:
# (text_template, sentiment, category, urgency, summary_template, region_hint)
TEMPLATES: list[tuple[str, str, str, str, str, str]] = [
    # --- FOOD QUALITY ---
    (
        "{area} khanavali {food} {pos_adj}, full maja bantu {honorific}!",
        "positive", "food_quality", "low",
        "Customer loved the {food} at the {area} restaurant.", "nk",
    ),
    (
        "{area} alli {food} taste {pos_adj}, {neg_clause_price}.",
        "mixed", "food_quality", "low",
        "Customer enjoyed the {food} in {area} but felt the pricing was high.", "blr",
    ),
    (
        "{food} {neg_food_issue}, {area} hotel alli quality tumba drop agide.",
        "negative", "food_quality", "medium",
        "Customer reported that the {food} in {area} was poor quality.", "blr",
    ),
    (
        "{food} super agide anta helak agalla, {neg_food_issue} mattu taste illa.",
        "negative", "food_quality", "medium",
        "Customer was dissatisfied with the {food} as it lacked taste.", "nk",
    ),
    (
        "{area} kade {food} okayish aiti, special anslilla but tinbohudu.",
        "neutral", "food_quality", "low",
        "Customer found the {food} in {area} average and acceptable.", "nk",
    ),
    (
        "{food} alli hair mattu plastic piece siktu, {area} kitchen hygiene zero ide!",
        "negative", "food_quality", "high",
        "Customer found foreign objects in {food} from {area} and raised a hygiene concern.", "blr",
    ),

    # --- AMBIENCE (CAFES, CLUBS, RESTAURANTS) ---
    (
        "{area} work cafe Wi-Fi speed {pos_adj}, power sockets kuda ella table hatra ide.",
        "positive", "ambience", "low",
        "Customer praised the {area} cafe for fast Wi-Fi and accessible power sockets.", "blr",
    ),
    (
        "{area} rooftop club DJ {pos_adj}, aadre {neg_clause_price}.",
        "mixed", "ambience", "low",
        "Customer enjoyed the DJ at the {area} club but found it expensive.", "blr",
    ),
    (
        "{area} cafe alli AC work agta illa {honorific}, tumba seke aiti.",
        "negative", "ambience", "medium",
        "Customer complained that the AC was not working at the {area} cafe.", "nk",
    ),
    (
        "Illi parking space illa {honorific}, {area} kade car ittu barodu tumba kasta aiti.",
        "negative", "ambience", "low",
        "Customer noted that lack of parking makes visiting {area} by car difficult.", "nk",
    ),
    (
        "{area} pub music tumba loud ittu, matadakke kuda agta irlilla.",
        "negative", "ambience", "low",
        "Customer felt the music at the {area} pub was too loud for conversation.", "blr",
    ),
    (
        "Interior decent agide {area} alli, family jote hogakke {pos_adj}.",
        "positive", "ambience", "low",
        "Customer liked the family-friendly interior at the {area} restaurant.", "blr",
    ),

    # --- DELIVERY SERVICE ---
    (
        "Order madi {mins} mins aytu, delivery partner phone pick madta illa. Tumba late aytu.",
        "negative", "delivery_service", "high",
        "Order has been delayed for {mins} minutes and the delivery partner is unreachable.", "blr",
    ),
    (
        "Wrong item deliver madidira {honorific}, naanu veg {food} order madidde bere item kottidira!",
        "negative", "delivery_service", "high",
        "Customer received the wrong item instead of the ordered vegetarian {food}.", "blr",
    ),
    (
        "Packaging leak agi {food} ella bag olage chellide, box sarigilla.",
        "negative", "delivery_service", "medium",
        "Poor packaging caused the {food} order to spill inside the delivery bag.", "blr",
    ),
    (
        "Delivery boy tumba polite agidru, male iddru time ge {food} tandu kottidru.",
        "positive", "delivery_service", "low",
        "Customer appreciated the polite delivery partner for delivering {food} on time in the rain.", "blr",
    ),
    (
        "{area} kade delivery swalpa slow aiti, aadre food bisi bisi bandide.",
        "mixed", "delivery_service", "low",
        "Delivery in {area} was slightly slow, though the food arrived hot.", "nk",
    ),
    (
        "Status alli delivered anta torstide, aadre nange innu order bandilla!",
        "negative", "delivery_service", "high",
        "App marks the order as delivered even though the customer has not received it.", "blr",
    ),

    # --- PRICING ---
    (
        "Payment UPI inda cut aytu, aadre order confirm aglilla. Refund bega kodi please.",
        "negative", "pricing", "high",
        "UPI payment was deducted without order confirmation; customer requests an urgent refund.", "blr",
    ),
    (
        "{area} shack alli {food} rate kammine aiti, quantity nu chennagi kottidare.",
        "positive", "pricing", "low",
        "Customer found {food} in {area} reasonably priced with generous portions.", "nk",
    ),
    (
        "Menu card nodidre rate normal anstu, aadre bill alli GST mattu service charge jasti hakidare.",
        "negative", "pricing", "medium",
        "Customer was unhappy with high GST and service charges added to the bill.", "blr",
    ),
    (
        "Double charge agide card inda {area} booking ge, immediate reversal madsi.",
        "negative", "pricing", "high",
        "Customer was charged twice for a {area} booking and needs an immediate refund.", "blr",
    ),
    (
        "{food} taste {pos_adj}, aadre quantity nodidre 400 rupees waste anstu.",
        "mixed", "pricing", "low",
        "Customer liked the {food} taste but felt the portion was too small for 400 rupees.", "blr",
    ),

    # --- RIDE & TRAVEL ---
    (
        "Cab driver tumba rash drive madtidane, {area} signal jump madidru!",
        "negative", "ride_travel", "high",
        "Customer is reporting dangerous rash driving and signal jumping near {area}.", "blr",
    ),
    (
        "Auto driver meter hakalla anta helidru, {area} ge extra 150 keli jhagala madidru.",
        "negative", "ride_travel", "medium",
        "Auto driver refused the meter and demanded an extra 150 rupees to {area}.", "blr",
    ),
    (
        "{area} bus journey {pos_adj}, AC cool ittu mattu time ge reach aytu.",
        "positive", "ride_travel", "low",
        "Customer had a comfortable and punctual AC bus journey to {area}.", "nk",
    ),
    (
        "{area} weekend homestay book madidvi, location peaceful agide mattu host helpful.",
        "positive", "ride_travel", "low",
        "Customer enjoyed a peaceful homestay in {area} with a helpful host.", "nk",
    ),
    (
        "{area} kade train 45 mins late aytu, aadre station clean aiti.",
        "mixed", "ride_travel", "low",
        "Train to {area} was delayed by 45 minutes, though the station was clean.", "nk",
    ),
    (
        "{area} tour guide timings mattu entry ticket details beku, online alli sigta illa.",
        "neutral", "ride_travel", "low",
        "User is asking for {area} visiting timings and entry ticket details.", "nk",
    ),

    # --- OTHER (APP / ACCOUNT / GENERAL) ---
    (
        "App alli coupon code apply agta illa, checkout time alli error torstide.",
        "negative", "other", "medium",
        "Customer is facing an app error preventing coupon application at checkout.", "blr",
    ),
    (
        "Account login OTP barthane illa {honorific}, 10 mins inda try madtidini.",
        "negative", "other", "medium",
        "User is unable to receive the login OTP despite multiple attempts.", "blr",
    ),
    (
        "Namma new feature update chennagide, app speed tumba improve agide.",
        "positive", "other", "low",
        "User praised the new app update for improved speed.", "blr",
    ),
    (
        "Support chat alli bot matra reply madtide, human agent ge connect madsi.",
        "negative", "other", "medium",
        "User is frustrated with the support bot and wants to connect to a human agent.", "blr",
    ),
]

POS_ADJ_BLR = ["sakath agide", "chindi ide", "super agide", "tumba chennagide", "awesome ide"]
POS_ADJ_NK = ["bhari aiti", "mast aiti", "barabar aiti", "ekdam mast aiti", "sakath aiti"]
NEG_FOOD_ISSUES = [
    "tumba thanda agide", "uppu jasti agide", "kharab agide",
    "swalpa stale anstu", "sariyagi bendilla", "oil tumba jasti ide",
]
NEG_PRICE_CLAUSES = [
    "rate tumba jasti aytu", "entry charges swalpa over anstu",
    "coffee price pocket ge bhari anstu", "bill nodi shock aytu",
]
HONORIFICS_BLR = ["guru", "macha", "boss", "bro"]
HONORIFICS_NK = ["ri", "anna", "guru", "pa"]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def generate_examples(target_count: int, rng: random.Random) -> list[dict[str, Any]]:
    generated: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    attempts = 0
    max_attempts = target_count * 25

    while len(generated) < target_count and attempts < max_attempts:
        attempts += 1
        tpl, sentiment, category, urgency, sum_tpl, region_hint = rng.choice(TEMPLATES)

        if region_hint == "nk":
            area = rng.choice(NORTH_KARNATAKA_CITIES)
            food = rng.choice(FOOD_ITEMS_NK + FOOD_ITEMS_BLR[:3])
            pos_adj = rng.choice(POS_ADJ_NK)
            honorific = rng.choice(HONORIFICS_NK)
            region = "north_karnataka"
        else:
            area = rng.choice(BENGALURU_AREAS)
            food = rng.choice(FOOD_ITEMS_BLR)
            pos_adj = rng.choice(POS_ADJ_BLR)
            honorific = rng.choice(HONORIFICS_BLR)
            region = "bengaluru"

        mins = rng.choice([45, 50, 60, 75, 90])
        neg_food_issue = rng.choice(NEG_FOOD_ISSUES)
        neg_clause_price = rng.choice(NEG_PRICE_CLAUSES)

        text = tpl.format(
            area=area,
            food=food,
            pos_adj=pos_adj,
            honorific=honorific,
            mins=mins,
            neg_food_issue=neg_food_issue,
            neg_clause_price=neg_clause_price,
        )
        summary = sum_tpl.format(area=area, food=food, mins=mins)

        norm = normalize_text(text)
        if norm in seen_texts:
            continue
        seen_texts.add(norm)

        label = {
            "sentiment": sentiment,
            "category": category,
            "urgency": urgency,
            "english_summary": summary,
        }
        ok, errs = validate_prediction(label)
        if not ok:
            raise ValueError(f"Generated invalid label: {errs}")

        generated.append(
            {
                "id": f"syn_{len(generated) + 1:04d}",
                "region": region,
                "text": text,
                "label": label,
            }
        )

    return generated


def stratified_split(
    records: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        key = (rec["label"]["category"], rec["label"]["sentiment"])
        buckets[key].append(rec)

    train, val, test = [], [], []
    for bucket in buckets.values():
        rng.shuffle(bucket)
        n = len(bucket)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio))) if n >= 3 else 0
        if n_train + n_val >= n and n >= 3:
            n_train = n - 2
            n_val = 1

        train.extend(bucket[:n_train])
        val.extend(bucket[n_train : n_train + n_val])
        test.extend(bucket[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare train/val/test Kanglish datasets.")
    parser.add_argument("--seed-file", type=Path, default=Path("data/seed_examples.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--target-size", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    seed_records = load_jsonl(args.seed_file)
    seen = {normalize_text(r["text"]) for r in seed_records}

    synthetic = generate_examples(max(0, args.target_size - len(seed_records)), rng)
    combined = list(seed_records)
    for rec in synthetic:
        norm = normalize_text(rec["text"])
        if norm not in seen:
            seen.add(norm)
            combined.append(rec)

    train, val, test = stratified_split(combined, train_ratio=0.70, val_ratio=0.15, rng=rng)

    # Verify zero text leakage across splits
    train_texts = {normalize_text(r["text"]) for r in train}
    val_texts = {normalize_text(r["text"]) for r in val}
    test_texts = {normalize_text(r["text"]) for r in test}
    assert train_texts.isdisjoint(val_texts), "Leakage between train and val!"
    assert train_texts.isdisjoint(test_texts), "Leakage between train and test!"
    assert val_texts.isdisjoint(test_texts), "Leakage between val and test!"

    save_jsonl(args.out_dir / "train.jsonl", train)
    save_jsonl(args.out_dir / "val.jsonl", val)
    save_jsonl(args.out_dir / "test.jsonl", test)

    print(f"Total unique examples : {len(combined)}")
    print(f"Train / Val / Test    : {len(train)} / {len(val)} / {len(test)}")
    print("Category distribution :", dict(Counter(r["label"]["category"] for r in combined)))
    print("Sentiment distribution:", dict(Counter(r["label"]["sentiment"] for r in combined)))
    print("Region distribution   :", dict(Counter(r["region"] for r in combined)))


if __name__ == "__main__":
    main()
