#!/usr/bin/env python3
"""
build_mt_bench_eval.py — Build mt_bench_hi_eval.jsonl from MT-Bench-Hi.

Dataset : nvidia/MT-Bench-Hi (200 multi-turn Hindi prompts, CC-BY 4.0)
Citation: NVIDIA, 2024. https://huggingface.co/datasets/nvidia/MT-Bench-Hi

Task: Open-ended Hindi instruction following (turn 1 only).
  Instruction = turns[0]   (first-turn Hindi question)
  Reference   = None       (no gold reference; LWG-CR is primary metric)

Category tiers
--------------
  primary      : writing, humanities, roleplay, extraction
                 → Hindi prose dominant; all LWG metrics reported
  secondary    : reasoning, stem
                 → Formally structured Hindi; LWG metrics reported with caveat
  out_of_scope : math, coding
                 → Likely English/code/numeric output; compliance ≈ 0 expected;
                   reported separately as sanity check, excluded from headline

All 200 prompts are evaluated; per-category breakdown is in the summary table.
"""

import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset

WORKSPACE = Path(__file__).parent
OUT_PATH = WORKSPACE / "mt_bench_hi_eval.jsonl"

HINDI_PROSE = {"writing", "humanities", "roleplay", "extraction"}
MIXED       = {"reasoning", "stem"}
NON_HINDI   = {"math", "coding"}


def tier(cat: str) -> str:
    c = cat.lower()
    if c in HINDI_PROSE:  return "primary"
    if c in MIXED:        return "secondary"
    return "out_of_scope"


def build_dataset() -> None:
    print("Loading MT-Bench-Hi (nvidia/MT-Bench-Hi)…")
    ds = load_dataset("nvidia/MT-Bench-Hi", split="test")
    print(f"  Total examples: {len(ds)}")

    records = []
    for ex in ds:
        turns = ex["turns"]
        cat   = str(ex.get("category", "unknown")).lower()
        records.append({
            "id":          ex["question_id"],
            "instruction": turns[0],
            "turn2":       turns[1] if len(turns) > 1 else "",
            "category":    cat,
            "tier":        tier(cat),
            "reference":   None,
            "source":      "MT-Bench-Hi (nvidia/MT-Bench-Hi, NVIDIA 2024, CC-BY 4.0)",
        })

    records.sort(key=lambda x: x["id"])

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n  Wrote {len(records)} records → {OUT_PATH}")

    cat_counts  = Counter(r["category"] for r in records)
    tier_counts = Counter(r["tier"]     for r in records)

    print("\nCategory breakdown:")
    for cat, count in sorted(cat_counts.items()):
        t = tier(cat)
        print(f"  {cat:15s}  {count:3d}  [{t}]")

    print("\nTier summary:")
    for t, count in sorted(tier_counts.items()):
        print(f"  {t:15s}  {count:3d}")


if __name__ == "__main__":
    build_dataset()
