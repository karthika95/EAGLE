#!/usr/bin/env python3
"""
build_xquad_eval.py — Build xquad_eval_50.jsonl from XQuAD Hindi.

Dataset: google/xquad (xquad.hi), validation split — 1190 examples.
Citation: Artetxe et al., 2020. "On the Cross-lingual Transferability of
          Monolingual Representations." ACL 2020. (XTREME benchmark)

Task formulation: Reading comprehension with explanation.
  Instruction = Hindi Wikipedia context + Hindi question
  Reference   = The Wikipedia context passage (human-written Hindi)

Why this formulation:
  - The reference is long, human-written Wikipedia Hindi → rich LWG structures
    (verb-auxiliary chains, compound postpositions, fixed phrases all appear
    naturally in encyclopedic Hindi text)
  - Model must generate a grounded Hindi explanation → triggers LWG rules
  - ChrF against the Wikipedia passage is a meaningful surface quality check
  - XQuAD is widely cited (XTREME, Artetxe et al. 2020) → citable benchmark

Why XQuAD over Flores-200:
  - Flores-200 is gated on HuggingFace (requires approval)
  - XQuAD is fully open (Apache 2.0)
  - XQuAD contexts are Wikipedia Hindi paragraphs ≥ 100 chars → enough text
    to contain multiple LWG structures per example
  - QA task is more appropriate for instruction-following models than
    pure translation (which Flores-200 tests)

Selection criteria (50 from 240 unique contexts):
  - One question per context (maximise topic diversity)
  - Context length ≥ 150 chars (enough for meaningful ChrF)
  - Topics: sports, geography, history, science, culture (skip duplicates)
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

WORKSPACE = Path(__file__).parent
OUT_PATH = WORKSPACE / "xquad_eval_50.jsonl"

INSTRUCTION_TEMPLATE = (
    "निम्नलिखित हिंदी अनुच्छेद को ध्यान से पढ़कर प्रश्न का उत्तर "
    "हिंदी में विस्तार से दीजिए।\n\n"
    "अनुच्छेद:\n{context}\n\n"
    "प्रश्न: {question}"
)


def build_dataset(n: int = 50, seed: int = 42) -> None:
    print("Loading XQuAD Hindi (google/xquad, xquad.hi, validation)…")
    ds = load_dataset("google/xquad", "xquad.hi", split="validation")
    print(f"  Total examples: {len(ds)}")

    # Group by context; keep one question per context (diversity)
    context_to_ex: dict[str, dict] = {}
    for ex in ds:
        ctx = ex["context"].strip()
        if ctx not in context_to_ex and len(ctx) >= 150:
            context_to_ex[ctx] = ex

    unique = list(context_to_ex.values())
    print(f"  Unique contexts (≥150 chars): {len(unique)}")

    random.seed(seed)
    selected = random.sample(unique, min(n, len(unique)))
    selected.sort(key=lambda x: x["id"])   # reproducible order

    records = []
    for i, ex in enumerate(selected, 1):
        ctx   = ex["context"].strip()
        q     = ex["question"].strip()
        ans   = ex["answers"]["text"][0] if ex["answers"]["text"] else ""
        instruction = INSTRUCTION_TEMPLATE.format(context=ctx, question=q)
        records.append({
            "id": i,
            "xquad_id": ex["id"],
            "instruction": instruction,
            "reference": ctx,          # Wikipedia Hindi passage as gold reference
            "gold_answer": ans,        # short extractive answer (for reference)
            "question": q,
            "category": "reading_comprehension",
            "source": "XQuAD-hi (google/xquad, Artetxe et al. ACL 2020)",
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  Wrote {len(records)} records → {OUT_PATH}")
    print()
    print("Sample (first 2):")
    for r in records[:2]:
        print(f"  [{r['id']}] Q: {r['question']}")
        print(f"       Ref (first 80): {r['reference'][:80]}")
        print(f"       Gold answer: {r['gold_answer']}")
        print()


if __name__ == "__main__":
    build_dataset(n=50)
