#!/usr/bin/env python3
"""
build_vocab_scan.py — Day 1 AM

Scans the tokenizer vocabulary of Hindi LLMs, classifies each token according
to Samanvaya LWG rules, pre-computes valid token sets per FSM state, and
computes the WGTF (Word Group Tokenization Fidelity) diagnostic.

Outputs: vocab_scan_{model_key}.json
"""

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

# Import Samanvaya constants
sys.path.insert(0, str(Path(__file__).parent))
from wordgrouping_rules import (
    ATTACH_TO_LEFT,
    AUX_AFTER_A,
    AUX_AFTER_EE,
    RULE1_PHRASES,
    RULE3_MULTIWORDS,
)

# ---------------------------------------------------------------------------
# Derived sets
# ---------------------------------------------------------------------------

ALL_AUX = AUX_AFTER_EE | AUX_AFTER_A

# First words of RULE1 + RULE3 phrases that trigger MUST_CONT
PHRASE_START_TO_PHRASE: dict[str, list[list[str]]] = {}
for phrase in RULE1_PHRASES + RULE3_MULTIWORDS:
    PHRASE_START_TO_PHRASE.setdefault(phrase[0], []).append(phrase)

# All continuation words across all phrases (position ≥ 1)
ALL_PHRASE_CONT: set[str] = set()
for phrase in RULE1_PHRASES + RULE3_MULTIWORDS:
    for word in phrase[1:]:
        ALL_PHRASE_CONT.add(word)

# MUST_CASE trigger words
MUST_CASE_TRIGGERS = {"के", "की"}

# Valid next words per MUST_CASE trigger
MUST_CASE_VALID: dict[str, set[str]] = {t: set() for t in MUST_CASE_TRIGGERS}
for phrase in RULE3_MULTIWORDS:
    if phrase[0] in MUST_CASE_TRIGGERS:
        MUST_CASE_VALID[phrase[0]].add(phrase[1])

ALL_MUST_CASE_VALID = MUST_CASE_VALID["के"] | MUST_CASE_VALID["की"]

# All Samanvaya vocabulary words (for WGTF)
SAMANVAYA_VOCAB: set[str] = set()
SAMANVAYA_VOCAB |= ATTACH_TO_LEFT
SAMANVAYA_VOCAB |= ALL_AUX
for phrase in RULE1_PHRASES + RULE3_MULTIWORDS:
    SAMANVAYA_VOCAB.update(phrase)


# ---------------------------------------------------------------------------
# Token classification
# ---------------------------------------------------------------------------

TOKEN_CLASSES = ("CASE_MARKER", "AUX_VERB", "PHRASE_START", "PHRASE_CONT",
                 "CASE_MARKER_MERGED", "FREE")


def strip_spiece(tok_str: str) -> str:
    """Remove SentencePiece ▁ prefix."""
    return tok_str.lstrip("▁")


def has_case_marker_suffix(word: str) -> bool:
    """Check if word ends with a case marker (and word != case marker)."""
    for marker in ATTACH_TO_LEFT:
        if word.endswith(marker) and word != marker and len(word) > len(marker):
            return True
    return False


def classify_token(tok_str: str) -> str:
    word = strip_spiece(tok_str)
    if not word:
        return "FREE"

    if word in ATTACH_TO_LEFT:
        return "CASE_MARKER"
    if word in ALL_AUX:
        return "AUX_VERB"
    if word in PHRASE_START_TO_PHRASE:
        return "PHRASE_START"
    if word in ALL_PHRASE_CONT:
        return "PHRASE_CONT"
    if has_case_marker_suffix(word):
        return "CASE_MARKER_MERGED"
    return "FREE"


# ---------------------------------------------------------------------------
# WGTF computation
# ---------------------------------------------------------------------------

def compute_wgtf(tokenizer, vocab_words: set[str]) -> dict:
    """
    Word Group Tokenization Fidelity: fraction of Samanvaya words that exist
    as a single token in the tokenizer vocabulary.

    A word is "single-token" if the tokenizer vocab contains either:
      - "▁" + word  (word-initial SentencePiece token), or
      - word itself (no-prefix form)
    We use direct vocab lookup rather than encoding a padded string because
    some tokenizers (e.g. LlamaTokenizerFast) insert a standalone ▁ before
    word-initial tokens when encoding a space-prefixed string, inflating the
    count to 2.
    """
    tok_vocab = tokenizer.get_vocab()
    total = 0
    single_token = 0
    multi_token_words = []

    for word in sorted(vocab_words):
        total += 1
        # Prefer the word-initial (▁-prefixed) form; fall back to bare word
        if ("▁" + word) in tok_vocab or word in tok_vocab:
            single_token += 1
        else:
            # Estimate subword count by encoding the bare word
            ids = tokenizer.encode(word, add_special_tokens=False)
            multi_token_words.append((word, len(ids)))

    return {
        "total_samanvaya_words": total,
        "single_token_count": single_token,
        "wgtf": round(single_token / total, 4) if total > 0 else 0.0,
        "multi_token_words": sorted(multi_token_words, key=lambda x: x[1], reverse=True)[:50],
    }


# ---------------------------------------------------------------------------
# Main scanning logic
# ---------------------------------------------------------------------------

def scan_tokenizer(tokenizer, model_key: str) -> dict:
    vocab = tokenizer.get_vocab()  # str -> int
    vocab_size = len(vocab)
    print(f"[{model_key}] Vocab size: {vocab_size}")

    # Classify every token
    token_records: list[dict] = []
    class_counts: dict[str, int] = {c: 0 for c in TOKEN_CLASSES}

    # state_valid_tokens: token IDs for each FSM constrained state
    state_valid: dict[str, list[int]] = {
        "MUST_AUX": [],
        "MUST_CASE_KE": [],   # valid tokens after "के"
        "MUST_CASE_KI": [],   # valid tokens after "की"
        "MUST_CASE": [],      # union of above two
    }

    for tok_str, tok_id in vocab.items():
        cls = classify_token(tok_str)
        class_counts[cls] += 1
        word = strip_spiece(tok_str)

        token_records.append({
            "id": tok_id,
            "token": tok_str,
            "word": word,
            "class": cls,
        })

        # Build state_valid_tokens
        if cls == "AUX_VERB":
            state_valid["MUST_AUX"].append(tok_id)

        if word in MUST_CASE_VALID["के"]:
            state_valid["MUST_CASE_KE"].append(tok_id)
            state_valid["MUST_CASE"].append(tok_id)

        if word in MUST_CASE_VALID["की"]:
            state_valid["MUST_CASE_KI"].append(tok_id)
            if tok_id not in state_valid["MUST_CASE"]:
                state_valid["MUST_CASE"].append(tok_id)

    # De-duplicate and sort for reproducibility
    for key in state_valid:
        state_valid[key] = sorted(set(state_valid[key]))

    print(f"[{model_key}] Token class counts: {class_counts}")
    print(f"[{model_key}] MUST_AUX tokens: {len(state_valid['MUST_AUX'])}")
    print(f"[{model_key}] MUST_CASE tokens: {len(state_valid['MUST_CASE'])}")

    # Compute WGTF
    wgtf_result = compute_wgtf(tokenizer, SAMANVAYA_VOCAB)
    print(f"[{model_key}] WGTF = {wgtf_result['wgtf']:.2%} "
          f"({wgtf_result['single_token_count']}/{wgtf_result['total_samanvaya_words']})")

    return {
        "model_key": model_key,
        "vocab_size": vocab_size,
        "token_class_counts": class_counts,
        "wgtf": wgtf_result,
        "state_valid_tokens": state_valid,
        "token_records": sorted(token_records, key=lambda r: r["id"]),
    }


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "airavata": {
        "model_id": "ai4bharat/Airavata",
        "trust_remote_code": False,
    },
    "param": {
        "model_id": "bharatgenai/Param-1-2.9B-Instruct",
        "trust_remote_code": True,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Build vocab scan JSON for Hindi LLMs")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        default=["all"],
        help="Which models to scan (default: all)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).parent),
        help="Output directory for vocab_scan_*.json files",
    )
    args = parser.parse_args()

    models_to_scan = (
        list(MODEL_REGISTRY.keys())
        if "all" in args.models
        else args.models
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for key in models_to_scan:
        cfg = MODEL_REGISTRY[key]
        print(f"\n{'='*60}")
        print(f"Scanning: {cfg['model_id']}")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(
            cfg["model_id"],
            trust_remote_code=cfg["trust_remote_code"],
        )

        result = scan_tokenizer(tokenizer, key)

        out_path = outdir / f"vocab_scan_{key}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[{key}] Written → {out_path}")


if __name__ == "__main__":
    main()
