#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

RULE1_PHRASES = [
    ["दे", "दिया"],
    ["मिला", "दें"],
    ["मुकर", "जाएं"],
    ["सम्मिलित", "करना"],
    ["हाल", "ही", "में"],
    ["कर", "दी"],
    ["दे", "दो"],
    ["दे", "दें"],
    ["दी", "थी"],
]

ATTACH_TO_LEFT = {
    "से",
    "में",
    "का",
    "के",
    "की",
    "को",
    "पर",
    "ने",
    "भी",
    "ही",
    "द्वारा",
    "वाला",
    "वाली",
    "वाले",
    "जी",
    "सी",
    "तरह",
    "दी_थी",
    "ईस्वी",
    "ई.पू.",
}

RULE3_MULTIWORDS = [
    ["रहे", "हैं"],
    ["रहा", "है"],
    ["रही", "है"],
    ["सकता", "है"],
    ["सकती", "है"],
    ["सकते", "हैं"],
    ["हो", "गयी"],
    ["हो", "गया"],
    ["के", "लिए"],
    ["के", "बाद"],
    ["के", "साथ"],
    ["के", "बीच"],
    ["के", "दौरान"],
    ["के", "खिलाफ़"],
    ["के", "प्रति"],
    ["की", "ओर"],
    ["बारे", "में"],
    ["के", "मुताबिक़"],
    ["के", "मुताबिक"],
    ["के", "तहत"],
    ["ओर", "से"],
    ["के", "कारण"],
    ["ने", "भी"],
    ["में", "ही"],
    ["ही", "में"],
]

ATTACH_MULTI_TO_LEFT = {"_".join(p) for p in RULE3_MULTIWORDS}

A_ENDING = "ा"
EE_ENDING = "ी"
E_ENDING = "े"

AUX_AFTER_EE = {
    "जाती",
    "गई",
    "जाएगी",
    "जायेगी",
    "है",
    "हैं",
    "थी",
    "थे",
    "था",
}
AUX_AFTER_A = {
    "जाता",
    "गया",
    "जाएगा",
    "जायेगा",
    "है",
    "हैं",
    "थी",
    "थे",
    "था",
}


def is_number_token(w: str) -> bool:
    cleaned = w.replace(",", "").replace(".", "")
    return cleaned.isdigit()


def apply_rule5_once(tokens: List[str]) -> List[str]:
    out: List[str] = []
    i = 0
    n = len(tokens)

    while i < n:
        w = tokens[i]

        if w.endswith(EE_ENDING) and i + 1 < n and tokens[i + 1] in AUX_AFTER_EE:
            out.append(w + "_" + tokens[i + 1])
            i += 2
            continue

        if w.endswith(A_ENDING) and i + 1 < n and tokens[i + 1] in AUX_AFTER_A:
            out.append(w + "_" + tokens[i + 1])
            i += 2
            continue

        if i + 1 < n and tokens[i + 1] == "चाहिए":
            if w.endswith(A_ENDING) or w.endswith(EE_ENDING):
                out.append(w + "_चाहिए")
                i += 2
                continue

        if (
            w.endswith(E_ENDING)
            and i + 2 < n
            and tokens[i + 1] in {"लगता", "लगती"}
            and tokens[i + 2] == "है"
        ):
            out.append(w + "_" + tokens[i + 1] + "_" + tokens[i + 2])
            i += 3
            continue

        out.append(w)
        i += 1

    return out


def apply_rule5_iterative(tokens: List[str]) -> List[str]:
    while True:
        new_tokens = apply_rule5_once(tokens)
        if new_tokens == tokens:
            break
        tokens = new_tokens
    return tokens


def apply_phrase_grouping(tokens: List[str]) -> List[str]:
    all_phrases = RULE1_PHRASES + RULE3_MULTIWORDS
    all_phrases = sorted(all_phrases, key=len, reverse=True)

    out: List[str] = []
    i = 0
    n = len(tokens)

    while i < n:
        matched = False
        for phrase in all_phrases:
            L = len(phrase)
            if i + L <= n and tokens[i : i + L] == phrase:
                out.append("_".join(phrase))
                i += L
                matched = True
                break
        if matched:
            continue

        out.append(tokens[i])
        i += 1

    return apply_rule5_iterative(out)


def apply_right_attachment(tokens: List[str]) -> List[str]:
    out: List[str] = []
    i = 0
    n = len(tokens)

    while i < n:
        w = tokens[i]
        if (w == "नहीं" or is_number_token(w)) and i + 1 < n:
            out.append(w + "_" + tokens[i + 1])
            i += 2
        else:
            out.append(w)
            i += 1

    return out


def apply_left_attachment(tokens: List[str]) -> List[str]:
    out: List[str] = []
    for tok in tokens:
        if tok in ATTACH_TO_LEFT or tok in ATTACH_MULTI_TO_LEFT:
            if out:
                out[-1] = out[-1] + "_" + tok
            else:
                out.append(tok)
        else:
            out.append(tok)
    return out


def group_sentence(sentence: str) -> str:
    tokens = sentence.strip().split()
    if not tokens:
        return ""

    tokens = apply_phrase_grouping(tokens)
    tokens = apply_right_attachment(tokens)
    tokens = apply_left_attachment(tokens)

    return " ".join(tokens)


def grouped_line_to_segments(grouped_line: str) -> List[str]:
    if not grouped_line.strip():
        return []
    return [t.replace("_", "##") for t in grouped_line.split()]


def turns_to_rows(turns: object) -> List[List[str]]:
    rows: List[List[str]] = []
    if not isinstance(turns, list):
        return rows

    for turn in turns:
        if isinstance(turn, (list, tuple)):
            for part in turn:
                if not isinstance(part, str) or not part.strip():
                    continue
                grouped = group_sentence(part)
                row = grouped_line_to_segments(grouped)
                if row:
                    rows.append(row)
        elif isinstance(turn, str) and turn.strip():
            grouped = group_sentence(turn)
            row = grouped_line_to_segments(grouped)
            if row:
                rows.append(row)
    return rows


def process_line(obj: dict) -> dict | None:
    key = "hin_Deva"
    if key not in obj:
        return None
    rows = turns_to_rows(obj[key])
    if not rows:
        return None
    return {key: rows}


def main() -> None:
    here = Path(__file__).resolve().parent
    default_in = here / "indic_instruct_align.jsonl"
    default_out = here / "grouped.jsonl"
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(default_in))
    parser.add_argument("--output", type=str, default=str(default_out))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    n_in = 0
    n_out = 0
    with open(args.input, "r", encoding="utf-8") as fin, open(
        args.output, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if args.limit and n_in >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skip bad JSON line {n_in + 1}: {e}", file=sys.stderr)
                n_in += 1
                continue
            n_in += 1
            out_obj = process_line(obj)
            if out_obj is None:
                continue
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"lines in: {n_in}, lines out: {n_out}, wrote {args.output}")


if __name__ == "__main__":
    main()