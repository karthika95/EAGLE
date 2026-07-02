#!/usr/bin/env python3
"""
trace_scg.py — Step-by-step decode trace for SCG methodology explanation.

Shows every decode step: what FSM state is active, what the model's greedy
choice would have been (unconstrained), and what SCG forced instead.

Usage
-----
  python trace_scg.py --model airavata --gpu 1
  python trace_scg.py --model param   --gpu 3 --mode adaptive
  python trace_scg.py --model airavata --gpu 1 --prompt "महात्मा गांधी..."

Three built-in example prompts cover the three LWG rule types:
  1. Verb-auxiliary chain  : "खाया/बोला/गया → था/गई/है ..."
  2. Compound postposition : "के → लिए/बाद/साथ/बारे में ..."
  3. Open creative         : triggers multiple rules
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from samanvaya_logits_processor import SamanvayaLogitsProcessor
from scg_generate import MODELS, load_model

WORKSPACE = Path(__file__).parent

# ── Three pedagogical trace prompts ────────────────────────────────────────
TRACE_PROMPTS = [
    {
        "label": "Example 1 — Verb-Auxiliary Chain (MUST_AUX rule)",
        "prompt": "महात्मा गांधी के जीवन और स्वतंत्रता संग्राम में उनके योगदान के बारे में बताइए।",
        "why":    "Past-tense narrative → ends-in-ā/ī verb stems trigger MUST_AUX obligation.",
    },
    {
        "label": "Example 2 — Compound Postposition (MUST_CASE rule)",
        "prompt": "पर्यावरण प्रदूषण के प्रमुख कारण और उसके समाधान के उपाय बताइए।",
        "why":    "'के' triggers MUST_CASE; next word must be a valid compound-postposition continuation.",
    },
    {
        "label": "Example 3 — Open Creative (multiple rules)",
        "prompt": "एक प्रेरणादायक कहानी लिखिए जिसमें एक ग्रामीण लड़की शिक्षा प्राप्त करती है।",
        "why":    "Creative writing triggers verb-aux chains, के-postpositions, and fixed phrases together.",
    },
]


ACTION_SYMBOLS = {
    "PASS":              "·",
    "PASS_NO_VALID_IDS": "·",
    "PASS_EOS":          "⏎",
    "OPP_SAVE":          "✓",
    "ENTROPY_SKIP":      "⊘",
    "MASK_APPLIED":      "✦",
    "INIT":              " ",
}

ACTION_LABELS = {
    "PASS":              "PASS         (no constraint)",
    "PASS_NO_VALID_IDS": "PASS         (no valid token IDs — relaxed)",
    "PASS_EOS":          "PASS         (EOS generated)",
    "OPP_SAVE":          "OPP SAVE     (model already chose valid token)",
    "ENTROPY_SKIP":      "ENTROPY SKIP (adaptive: model confident, skip mask)",
    "MASK_APPLIED":      "MASK APPLIED (constraint enforced)",
    "INIT":              "INIT",
}


def _pp_token(s: str) -> str:
    """Pretty-print a token string for display."""
    return repr(s.replace("▁", "·"))[1:-1]  # show ▁ as · for readability


def format_trace(trace_log: list[dict], tokenizer, max_steps: int = 80) -> str:
    lines = []
    shown = 0

    for entry in trace_log:
        if entry.get("action") == "INIT":
            continue

        step   = entry["step"]
        tok    = _pp_token(entry.get("committed_token", ""))
        action = entry.get("action", "?")
        sym    = ACTION_SYMBOLS.get(action, "?")
        eff    = entry.get("eff_state", "FREE")
        buf    = entry.get("buffer", "")
        fsm_b  = entry.get("fsm_before", "FREE")
        fsm_a  = entry.get("fsm_after",  "FREE")
        pending= entry.get("pending",    "FREE")

        state_change = f"{fsm_b}→{fsm_a}" if fsm_b != fsm_a else fsm_b

        if action in ("PASS", "PASS_NO_VALID_IDS", "PASS_EOS"):
            note = f"buf='{buf}'"
            if pending != "FREE":
                note += f"  ⚑pending={pending}"
            lines.append(
                f" {step:4d} {sym}  {tok:22s}  {state_change:22s}  {note}"
            )

        elif action == "OPP_SAVE":
            greedy = _pp_token(entry.get("greedy_str", ""))
            prob   = entry.get("greedy_prob", 0)
            lines.append(
                f" {step:4d} {sym}  {tok:22s}  {state_change:22s}  "
                f"eff={eff}  greedy='{greedy}'({prob:.3f}) already valid ✓"
            )

        elif action == "ENTROPY_SKIP":
            greedy = _pp_token(entry.get("greedy_str", ""))
            H      = entry.get("entropy", 0)
            TG     = entry.get("T_G", 0)
            lines.append(
                f" {step:4d} {sym}  {tok:22s}  {state_change:22s}  "
                f"eff={eff}  H={H:.3f}≤T_G={TG:.3f} → skip"
            )

        elif action == "MASK_APPLIED":
            greedy = _pp_token(entry.get("greedy_str", ""))
            g_prob = entry.get("greedy_prob", 0)
            forced = _pp_token(entry.get("forced_str", ""))
            f_prob = entry.get("forced_prob", 0)
            H_str  = f"  H={entry['entropy']:.3f}" if entry.get("entropy") else ""
            lines.append(
                f" {step:4d} {sym}  {tok:22s}  {state_change:22s}  "
                f"eff={eff}{H_str}"
            )
            lines.append(
                f"       {'':22s}  BLOCKED: '{greedy}' (p={g_prob:.4f})  "
                f"→  FORCED: '{forced}' (p={f_prob:.4f})"
            )

        shown += 1
        if shown >= max_steps:
            lines.append(f"  ... (truncated at {max_steps} steps)")
            break

    return "\n".join(lines)


def run_trace(
    prompt: str,
    label: str,
    why: str,
    model_key: str,
    mode: str,
    device: str,
    max_new_tokens: int = 150,
) -> None:
    cfg = MODELS[model_key]
    model, tokenizer = load_model(model_key, device)

    full_prompt = cfg["prompt_template"].format(instruction=prompt)
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(device)

    processor = SamanvayaLogitsProcessor(
        tokenizer=tokenizer,
        vocab_scan=WORKSPACE / cfg["vocab_scan"],
        use_entropy_gate=(mode == "adaptive"),
        T_G_init=0.5,
        eos_token_id=tokenizer.eos_token_id,
        trace=True,
    )
    processor.reset_stats()

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            logits_processor=[processor],
            pad_token_id=tokenizer.eos_token_id,
        )
    processor._flush_buffer()

    new_ids  = output_ids[0, input_ids.shape[1]:]
    response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    gs = processor.generation_stats()

    # ── Also run baseline for direct comparison ───────────────────────────
    print()
    print("═" * 78)
    print(label)
    print("─" * 78)
    print(f"Why this prompt: {why}")
    print()
    print(f"PROMPT : {prompt}")
    print(f"MODE   : {mode}")
    print(f"MODEL  : {cfg['model_id']}")
    print()

    # Baseline
    with torch.no_grad():
        base_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    base_resp = tokenizer.decode(base_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

    print("─── Baseline (unconstrained) response ───────────────────────────────────")
    for line in textwrap.wrap(base_resp, width=76):
        print("  " + line)

    print()
    print(f"─── {mode.upper()} (SCG) response ──────────────────────────────────────────")
    for line in textwrap.wrap(response, width=76):
        print("  " + line)

    print()
    print("─── Decode Trace ────────────────────────────────────────────────────────")
    print()
    print("  Legend:  · PASS  ✓ OPP_SAVE (already valid)  ✦ MASK_APPLIED  ⊘ ENTROPY_SKIP")
    print()
    print(f" {'Step':>4}    {'Committed Token':22s}  {'FSM State':22s}  Note")
    print(f" {'────':>4}    {'──────────────────────':22s}  {'──────────────────────':22s}  ────")
    print(format_trace(processor.get_trace_log(), tokenizer, max_steps=100))

    print()
    print("─── Step Summary ────────────────────────────────────────────────────────")
    print(f"  Total decode steps    : {gs['total_steps']}")
    print(f"  Constrained steps     : {gs['constrained_steps']}  "
          f"({gs['constrained_steps']/gs['total_steps']*100:.1f}% of all steps)")
    print(f"  Opportunistic saves   : {gs['opportunistic_saves']}  "
          f"(model already correct in {gs['opp_save_rate']*100:.1f}% of constrained steps)")
    print(f"  Mask applied          : {gs['mask_applied']}  "
          f"(intervention rate = {gs['intervention_rate']*100:.1f}%)")
    if mode == "adaptive":
        print(f"  Entropy gate skips    : {gs['entropy_gate_skips']}")
    print(f"  LWG compliance        : {processor.parser.compliance_rate():.2%}")
    ps = gs["parser_stats"]
    print(f"  Obligations seen      : {ps['obligations']}  "
          f"(completions={ps['completions']}, violations={ps['violations']})")
    print()

    # Diff summary: where did the two responses diverge?
    base_words = base_resp.split()
    scg_words  = response.split()
    diffs = [(i, b, s) for i, (b, s) in enumerate(zip(base_words, scg_words)) if b != s]
    if diffs:
        print("─── Word-level Divergences (baseline vs SCG) ────────────────────────────")
        for idx, bw, sw in diffs[:10]:
            print(f"  Word {idx+1:3d}: '{bw}'  →  '{sw}'")
    else:
        print("─── Responses are identical at word level (no mask was applied) ─────────")
    print("═" * 78)


def main():
    ap = argparse.ArgumentParser(description="SCG decode trace")
    ap.add_argument("--model",  choices=list(MODELS), default="airavata")
    ap.add_argument("--mode",   choices=["constrained", "adaptive"], default="constrained")
    ap.add_argument("--gpu",    type=int, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-new-tokens", type=int, default=150)
    ap.add_argument("--prompt", type=str, default=None,
                    help="Single custom prompt (otherwise runs all 3 built-in examples)")
    ap.add_argument("--example", type=int, default=None, choices=[1, 2, 3],
                    help="Run only built-in example N (1=verb-aux, 2=case, 3=creative)")
    args = ap.parse_args()

    device = f"cuda:{args.gpu}" if args.gpu is not None else args.device

    if args.prompt:
        examples = [{"label": "Custom Prompt", "prompt": args.prompt, "why": "user-supplied"}]
    elif args.example:
        examples = [TRACE_PROMPTS[args.example - 1]]
    else:
        examples = TRACE_PROMPTS

    for ex in examples:
        run_trace(
            prompt=ex["prompt"],
            label=ex["label"],
            why=ex["why"],
            model_key=args.model,
            mode=args.mode,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )


if __name__ == "__main__":
    main()
