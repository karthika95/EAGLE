#!/usr/bin/env python3
"""
eval_scg.py — Evaluation harness for Samanvaya-Constrained Generation.

Runs all three modes (baseline / constrained / adaptive) on a JSONL eval file,
aggregates metrics, and writes:
  results/eval_{model}_{tag}.json         — per-prompt raw results
  results/summary_table_{tag}.md          — Markdown comparison table

Primary metrics (no reference needed)
--------------------------------------
  LWG Compliance Rate   : parser.compliance_rate() per response
  Intervention Rate     : mask_applied / total_steps
  Opportunistic Save Rate: opportunistic_saves / constrained_steps
  Cross-PPL Ratio       : mean(PPL_SCG) / mean(PPL_baseline)
                          Measures fluency cost of constraint.
                          Computed by teacher-forcing each response through
                          the unconstrained model.
  Tokens/second         : wall-clock throughput

Secondary metrics (only when reference field is non-null)
----------------------------------------------------------
  ChrF                  : character n-gram F-score vs reference

Category breakdown (MT-Bench-Hi)
---------------------------------
  per-category compliance rates reported if "category" field present.
  Headline uses "primary" tier (writing, humanities, roleplay, extraction).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import sacrebleu

from scg_generate import MODELS, compute_cross_ppl, generate

WORKSPACE   = Path(__file__).parent
EVAL_FILE   = WORKSPACE / "hindi_eval_50.jsonl"
RESULTS_DIR = WORKSPACE / "results"

PRIMARY_TIERS = {"primary"}


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def aggregate(results: list[dict]) -> dict:
    """Aggregate per-prompt result dicts into summary stats."""
    compliances = [r["parser_compliance"] for r in results
                   if r.get("parser_compliance") is not None]
    tps_list    = [r["tokens_per_sec"] for r in results]

    constrained_steps_list = [r["gen_stats"].get("constrained_steps", 0) for r in results]
    opp_saves_list         = [r["gen_stats"].get("opportunistic_saves", 0) for r in results]
    mask_list              = [r["gen_stats"].get("mask_applied", 0) for r in results]
    total_steps_list       = [r["gen_stats"].get("total_steps", 0) for r in results]

    total_steps      = sum(total_steps_list)
    total_constrained = sum(constrained_steps_list)
    total_mask       = sum(mask_list)
    total_opp        = sum(opp_saves_list)

    # Cross-PPL (only where present)
    ppl_values = [r["cross_ppl"] for r in results
                  if r.get("cross_ppl") is not None and r["cross_ppl"] == r["cross_ppl"]]

    # ChrF (only when reference available)
    refs_flat: list[str] = []
    hyps_flat: list[str] = []
    for r in results:
        if r.get("reference"):
            refs_flat.append(r["reference"])
            hyps_flat.append(r["response"])
    chrf_score = 0.0
    if refs_flat:
        chrf_score = sacrebleu.corpus_chrf(hyps_flat, [refs_flat]).score

    return {
        "lwg_compliance_mean":  safe_mean(compliances),
        "tokens_per_sec_mean":  safe_mean(tps_list),
        "intervention_rate":    total_mask / total_steps if total_steps > 0 else 0.0,
        "opp_save_rate":        total_opp / total_constrained if total_constrained > 0 else 0.0,
        "cross_ppl_mean":       safe_mean(ppl_values) if ppl_values else None,
        "chrf":                 round(chrf_score, 2),
        "n_prompts":            len(results),
        "has_reference":        bool(refs_flat),
    }


def aggregate_by_category(all_results: dict[str, list[dict]]) -> dict:
    """Group per-prompt results by category and compute per-category compliance."""
    # Collect all categories
    categories: dict[str, list[str]] = {}  # category → list of tiers
    for mode, results in all_results.items():
        for r in results:
            cat  = r.get("category", "unknown")
            tier = r.get("tier", "unknown")
            categories[cat] = tier

    # Per-category aggregation (baseline compliance only for brevity)
    cat_stats: dict[str, dict] = {}
    for cat in sorted(categories):
        cat_stats[cat] = {"tier": categories[cat]}
        for mode, results in all_results.items():
            mode_results = [r for r in results if r.get("category") == cat]
            comps = [r["parser_compliance"] for r in mode_results
                     if r.get("parser_compliance") is not None]
            cat_stats[cat][mode] = {
                "compliance": safe_mean(comps),
                "n": len(mode_results),
            }
    return cat_stats


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_eval(
    model_key: str,
    modes: list[str],
    max_new_tokens: int,
    device: str,
    limit: int = 0,
    T_G_init: float = 0.5,
    eval_file: Path = EVAL_FILE,
    compute_ppl: bool = True,
) -> dict[str, list[dict]]:
    """Run all modes on the eval set. Returns {mode: [result, ...]}."""

    prompts = []
    with open(eval_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))

    if limit > 0:
        prompts = prompts[:limit]

    print(f"\nRunning eval: model={model_key}, modes={modes}, n={len(prompts)}")

    all_results: dict[str, list[dict]] = {m: [] for m in modes}

    for i, item in enumerate(prompts):
        instruction = item["instruction"]
        reference   = item.get("reference") or ""
        category    = item.get("category", "")
        tier        = item.get("tier", "")

        print(f"\n[{i+1}/{len(prompts)}] [{category}] {instruction[:55]}…")

        # Generate all modes for this prompt
        mode_results = {}
        for mode in modes:
            result = generate(
                prompt=instruction,
                model_key=model_key,
                mode=mode,
                max_new_tokens=max_new_tokens,
                device=device,
                T_G_init=T_G_init,
                workspace=WORKSPACE,
            )
            result["reference"] = reference
            result["category"]  = category
            result["tier"]      = tier
            mode_results[mode]  = result

            compliance = result["parser_compliance"]
            tps        = result["tokens_per_sec"]
            print(f"  [{mode:12s}] compliance={compliance:.2%}  tps={tps:.1f}")

        # Cross-PPL: score all responses under the baseline (unconstrained) model
        if compute_ppl and "baseline" in mode_results:
            base_fp = mode_results["baseline"]["formatted_prompt"]
            for mode in modes:
                ppl = compute_cross_ppl(
                    response_text=mode_results[mode]["response"],
                    formatted_prompt=base_fp,
                    model_key=model_key,
                    device=device,
                )
                mode_results[mode]["cross_ppl"] = ppl

        for mode in modes:
            all_results[mode].append(mode_results[mode])

    return all_results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_results(
    model_key: str,
    all_results: dict[str, list[dict]],
    eval_tag: str = "",
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    suffix   = f"_{eval_tag}" if eval_tag else ""
    out_path = RESULTS_DIR / f"eval_{model_key}{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out_path}")

    summaries: dict[str, dict] = {}
    for mode, results in all_results.items():
        summaries[mode] = aggregate(results)

    # Cross-PPL ratio (SCG / baseline, AdaSCG / baseline)
    base_ppl = summaries.get("baseline", {}).get("cross_ppl_mean")
    ppl_ratios: dict[str, str] = {}
    for mode, s in summaries.items():
        mppl = s.get("cross_ppl_mean")
        if base_ppl and mppl and base_ppl > 0:
            ppl_ratios[mode] = f"{mppl / base_ppl:.3f}"
        elif mppl:
            ppl_ratios[mode] = f"{mppl:.1f}"
        else:
            ppl_ratios[mode] = "N/A"

    # WGTF
    vocab_scan_path = WORKSPACE / MODELS[model_key]["vocab_scan"]
    wgtf = "N/A"
    if vocab_scan_path.exists():
        with open(vocab_scan_path) as f:
            vs = json.load(f)
        wgtf = f"{vs['wgtf']['wgtf']*100:.1f}%"

    has_ref   = any(s.get("has_reference") for s in summaries.values())
    has_cat   = any(r.get("category") for r in next(iter(all_results.values())))
    has_ppl   = any(s.get("cross_ppl_mean") for s in summaries.values())

    # --- Headline table ---
    lines = [
        f"# SCG Evaluation Summary — Model: `{MODELS[model_key]['model_id']}`",
        "",
        f"**WGTF** (Word Group Tokenization Fidelity): {wgtf}",
        "> Fraction of Samanvaya vocabulary words that are single tokens in the tokenizer.",
        "",
        "## Overall Results",
        "",
    ]

    if has_ppl:
        lines += [
            "| Method | LWG Compliance | Δ vs Base | Tokens/s | IR | OSR | PPL (×base) |",
            "|--------|---------------|-----------|----------|-----|-----|-------------|",
        ]
        base_cr = summaries.get("baseline", {}).get("lwg_compliance_mean", 0)
        for mode, s in summaries.items():
            delta = f"+{(s['lwg_compliance_mean']-base_cr)*100:.1f} pp" if mode != "baseline" else "—"
            lines.append(
                f"| {mode:12s} "
                f"| {s['lwg_compliance_mean']*100:.1f}% "
                f"| {delta} "
                f"| {s['tokens_per_sec_mean']:.1f} "
                f"| {s['intervention_rate']*100:.1f}% "
                f"| {s['opp_save_rate']*100:.1f}% "
                f"| {ppl_ratios[mode]} |"
            )
    else:
        lines += [
            "| Method | LWG Compliance | Tokens/s | IR | OSR |",
            "|--------|---------------|----------|-----|-----|",
        ]
        for mode, s in summaries.items():
            lines.append(
                f"| {mode:12s} "
                f"| {s['lwg_compliance_mean']*100:.1f}% "
                f"| {s['tokens_per_sec_mean']:.1f} "
                f"| {s['intervention_rate']*100:.1f}% "
                f"| {s['opp_save_rate']*100:.1f}% |"
            )

    lines.append("")
    lines.append(f"*n_prompts = {summaries[list(summaries)[0]]['n_prompts']}*")

    if has_ref:
        lines += [
            "",
            "| Method | ChrF |",
            "|--------|------|",
        ]
        for mode, s in summaries.items():
            lines.append(f"| {mode:12s} | {s['chrf']:.1f} |")

    # --- Per-category breakdown ---
    if has_cat:
        cat_stats = aggregate_by_category(all_results)
        lines += [
            "",
            "## Per-Category LWG Compliance",
            "",
            "| Category | Tier | n | Baseline | SCG (constrained) | AdaSCG (adaptive) |",
            "|----------|------|---|----------|-------------------|-------------------|",
        ]
        for cat, cs in cat_stats.items():
            tier_lbl = cs["tier"]
            n = cs.get("baseline", {}).get("n", "?")
            b = cs.get("baseline",    {}).get("compliance", 0)
            c = cs.get("constrained", {}).get("compliance", 0)
            a = cs.get("adaptive",    {}).get("compliance", 0)
            lines.append(
                f"| {cat:12s} | {tier_lbl:12s} | {n} "
                f"| {b*100:.1f}% | {c*100:.1f}% | {a*100:.1f}% |"
            )

        # Primary-tier headline
        primary_results: dict[str, list[dict]] = {
            m: [r for r in res if r.get("tier") == "primary"]
            for m, res in all_results.items()
        }
        if any(primary_results.values()):
            primary_summaries = {m: aggregate(r) for m, r in primary_results.items()}
            lines += [
                "",
                "### Primary-tier Headline (writing + humanities + roleplay + extraction)",
                "",
                "| Method | LWG Compliance | Δ vs Base | IR | OSR |",
                "|--------|---------------|-----------|-----|-----|",
            ]
            base_cr_primary = primary_summaries.get("baseline", {}).get("lwg_compliance_mean", 0)
            for mode, s in primary_summaries.items():
                delta = f"+{(s['lwg_compliance_mean']-base_cr_primary)*100:.1f} pp" if mode != "baseline" else "—"
                lines.append(
                    f"| {mode:12s} "
                    f"| {s['lwg_compliance_mean']*100:.1f}% "
                    f"| {delta} "
                    f"| {s['intervention_rate']*100:.1f}% "
                    f"| {s['opp_save_rate']*100:.1f}% |"
                )
            lines.append(f"\n*n_primary = {primary_summaries[list(primary_summaries)[0]]['n_prompts']}*")

    md_path = RESULTS_DIR / f"summary_table{suffix}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {md_path}")
    print("\n" + "\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SCG evaluation harness")
    parser.add_argument("--model", choices=list(MODELS), default="airavata")
    parser.add_argument("--modes", nargs="+",
                        choices=["baseline", "constrained", "adaptive"],
                        default=["baseline", "constrained", "adaptive"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device",  type=str, default="cuda")
    parser.add_argument("--gpu",     type=int, default=None,
                        help="CUDA device index")
    parser.add_argument("--limit",   type=int, default=0,
                        help="Limit to first N prompts (0 = all)")
    parser.add_argument("--T-G",     type=float, default=0.5,
                        help="Initial AdaSD entropy gate threshold")
    parser.add_argument("--eval-file", type=str, default=None,
                        help="Path to evaluation JSONL (default: hindi_eval_50.jsonl)")
    parser.add_argument("--no-ppl",  action="store_true",
                        help="Skip cross-perplexity computation (faster)")
    args = parser.parse_args()

    device    = f"cuda:{args.gpu}" if args.gpu is not None else args.device
    eval_file = Path(args.eval_file) if args.eval_file else EVAL_FILE
    eval_tag  = eval_file.stem if eval_file != EVAL_FILE else ""

    all_results = run_eval(
        model_key=args.model,
        modes=args.modes,
        max_new_tokens=args.max_new_tokens,
        device=device,
        limit=args.limit,
        T_G_init=args.T_G,
        eval_file=eval_file,
        compute_ppl=not args.no_ppl,
    )

    write_results(model_key=args.model, all_results=all_results, eval_tag=eval_tag)


if __name__ == "__main__":
    main()
