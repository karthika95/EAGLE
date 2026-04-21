"""
Quality comparison: Airavata baseline vs SAM decoding vs WordGroupAware SAM.

Metrics computed:
  1. BLEU          - Bilingual Evaluation Understudy (n-gram precision)
  2. chrF / chrF++ - Character n-gram F-score (best for Hindi/Devanagari)
  3. ROUGE-1/2/L   - Recall-Oriented Understudy for Gisting Evaluation
  4. BERTScore     - Contextual similarity via multilingual BERT
  5. Semantic Sim  - Cosine similarity via LaBSE / paraphrase-multilingual-MiniLM

Usage:
    python -m evaluation.compare_quality [--output results/quality_report.json]
"""

import json
import argparse
import os
import numpy as np
from pathlib import Path

import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn

# ─── File paths ──────────────────────────────────────────────────────────────

BASE_DIR = Path("evaluation/data/mt_bench/model_answer")
FILES = {
    "Airavata-7B (baseline)":       BASE_DIR / "airavata-7b-run2.jsonl",
    "SAMD bias=0":                   BASE_DIR / "airavata-samd-bias0.jsonl",
    "WordGroupAware-SAMD bias=0":    BASE_DIR / "airavata-samd-wordgroup-bias0.jsonl",
    "Airavata-Eagle2 (temperature=0.0)": BASE_DIR / "airavata-eagle2-temperature-0.0.jsonl"
}
REFERENCE_KEY = "Airavata-7B (baseline)"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> dict[int, dict]:
    """Return {question_id: record} for all entries in a JSONL file."""
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records[rec["question_id"]] = rec
    return records


def get_turns(record: dict) -> list[str]:
    """Extract the list of text turns from a record's first choice."""
    return record["choices"][0]["turns"]


def align_pairs(ref_data: dict, hyp_data: dict) -> tuple[list[str], list[str]]:
    """Return parallel lists of (reference, hypothesis) texts across all turns."""
    refs, hyps = [], []
    for qid, ref_rec in sorted(ref_data.items()):
        if qid not in hyp_data:
            continue
        ref_turns = get_turns(ref_rec)
        hyp_turns = get_turns(hyp_data[qid])
        for r, h in zip(ref_turns, hyp_turns):
            if r.strip() and h.strip():
                refs.append(r.strip())
                hyps.append(h.strip())
    return refs, hyps


def compute_bleu(refs: list[str], hyps: list[str]) -> dict:
    """Corpus-level BLEU via sacrebleu."""
    result = sacrebleu.corpus_bleu(hyps, [refs])
    return {"bleu": round(result.score, 4)}


def compute_chrf(refs: list[str], hyps: list[str]) -> dict:
    """Corpus-level chrF and chrF++ via sacrebleu."""
    chrf   = sacrebleu.corpus_chrf(hyps, [refs])
    chrfpp = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)
    return {
        "chrF":   round(chrf.score, 4),
        "chrF++": round(chrfpp.score, 4),
    }


def compute_rouge(refs: list[str], hyps: list[str]) -> dict:
    """Average ROUGE-1, ROUGE-2, ROUGE-L (F1) over all pairs."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    r1_f, r2_f, rl_f = [], [], []
    for r, h in zip(refs, hyps):
        scores = scorer.score(r, h)
        r1_f.append(scores["rouge1"].fmeasure)
        r2_f.append(scores["rouge2"].fmeasure)
        rl_f.append(scores["rougeL"].fmeasure)
    return {
        "rouge1_f1": round(np.mean(r1_f), 4),
        "rouge2_f1": round(np.mean(r2_f), 4),
        "rougeL_f1": round(np.mean(rl_f), 4),
    }


def compute_bertscore(refs: list[str], hyps: list[str], batch_size: int = 64) -> dict:
    """
    BERTScore with multilingual BERT.
    Using 'bert-base-multilingual-cased' which covers Hindi/Devanagari.
    """
    print("    Computing BERTScore (this may take a moment)...")
    P, R, F1 = bert_score_fn(
        hyps, refs,
        model_type="bert-base-multilingual-cased",
        lang="hi",
        batch_size=batch_size,
        verbose=False,
    )
    return {
        "bertscore_precision": round(P.mean().item(), 4),
        "bertscore_recall":    round(R.mean().item(), 4),
        "bertscore_f1":        round(F1.mean().item(), 4),
    }


def compute_semantic_similarity(refs: list[str], hyps: list[str]) -> dict:
    """
    Cosine similarity using paraphrase-multilingual-MiniLM-L12-v2 (supports Hindi).
    Falls back gracefully if model download fails.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        print("    Computing semantic similarity (downloading/loading model)...")
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        ref_emb = model.encode(refs, batch_size=64, convert_to_tensor=True, show_progress_bar=False)
        hyp_emb = model.encode(hyps, batch_size=64, convert_to_tensor=True, show_progress_bar=False)
        cos_sim = torch.nn.functional.cosine_similarity(ref_emb, hyp_emb, dim=1)
        return {"semantic_similarity": round(cos_sim.mean().item(), 4)}
    except Exception as e:
        print(f"    Semantic similarity skipped: {e}")
        return {"semantic_similarity": None}


def compute_per_category(ref_data: dict, hyp_data: dict) -> dict[str, dict]:
    """Compute chrF per MT-Bench category for fine-grained analysis."""
    from collections import defaultdict
    by_cat: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))

    for qid, ref_rec in sorted(ref_data.items()):
        if qid not in hyp_data:
            continue
        cat = ref_rec.get("category", "unknown")
        ref_turns = get_turns(ref_rec)
        hyp_turns = get_turns(hyp_data[qid])
        for r, h in zip(ref_turns, hyp_turns):
            if r.strip() and h.strip():
                by_cat[cat][0].append(r.strip())
                by_cat[cat][1].append(h.strip())

    cat_scores = {}
    for cat, (refs, hyps) in sorted(by_cat.items()):
        chrf = sacrebleu.corpus_chrf(hyps, [refs])
        bleu = sacrebleu.corpus_bleu(hyps, [refs])
        cat_scores[cat] = {
            "n_turns": len(refs),
            "chrF":    round(chrf.score, 4),
            "bleu":    round(bleu.score, 4),
        }
    return cat_scores


def print_summary(name: str, metrics: dict, per_cat: dict):
    w = 60
    print(f"\n{'─'*w}")
    print(f"  {name}")
    print(f"{'─'*w}")

    print(f"  {'Metric':<30} {'Score':>10}")
    print(f"  {'─'*38}")
    flat = {}
    for group in metrics.values():
        flat.update(group)
    for k, v in flat.items():
        display = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"  {k:<30} {display:>10}")

    print(f"\n  Per-category chrF / BLEU:")
    print(f"  {'Category':<20} {'turns':>6}  {'chrF':>8}  {'BLEU':>8}")
    print(f"  {'─'*46}")
    for cat, s in per_cat.items():
        print(f"  {cat:<20} {s['n_turns']:>6}  {s['chrF']:>8.4f}  {s['bleu']:>8.4f}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Quality comparison of speculative decoding outputs")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save JSON report (e.g. results/quality_report.json)")
    parser.add_argument("--skip-bertscore", action="store_true",
                        help="Skip BERTScore (slow if no GPU)")
    parser.add_argument("--skip-semsim", action="store_true",
                        help="Skip semantic similarity (requires model download)")
    args = parser.parse_args()

    print("Loading reference data...")
    ref_data = load_jsonl(FILES[REFERENCE_KEY])

    all_results = {}

    candidates = {k: v for k, v in FILES.items() if k != REFERENCE_KEY}
    for model_name, path in candidates.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"  vs reference: {REFERENCE_KEY}")
        print(f"  file: {path}")

        hyp_data = load_jsonl(path)
        refs, hyps = align_pairs(ref_data, hyp_data)
        print(f"  Aligned pairs: {len(refs)}")

        metrics = {}

        print("  Computing BLEU...")
        metrics["BLEU"] = compute_bleu(refs, hyps)

        print("  Computing chrF / chrF++...")
        metrics["chrF"] = compute_chrf(refs, hyps)

        print("  Computing ROUGE...")
        metrics["ROUGE"] = compute_rouge(refs, hyps)

        if not args.skip_bertscore:
            metrics["BERTScore"] = compute_bertscore(refs, hyps)
        else:
            print("  BERTScore skipped.")

        if not args.skip_semsim:
            metrics["SemanticSim"] = compute_semantic_similarity(refs, hyps)
        else:
            print("  Semantic similarity skipped.")

        per_cat = compute_per_category(ref_data, hyp_data)

        print_summary(model_name, metrics, per_cat)
        all_results[model_name] = {"metrics": metrics, "per_category": per_cat}

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nReport saved to {args.output}")

    print("\n\nDone.")


if __name__ == "__main__":
    main()
