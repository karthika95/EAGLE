"""
Comprehensive Draft Token Statistics Evaluation

Evaluates and compares original SAMD vs word-group-aware SAMD on full MT_bench dataset.
Tracks draft token generation, acceptance rates, and efficiency metrics for both 
static SAM and dynamic SAM individually and combined.

Usage:
python3 evaluation/evaluate_draft_statistics.py \
    --model-path ai4bharat/Airavata \
    --bench-name mt_bench \
    --sam-path downloads/processed_file.pkl \
    --tree-model-path downloads/airavata_bs1/state_20 \
    --wordgroup-sam-path preprocess/sam_airavata_wikichat.pkl \
    --output-dir evaluation/statistics_results
"""

import argparse
import os
import json
import torch
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
from tqdm import tqdm

from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template
from fastchat.utils import str_to_torch_dtype
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from samd import SamdConfig, SamdModel, SamdGenerationConfig, DraftModel, load_sam
from samd.draft import CandidateType
from samd.sam.wordgroup_dyn_sam import WordGroupAwareDynSAM
from samd.wordgroup.grouping import boundaries_for_token_ids
from samd.draft_statistics import BatchStatistics, SequenceStatistics


class WordGroupAwareDraftModel(DraftModel):
    """Draft model that respects word group boundaries for Hindi/Indic languages"""
    
    def __init__(
        self,
        config,
        sam_static=None,
        lm=None,
        dtype=None,
        device=None,
        tokenizer=None,
        disable_dyn: bool = False,
        disable_eagle: bool = False,
    ):
        sam_dyn = WordGroupAwareDynSAM(config.n_predicts)
        self.disable_dyn = disable_dyn
        self.disable_eagle = disable_eagle
        self.tokenizer = tokenizer
        
        super().__init__(
            config=config,
            sam_dyn=sam_dyn,
            sam_static=sam_static,
            lm=lm,
            dtype=dtype,
            device=device
        )
    
    def update(self, tokens=None, last_hidden_states=None, tree_tokens=None, tree_logits=None):
        if tokens is not None:
            tokens_list = tokens.tolist()
            
            # Compute word boundaries for the accepted tokens
            text = self.tokenizer.decode(tokens_list, skip_special_tokens=True)
            boundaries = boundaries_for_token_ids(
                self.tokenizer,
                text,
                tokens_list
            )
            
            # Add tokens with boundaries to dynamic SAM
            if not self.disable_dyn:
                self.sam_dyn.add_tokens(tokens_list, boundaries)
            
            # Transfer tokens to static SAM
            if self.sam_static is not None:
                self.sam_static.transfer_tokens(tokens_list)
        
        if not self.disable_eagle:
            self.tree_model.update(
                tokens=tokens,
                last_hidden_states=last_hidden_states,
                tree_tokens=tree_tokens,
                tree_logits=tree_logits,
            )
    
    def lookup(self, start_token: int, step: int = 0):
        """Word-group-aware lookup that chooses between dynamic SAM, static SAM, or tree drafting"""
        counter = 0
        
        # Try dynamic SAM first
        if not self.disable_dyn:
            index_dyn, match_dyn, counter = self.sam_dyn.lookup(start_token, step, counter)
        else:
            index_dyn, match_dyn = -1, float('-inf')
        
        # Try static SAM
        if self.sam_static is not None:
            index_static, match_static, counter = self.sam_static.lookup(start_token, step, counter)
            match_static -= self.len_bias
        else:
            index_static, match_static = -1, float('-inf')
        
        # Choose best match between dynamic and static
        best_match = max(match_dyn, match_static)
        threshold_met = best_match >= self.len_threshold
        
        # If threshold met or EAGLE disabled, use SAM drafting
        if threshold_met or self.disable_eagle:
            # Prefer dynamic SAM if it has equal or better match
            use_dynamic = (not self.disable_dyn) and (match_dyn >= match_static)
            if use_dynamic:
                seq = self.sam_dyn.gen_draft(index_dyn, start_token)
                seqtype = "dynamic"
            else:
                seq = self.sam_static.gen_draft(index_static, start_token)
                seqtype = "static"

            # Normalize different gen_draft return styles.
            # Some SAMs return just token list, others return (token_list, meta).
            seq_tokens = seq[0] if isinstance(seq, tuple) and len(seq) >= 1 else seq
            if isinstance(seq_tokens, torch.Tensor):
                seq_tokens = seq_tokens.tolist()

            if isinstance(seq_tokens, (list, tuple)):
                n_draft = max(0, len(seq_tokens) - 1)
            else:
                n_draft = 0

            return (CandidateType.sequence, seqtype, seq_tokens, {}, n_draft)
        
        # Fall back to tree/EAGLE drafting
        tree_tokens, buffers_kwargs = self.tree_model.gen_draft(start_token)
        n_draft = max(0, len(tree_tokens) - 1)
        return (CandidateType.tree, "tree", tree_tokens, buffers_kwargs, n_draft)


def load_wordgroup_sam(path: str):
    """Load word-group-aware SAM from pickle format"""
    if not path or not os.path.exists(path):
        print(f"Warning: SAM file not found at {path}")
        return None
    
    import pickle
    import time
    start = time.perf_counter()
    
    print(f"Loading word-group-aware SAM from {path}...")
    with open(path, "rb") as f:
        sam = pickle.load(f)
    
    end = time.perf_counter()
    print(f"Loaded SAM in {end - start:.2f} seconds")
    return sam


def setup_model_and_draft(
    model_path: str,
    sam_path: Optional[str],
    tree_model_path: str,
    dtype_str: str,
    device_map: str,
    samd_config: SamdConfig,
    use_wordgroup: bool = False,
    wordgroup_sam_path: Optional[str] = None,
    disable_dyn: bool = False,
    disable_eagle: bool = False,
):
    """Setup language model and draft model"""
    
    print(f"\nLoading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=str_to_torch_dtype(dtype_str),
        low_cpu_mem_usage=True,
        device_map=device_map,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = next(model.lm_head.parameters()).device
    
    # Load SAM and create appropriate draft model
    if use_wordgroup:
        print("Using word-group-aware SAM...")
        sam = load_wordgroup_sam(wordgroup_sam_path)
        
        # Create word-group-aware draft model
        draft = WordGroupAwareDraftModel(
            samd_config,
            sam_static=sam,
            lm=model,
            dtype=str_to_torch_dtype(dtype_str),
            device=device,
            tokenizer=tokenizer,
            disable_dyn=disable_dyn,
            disable_eagle=disable_eagle,
        )
    else:
        print("Using standard SAM...")
        sam = load_sam(sam_path) if sam_path else None
        
        # Create standard draft model
        draft = DraftModel(
            samd_config,
            sam_static=sam,
            lm=model,
            dtype=str_to_torch_dtype(dtype_str),
            device=device,
        )
    
    # Create SAMD model
    samd_model = SamdModel(
        samd_config,
        model,
        draft,
        tokenizer.eos_token_id,
        str_to_torch_dtype(dtype_str),
        device,
        tokenizer=tokenizer,
    )
    
    return samd_model, tokenizer


@torch.inference_mode()
def evaluate_batch(
    model: SamdModel,
    tokenizer: PreTrainedTokenizer,
    questions: List[Dict],
    model_id: str,
    max_new_tokens: int,
    collect_stats: bool = True,
) -> BatchStatistics:
    """Evaluate a batch of questions and collect statistics"""
    
    model.eval()
    batch_stats = BatchStatistics(model_id=model_id, dataset_name="mt_bench")
    
    for seq_id, question in enumerate(tqdm(questions, desc=f"Evaluating {model_id}")):
        try:
            seq_stats = evaluate_single(
                model,
                tokenizer,
                question,
                seq_id,
                max_new_tokens,
                collect_stats=collect_stats,
            )
            batch_stats.add_sequence(seq_stats)
        except Exception as e:
            print(f"\nError processing question {seq_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    batch_stats.compute_metrics()
    return batch_stats


@torch.inference_mode()
def evaluate_single(
    model: SamdModel,
    tokenizer: PreTrainedTokenizer,
    question: Dict,
    sequence_id: int,
    max_new_tokens: int,
    collect_stats: bool = True,
) -> SequenceStatistics:
    """Evaluate a single question and collect detailed statistics"""
    
    conv = get_conversation_template("tulu")
    qs = question["turns"][0]  # First turn only
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    conv.stop_str = "</s>"
    
    prompt = conv.get_prompt()
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    # Generate with statistics
    if collect_stats and hasattr(model, 'generate_with_stats'):
        outputs = model.generate_with_stats(
            inputs.input_ids,
            generation_config=SamdGenerationConfig(
                max_new_tokens=max_new_tokens,
                max_cache_len=model.lm.config.max_position_embeddings,
                greedy=True,
            ),
            sequence_id=sequence_id,
        )
        return outputs.sequence_stats
    else:
        # Fallback to standard generation
        outputs = model.generate(
            inputs.input_ids,
            generation_config=SamdGenerationConfig(
                max_new_tokens=max_new_tokens,
                max_cache_len=model.lm.config.max_position_embeddings,
                greedy=True,
            ),
        )
        # Create basic stats for this sequence
        seq_stats = SequenceStatistics(
            sequence_id=sequence_id,
            total_steps=outputs.decode_steps,
            total_output_tokens=outputs.decode_tokens,
        )
        return seq_stats


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate draft token statistics for SAM-D models"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the language model",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="tulu",
        choices=["vicuna", "llama3", "tulu"],
        help="Model type for template",
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="Benchmark dataset name",
    )
    parser.add_argument(
        "--sam-path",
        type=str,
        default=None,
        help="Path to static SAM file",
    )
    parser.add_argument(
        "--wordgroup-sam-path",
        type=str,
        default=None,
        help="Path to word-group-aware SAM file",
    )
    parser.add_argument(
        "--tree-model-path",
        type=str,
        default=None,
        help="Path to EAGLE/tree model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation/statistics_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Data type",
    )
    parser.add_argument(
        "--samd-n-predicts",
        type=int,
        default=15,
        help="Number of tokens to predict with SAM",
    )
    parser.add_argument(
        "--samd-len-threshold",
        type=int,
        default=3,
        help="Minimum match length threshold",
    )
    parser.add_argument(
        "--samd-len-bias",
        type=int,
        default=2,
        help="Bias for static SAM",
    )
    parser.add_argument(
        "--question-count",
        type=int,
        default=None,
        help="Number of questions to evaluate (None for all)",
    )
    parser.add_argument(
        "--disable_dyn",
        action="store_true",
        help="Disable dynamic SAM (only for word-group variant)",
    )
    parser.add_argument(
        "--disable_eagle",
        action="store_true",
        help="Disable EAGLE/tree fallback (only for word-group variant)",
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load questions
    question_file = f"evaluation/data/{args.bench_name}/question.jsonl"
    questions = load_questions(question_file, None, None)
    
    if args.question_count:
        questions = questions[:args.question_count]
    
    print(f"Loaded {len(questions)} questions from {args.bench_name}")
    
    # Setup device
    if args.num_gpus == 1:
        device_map = "cuda"
    else:
        device_map = "auto"
    
    # SAMD configuration (same for both versions)
    samd_config = SamdConfig(
        n_predicts=args.samd_n_predicts,
        tree_method="eagle2",
        tree_model_path=args.tree_model_path,
        len_threshold=args.samd_len_threshold,
        len_bias=args.samd_len_bias,
    )
    
    # Test both versions
    results = {}
    
    # 1. Original SAMD
    print("\n" + "="*80)
    print("EVALUATING: Original SAMD (Standard SAM)")
    print("="*80)
    
    samd_original, tokenizer = setup_model_and_draft(
        args.model_path,
        args.sam_path,
        args.tree_model_path,
        args.dtype,
        device_map,
        samd_config,
        use_wordgroup=False,
        disable_dyn=args.disable_dyn,
        disable_eagle=args.disable_eagle,
    )
    
    batch_stats_original = evaluate_batch(
        samd_original,
        tokenizer,
        questions,
        "SAMD-Original",
        args.max_new_tokens,
        collect_stats=True,
    )
    
    batch_stats_original.print_report()
    results["original_samd"] = batch_stats_original.to_dict()
    
    # 2. Word-Group-Aware SAMD
    if args.wordgroup_sam_path:
        print("\n" + "="*80)
        print("EVALUATING: Word-Group-Aware SAMD")
        print("="*80)
        
        samd_wordgroup, tokenizer = setup_model_and_draft(
            args.model_path,
            args.wordgroup_sam_path,
            args.tree_model_path,
            args.dtype,
            device_map,
            samd_config,
            use_wordgroup=True,
            wordgroup_sam_path=args.wordgroup_sam_path,
            disable_dyn=args.disable_dyn,
            disable_eagle=args.disable_eagle,
        )
        
        batch_stats_wordgroup = evaluate_batch(
            samd_wordgroup,
            tokenizer,
            questions,
            "SAMD-WordGroup",
            args.max_new_tokens,
            collect_stats=True,
        )
        
        batch_stats_wordgroup.print_report()
        results["wordgroup_samd"] = batch_stats_wordgroup.to_dict()
    
    # Generate comparison report
    print("\n" + "="*80)
    print("COMPARISON REPORT")
    print("="*80)
    
    if "wordgroup_samd" in results:
        orig = results["original_samd"]
        wg = results["wordgroup_samd"]
        
        print(f"\nAcceptance Rate Improvement:")
        print(f"  Static SAM:  {orig['static']['acceptance_rate']:.2%} → {wg['static']['acceptance_rate']:.2%} "
              f"({(wg['static']['acceptance_rate'] - orig['static']['acceptance_rate'])*100:+.2f}pp)")
        print(f"  Dynamic SAM: {orig['dynamic']['acceptance_rate']:.2%} → {wg['dynamic']['acceptance_rate']:.2%} "
              f"({(wg['dynamic']['acceptance_rate'] - orig['dynamic']['acceptance_rate'])*100:+.2f}pp)")
        print(f"  Tree/EAGLE:  {orig['tree']['acceptance_rate']:.2%} → {wg['tree']['acceptance_rate']:.2%} "
              f"({(wg['tree']['acceptance_rate'] - orig['tree']['acceptance_rate'])*100:+.2f}pp)")
        print(f"  Overall:     {orig['overall_acceptance_rate']:.2%} → {wg['overall_acceptance_rate']:.2%} "
              f"({(wg['overall_acceptance_rate'] - orig['overall_acceptance_rate'])*100:+.2f}pp)")
        
        print(f"\nSpeedup Comparison:")
        print(f"  Original:   {orig['speedup']['mean']:.4f}x (±{orig['speedup']['std']:.4f})")
        print(f"  WordGroup:  {wg['speedup']['mean']:.4f}x (±{wg['speedup']['std']:.4f})")
        print(f"  Improvement: {(wg['speedup']['mean'] - orig['speedup']['mean'])*100:+.2f}%")
    
    # Save results
    results_file = output_dir / f"statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # Also save configuration
    config_file = output_dir / f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    config_dict = {
        "model_path": args.model_path,
        "bench_name": args.bench_name,
        "num_questions": len(questions),
        "max_new_tokens": args.max_new_tokens,
        "samd_n_predicts": args.samd_n_predicts,
        "samd_len_threshold": args.samd_len_threshold,
        "samd_len_bias": args.samd_len_bias,
        "sam_path": args.sam_path,
        "wordgroup_sam_path": args.wordgroup_sam_path,
        "tree_model_path": args.tree_model_path,
    }
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Config saved to {config_file}")


if __name__ == "__main__":
    main()
