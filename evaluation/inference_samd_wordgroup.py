"""Generate answers with word-group-aware SAM for Hindi models.

Usage:
python3 evaluation/inference_samd_wordgroup.py \
    --model-type tulu \
    --model-path ai4bharat/Airavata \
    --model-id airavata-samd-wordgroup \
    --sam_path downloads/processed_file.pkl \
    --bench-name mt_bench
"""
import argparse
import torch
import os
from typing import Optional, Dict, Any

from fastchat.utils import str_to_torch_dtype
from evaluation.eval import run_evals, reorg_answer_files
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from samd import SamdConfig, SamdModel, SamdGenerationConfig, DraftModel
from samd.draft import CandidateType
from samd.wordgroup_sam import WordGroupAwareSAM
from samd.sam.wordgroup_dyn_sam import WordGroupAwareDynSAM
from samd.wordgroup.grouping import boundaries_for_token_ids


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
            n_draft = (max(0, len(seq) - 1) if isinstance(seq, (list, tuple)) else 0)
            return (CandidateType.sequence, seqtype, seq, {}, n_draft)
        
        # Fall back to tree/EAGLE drafting
        tree_tokens, buffers_kwargs = self.tree_model.gen_draft(start_token)
        n_draft = max(0, len(tree_tokens) - 1)
        return (CandidateType.tree, "tree", tree_tokens, buffers_kwargs, n_draft)


def load_wordgroup_sam(path: str):
    """Load SAM from pickle or PyTorch format (auto-detect)"""
    
    if not path or not os.path.exists(path):
        print(f"Warning: SAM file not found at {path}")
        return None
    
    print(f"Loading word-group-aware SAM from {path} (pickle format)...")
    import pickle
    import time
    start = time.perf_counter()
    
    with open(path, "rb") as f:
        sam = pickle.load(f)
    
    end = time.perf_counter()
    print(f"Loaded SAM in {end - start:.2f} seconds ({(end-start)/60:.1f} minutes)")
    
    return sam


def samd_forward(
    inputs, 
    model: SamdModel, 
    tokenizer: PreTrainedTokenizer, 
    max_new_tokens: int, 
    temperature: float = 0.0,
    do_sample: bool = False
):
    """Forward function for SAMD generation with word-group awareness"""
    max_cache_len = model.lm.config.max_position_embeddings
    input_ids = inputs.input_ids
    outputs = model.generate(
        input_ids,
        generation_config=SamdGenerationConfig(
            max_new_tokens=max_new_tokens,
            max_cache_len=max_cache_len,
            greedy=not do_sample,
            temperature=temperature
        ),
    )
    output_ids = outputs.output_ids
    new_token = outputs.decode_tokens
    step = outputs.decode_steps
    accept_length_list = outputs.accept_length_per_step
    return output_ids, new_token, step, accept_length_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="Override template (if None, uses model-type)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["vicuna", "llama3", "tulu"],
        help="Model type for conversation template"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the language model"
    )
    parser.add_argument("--model-id", type=str, required=True, help="Model identifier for results")
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end",
        type=int,
        help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for sampling.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )
    parser.add_argument(
        "--samd_n_predicts",
        type=int,
        default=15,
        help="Number of tokens to predict with SAM"
    )
    parser.add_argument(
        "--sam_path",
        type=str,
        default=None,
        help="Path to word-group-aware SAM file (.pkl or .pt)"
    )
    parser.add_argument(
        "--samd_len_threshold",
        type=int,
        default=5,
        help="Minimum match length threshold for SAM acceptance"
    )
    parser.add_argument(
        "--samd_len_bias",
        type=int,
        default=5,
        help="Bias for static SAM comparison"
    )
    parser.add_argument(
        "--samd_tree_path",
        type=str,
        default=None,
        help="Path to tree data (optional)"
    )
    parser.add_argument("--tree_method", type=str, default="eagle2", help="Tree drafting method")
    parser.add_argument("--tree_model_path", type=str, default=None, help="Path to EAGLE/tree model")
    parser.add_argument("--attn_implementation", type=str, default="sdpa", help="Attention implementation")
    parser.add_argument("--disable_dyn", action="store_true", help="Disable dynamic SAM")
    parser.add_argument("--disable_eagle", action="store_true", help="Disable EAGLE/tree fallback")
    
    args = parser.parse_args()

    # Use model-type as template if not specified
    if args.template is None:
        args.template = args.model_type

    question_file = f"evaluation/data/{args.bench_name}/question.jsonl"

    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"evaluation/data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")
    print(f"Model: {args.model_path}")
    print(f"Template: {args.template}")
    print(f"SAM path: {args.sam_path}")
    print(f"len_bias: {args.samd_len_bias}")
    print(f"len_threshold: {args.samd_len_threshold}")
    print(f"n_predicts: {args.samd_n_predicts}")
    print(f"disable_dyn: {args.disable_dyn}")
    print(f"disable_eagle: {args.disable_eagle}")
    
    if args.num_gpus_total == 1:
        device_map = "cuda"
    else:
        device_map = "auto"

    print(f"\nLoading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map=device_map,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Handle stop tokens for different model types
    if args.model_type == "llama3":
        stop_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        assert isinstance(stop_token_id, int)
    else:
        stop_token_id = None

    device = next(model.lm_head.parameters()).device
    
    # Load word-group-aware SAM
    sam = load_wordgroup_sam(args.sam_path)
    
    # Configure SAMD
    samd_config = SamdConfig(
        n_predicts=args.samd_n_predicts,
        tree_method=args.tree_method,
        tree_model_path=args.tree_model_path,
        len_threshold=args.samd_len_threshold,
        len_bias=args.samd_len_bias,
        tree_path=args.samd_tree_path,
    )
    
    # Create word-group-aware draft model
    draft = WordGroupAwareDraftModel(
        samd_config, 
        sam_static=sam,
        lm=model,
        dtype=str_to_torch_dtype(args.dtype),
        device=device,
        tokenizer=tokenizer,
        disable_dyn=args.disable_dyn,
        disable_eagle=args.disable_eagle,
    )
    
    # Create SAMD model
    samd_model = SamdModel(
        samd_config, 
        model, 
        draft, 
        tokenizer.eos_token_id,
        str_to_torch_dtype(args.dtype),
        device, 
        stop_token_id=stop_token_id,
        tokenizer=tokenizer,
    )

    if args.temperature > 0:
        do_sample = True
    else:
        do_sample = False

    print(f"\nStarting evaluation on {args.bench_name}...")
    run_evals[args.template](
        model=samd_model,
        tokenizer=tokenizer,
        forward_func=samd_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        temperature=args.temperature,
        do_sample=do_sample,
    )

    reorg_answer_files[args.template](answer_file)
    print(f"\nEvaluation complete! Results saved to {answer_file}")
