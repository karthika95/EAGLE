
import os
import sys
import argparse
import pickle
import torch
import time
import csv
from datetime import datetime
from typing import Optional, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from samd import SamdConfig, SamdModel, SamdGenerationConfig, DraftModel
from samd.draft import CandidateType
from samd.wordgroup_sam import WordGroupAwareSAM
from samd.sam.wordgroup_dyn_sam import WordGroupAwareDynSAM
from samd.sam.torch_static_sam import TorchStaticSAM
from samd.wordgroup.grouping import boundaries_for_token_ids


class WordGroupAwareDraftModel(DraftModel):

    
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
        
        self.static_sam_accepts = 0
        self.dynamic_sam_accepts = 0
        self.eagle_accepts = 0
        self.method_stats: Dict[str, Dict[str, Any]] = {
            "dynamic": {
                "invocations": 0,
                "generated_tokens": 0,
                "accepted_tokens": 0,
                "accept_lengths": [],
            },
            "static": {
                "invocations": 0,
                "generated_tokens": 0,
                "accepted_tokens": 0,
                "accept_lengths": [],
            },
            "tree": {
                "invocations": 0,
                "generated_tokens": 0,
                "accepted_tokens": 0,
                "accept_lengths": [],
            },
        }
        self._last_method: Optional[str] = None
        self._last_generated_tokens: int = 0
        
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
            
            if self.tokenizer:
                decoded_tokens = [self.tokenizer.decode([t], skip_special_tokens=True) for t in tokens_list]
                print(f"\nUpdating with {len(tokens_list)} accepted tokens:")
                print(f"Tokens: {tokens_list}")
                print(f"Decoded: {decoded_tokens}")
                print(f"Combined: '{text}'")
                print(f"Boundaries: {boundaries}")
            else:
                print(f"\nUpdating Dynamic SAM with {len(tokens_list)} accepted tokens: {tokens_list}")
            
            # Add tokens with boundaries to dynamic SAM
            if not self.disable_dyn:
                self.sam_dyn.add_tokens(tokens_list, boundaries)
            
            # Transfer tokens to static SAM (also supports boundaries now)
            if self.sam_static is not None:
                self.sam_static.transfer_tokens(tokens_list)
            
            if self._last_method is not None:
                stats = self.method_stats[self._last_method]
                stats["accepted_tokens"] += len(tokens_list)
                stats["accept_lengths"].append(len(tokens_list))
                self._last_method = None
                self._last_generated_tokens = 0
        
        if not self.disable_eagle:
            self.tree_model.update(
                tokens=tokens,
                last_hidden_states=last_hidden_states,
                tree_tokens=tree_tokens,
                tree_logits=tree_logits,
            )
    
    def lookup(self, start_token: int, step: int = 0):
        print(f"\n Lookup for token {start_token}")

        if self.tokenizer:
            start_text = self.tokenizer.decode([start_token], skip_special_tokens=True)
            print(f"Start token: {start_token} → '{start_text}'")
        
        counter = 0
        if not self.disable_dyn:
            index_dyn, match_dyn, counter = self.sam_dyn.lookup(start_token, step, counter)
            print(f"Dynamic SAM: index={index_dyn}, match_length={match_dyn}")
        else:
            index_dyn, match_dyn = -1, float('-inf')
            print("Dynamic SAM disabled")
        
        index_static, match_static, counter = self.sam_static.lookup(start_token, step, counter)
        match_static -= self.len_bias
        print(f"Static SAM: index={index_static}, match_length={match_static} (after bias)")
        
        best_match = max(match_dyn, match_static)
        threshold_met = best_match >= self.len_threshold

        if threshold_met or self.disable_eagle:
            use_dynamic = (not self.disable_dyn) and (match_dyn >= match_static)

            if use_dynamic:
                print(f"Using DYNAMIC SAM (match_dyn={match_dyn} >= match_static={match_static})")
                self.dynamic_sam_accepts += 1
                seq = self.sam_dyn.gen_draft(index_dyn, start_token)
                chosen_method = "dynamic"
            else:
                print(f"Using STATIC SAM (match_static={match_static} {'>=' if match_static >= match_dyn else '<'} match_dyn={match_dyn})")
                self.static_sam_accepts += 1
                seq = self.sam_static.gen_draft(index_static, start_token)
                chosen_method = "static"
            generated_len = len([t for t in seq if t != 0])
            stats = self.method_stats[chosen_method]
            stats["invocations"] += 1
            stats["generated_tokens"] += generated_len
            self._last_method = chosen_method
            self._last_generated_tokens = generated_len
            
            if self.tokenizer:
                non_zero_seq = [t for t in seq if t != 0]
                draft_tokens_decoded = [self.tokenizer.decode([t], skip_special_tokens=True) for t in non_zero_seq]
                draft_text = self.tokenizer.decode(non_zero_seq, skip_special_tokens=True)
                print(f"Draft: {len(non_zero_seq)} tokens")
                print(f"Token IDs: {non_zero_seq}")
                print(f"Decoded tokens: {draft_tokens_decoded}")
                print(f"Combined text: '{draft_text}'")
            
            return (CandidateType.sequence, chosen_method, seq, {})
        else:
            if self.disable_eagle:
                print("Tree model disabled; resorting to STATIC SAM despite threshold miss")
                self.static_sam_accepts += 1
                seq = self.sam_static.gen_draft(index_static, start_token)
                return (CandidateType.sequence, "static", seq, {})

            print(f" Neither SAM meets threshold (best={best_match} < {self.len_threshold}), using TREE/EAGLE model")
            self.eagle_accepts += 1
            tree_tokens, buffers_kwargs = self.tree_model.gen_draft(start_token)
            tree_len = len(tree_tokens)
            stats = self.method_stats["tree"]
            stats["invocations"] += 1
            stats["generated_tokens"] += tree_len
            self._last_method = "tree"
            self._last_generated_tokens = tree_len
            return (CandidateType.tree, "tree", tree_tokens, buffers_kwargs)


def load_wordgroup_sam(path: str):
    """Load SAM from pickle or PyTorch format (auto-detect)"""
    
    # Auto-detect PyTorch format
    if path.endswith('.pt'):
        # Fast PyTorch loading
        sam = TorchStaticSAM.load_torch(path, device='cpu')
        return sam
    
    # Check if .pt version exists
    # pt_path = path.replace('.pkl', '.pt')
    # if os.path.exists(pt_path):
    #     print(f"Found PyTorch version: {pt_path}")
    #     print(f"Using fast loader (recommended over pickle)")
    #     print(f"Hint: Update --sam_path to {pt_path} to skip this check\n")
    #     sam = TorchStaticSAM.load_torch(pt_path, device='cpu')
    #     return sam
    
    # # Fall back to pickle
    print(f"Loading word-group-aware SAM from {path}...")
    # print(f"(This is slow - consider converting to .pt format)")
    # print(f"Run: python tools/convert_sam_to_torch.py {path} {pt_path}\n")
    start = time.perf_counter()
    
    with open(path, "rb") as f:
        sam = pickle.load(f)
    
    end = time.perf_counter()
    print(f"Loaded SAM in {end - start:.2f} seconds ({(end-start)/60:.1f} minutes)")
    print(f"  States: {len(sam.states)}")
    
    # Handle both tensor and list versions
    if hasattr(sam, '_input_ids_tensor') and sam._input_ids_tensor is not None:
        print(f"  Tokens: {len(sam._input_ids_tensor)}")
    else:
        print(f"  Tokens: {len(sam.input_ids)}")
    
    if hasattr(sam, 'word_boundaries'):
        # Check if it's a tensor or list
        if hasattr(sam, '_word_boundaries_tensor') and sam._word_boundaries_tensor is not None:
            print(f"  Word boundaries: {sam._word_boundaries_tensor.sum().item()}")
        elif sam.word_boundaries:
            print(f"  Word boundaries: {sum(sam.word_boundaries)}")
    
    return sam


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "t", "1", "yes", "y"}:
        return True
    if value in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference with word-group-aware SAM for Hindi"
    )
    parser.add_argument(
        '--model_path', 
        type=str, 
        default=" ",
        help='Path to the language model'
    )
    parser.add_argument(
        '--sam_path', 
        type=str, 
        default="downloads/sam_hindi_wordgroup.pkl",
        help='Path to word-group-aware SAM pickle file'
    )
    parser.add_argument(
        '--samd_n_predicts', 
        type=int, 
        default=15,
        help='Number of tokens to predict (will stop at word boundaries)'
    )
    parser.add_argument(
        '--max_new_tokens', 
        type=int, 
        default=512,
        help='Maximum new tokens to generate'
    )
    parser.add_argument(
        '--max_cache_len', 
        type=int, 
        default=2048,
        help='Maximum cache length'
    )
    parser.add_argument(
        "--tree_method", 
        type=str, 
        default="eagle2",
        choices=["token_recycle", "eagle2"],
        help='Tree method for fallback'
    )
    parser.add_argument(
        "--tree_model_path", 
        type=str, 
        default=" ",
        help='Path to tree model (if using eagle2)'
    )
    parser.add_argument(
        '--dtype', 
        type=str, 
        default='float16', 
        choices=['float16', 'float32'],
        help='Model dtype'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default="cuda", 
        choices=['cuda', 'cpu', 'auto'],
        help='Device to run on'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='Hindi prompt for testing'
    )
    parser.add_argument(
        '--len_threshold',
        type=int,
        default=5,
        help='Minimum match length required to accept SAM drafts'
    )
    parser.add_argument(
        '--len_bias',
        type=int,
        default=5,
        help='Bias subtracted from static SAM matches when comparing to dynamic SAM'
    )
    parser.add_argument(
        '--disable_dyn',
        type=str2bool,
        default=False,
        help='Disable dynamic SAM drafting entirely'
    )
    parser.add_argument(
        '--disable_eagle',
        type=str2bool,
        default=False,
        help='Disable tree/EAGLE fallback drafting'
    )
    parser.add_argument(
        '--results_csv',
        type=str,
        default='results/hindi_wordgroup_runs.csv',
        help='File where run statistics will be appended as CSV rows'
    )
    parser.add_argument(
        '--skip_baseline',
        action='store_true',
        help='Skip baseline generation (speedup will be 0 if skipped)'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode - load SAM once and run multiple inferences'
    )
    
    args = parser.parse_args()
    args.dtype = {
        'float16': torch.float16,
        'float32': torch.float32,
    }[args.dtype]
    
    return args


@torch.inference_mode()
def samd_generate_wordgroup(args, inputs, model, tokenizer, sam):
    
    samd_config = SamdConfig(
        n_predicts=args.samd_n_predicts,
        tree_method=args.tree_method,
        tree_model_path=args.tree_model_path,
        len_threshold=args.len_threshold,
        len_bias=args.len_bias,
    )
    
    draft = WordGroupAwareDraftModel(
        samd_config,
        sam_static=sam,
        lm=model,
        dtype=args.dtype,
        device=args.device,
        tokenizer=tokenizer,
        disable_dyn=args.disable_dyn,
        disable_eagle=args.disable_eagle,
    )
    
    samd_model = SamdModel(
        samd_config,
        model,
        draft,
        tokenizer.eos_token_id,
        args.dtype,
        args.device,
        tokenizer=tokenizer,
    )
    samd_model.eval()
    
    gen_config = SamdGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        max_cache_len=args.max_cache_len,
        greedy=True,
        temperature=0.0
    )
    
    print("\n" + "="*80)
    print("Starting generation with word-group-aware SAM...")
    print("="*80)
    
    st = time.perf_counter()
    outputs = samd_model.generate(**inputs, generation_config=gen_config)
    ed = time.perf_counter()
    
    response = tokenizer.decode(outputs.output_ids[0], skip_special_tokens=True)
    
    print("\n" + "-"*80)
    print("RESULTS")
    print("-"*80)
    generation_time = ed - st
    print(f"Generation time: {generation_time:.2f} seconds")
    print(f"Decode steps: {outputs.decode_steps}")
    print(f"Decode tokens: {outputs.decode_tokens}")
    tokens_per_second = outputs.decode_tokens / generation_time if generation_time > 0 else 0.0
    avg_accept_length = (
        sum(outputs.accepet_length_per_step) / len(outputs.accepet_length_per_step)
        if outputs.accepet_length_per_step else 0.0
    )
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print(f"Average accept length per step: {avg_accept_length:.2f}")
    print(f"\nAccept lengths per step: {outputs.accepet_length_per_step}")
    
    print("\n" + "="*80)
    print("ACCEPTANCE STATISTICS")
    print("="*80)
    total_lookups = draft.static_sam_accepts + draft.dynamic_sam_accepts + draft.eagle_accepts
    print(f"Static SAM accepted: {draft.static_sam_accepts:3d} times ({draft.static_sam_accepts/total_lookups*100:5.1f}%)")
    print(f"Dynamic SAM accepted: {draft.dynamic_sam_accepts:3d} times ({draft.dynamic_sam_accepts/total_lookups*100:5.1f}%)")
    print(f"EAGLE accepted: {draft.eagle_accepts:3d} times ({draft.eagle_accepts/total_lookups*100:5.1f}%)")
    print(f"Total lookups: {total_lookups:3d}")
    print("="*80)
    
    print("\n" + "-"*80)
    print("GENERATED TEXT")
    print("-"*80)
    print(response)
    print("-"*80 + "\n")
    
    return {
        "outputs": outputs,
        "generation_time": generation_time,
        "tokens_per_sec": tokens_per_second,
        "avg_accept_length": avg_accept_length,
        "response": response,
        "draft": draft,
    }


@torch.inference_mode()
def baseline_generate(args, inputs, model, tokenizer):

    model.eval()
    
    from samd import SamdGenerationConfig
    gen_config = SamdGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        max_cache_len=args.max_cache_len,
        greedy=True,
        temperature=0.0
    )
    
    print("\n" + "="*80)
    print("Baseline generation (without SAM)...")
    print("="*80)
    
    st = time.perf_counter()
    tokens = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)[0]
    ed = time.perf_counter()
    
    response = tokenizer.decode(tokens, skip_special_tokens=True)
    
    print("\n" + "-"*80)
    print("BASELINE RESULTS")
    print("-"*80)
    baseline_time = ed - st
    tokens_generated = len(tokens) - inputs.input_ids.shape[-1]
    tokens_per_sec = tokens_generated / baseline_time if baseline_time > 0 else 0.0
    print(f"Generation time: {baseline_time:.2f} seconds")
    print(f"Tokens generated: {tokens_generated}")
    print(f"\nGenerated text:")
    print(response)
    print("-"*80 + "\n")
    
    return {
        "tokens": tokens,
        "tokens_generated": tokens_generated,
        "generation_time": baseline_time,
        "tokens_per_sec": tokens_per_sec,
        "response": response,
    }


def summarize_method_stats(method_stats: Dict[str, Dict[str, Any]]):
    summary: Dict[str, Dict[str, float]] = {}
    for name, stats in method_stats.items():
        invocations = stats.get("invocations", 0)
        accepted = stats.get("accepted_tokens", 0)
        generated = stats.get("generated_tokens", 0)
        accept_lengths = stats.get("accept_lengths", [])
        avg_token_len = accepted / invocations if invocations else 0.0
        avg_accept_len = (
            sum(accept_lengths) / len(accept_lengths) if accept_lengths else 0.0
        )
        summary[name] = {
            "invocations": invocations,
            "accepted_tokens": accepted,
            "generated_tokens": generated,
            # "avg_token_len": avg_token_len,
            "avg_accept_len": avg_accept_len,
            "accept_rate": accepted / generated if generated else 0.0,
        }
    return summary


RESULT_METHODS = ["dynamic", "static", "tree"]
RESULT_FIELDS = [
    # "timestamp",
    "prompt",
    "response",
    "len_threshold",
    "len_bias",
    "disable_dyn",
    "disable_eagle",
    "samd_n_predicts",
    "max_new_tokens",
    "samd_tokens_per_sec",
    "baseline_tokens_per_sec",
    "speedup",
    "decode_steps",
    "decode_tokens",
    "generation_time",
    "baseline_time",
    "overall_avg_accept_len",
]
for method in RESULT_METHODS:
    RESULT_FIELDS.extend([
        f"{method}_invocations",
        f"{method}_accepted_tokens",
        f"{method}_generated_tokens",
        # f"{method}_avg_token_len",
        f"{method}_avg_accept_len",
        f"{method}_accept_rate",
    ])


def build_results_row(args, prompt: str, samd_metrics: Dict[str, Any], baseline_metrics: Optional[Dict[str, Any]]):
    draft: WordGroupAwareDraftModel = samd_metrics["draft"]
    method_summary = summarize_method_stats(draft.method_stats)
    baseline_tokens_per_sec = baseline_metrics["tokens_per_sec"] if baseline_metrics else 0.0
    baseline_time = baseline_metrics["generation_time"] if baseline_metrics else 0.0
    speedup = (
        samd_metrics["tokens_per_sec"] / baseline_tokens_per_sec
        if baseline_tokens_per_sec else 0.0
    )
    outputs = samd_metrics["outputs"]
    row = {
        # "timestamp": datetime.utcnow().isoformat(),
        "prompt": prompt,
        "response": samd_metrics["response"],
        "len_threshold": args.len_threshold,
        "len_bias": args.len_bias,
        "disable_dyn": args.disable_dyn,
        "disable_eagle": args.disable_eagle,
        "samd_n_predicts": args.samd_n_predicts,
        "max_new_tokens": args.max_new_tokens,
        "samd_tokens_per_sec": samd_metrics["tokens_per_sec"],
        "baseline_tokens_per_sec": baseline_tokens_per_sec,
        "speedup": speedup,
        "decode_steps": getattr(outputs, "decode_steps", 0),
        "decode_tokens": getattr(outputs, "decode_tokens", 0),
        "generation_time": samd_metrics["generation_time"],
        "baseline_time": baseline_time,
        "overall_avg_accept_len": samd_metrics["avg_accept_length"],
    }
    for method in RESULT_METHODS:
        metrics = method_summary.get(method, {})
        row[f"{method}_invocations"] = metrics.get("invocations", 0)
        row[f"{method}_accepted_tokens"] = metrics.get("accepted_tokens", 0)
        row[f"{method}_generated_tokens"] = metrics.get("generated_tokens", 0)
        # row[f"{method}_avg_token_len"] = metrics.get("avg_token_len", 0.0)
        row[f"{method}_avg_accept_len"] = metrics.get("avg_accept_len", 0.0)
        row[f"{method}_accept_rate"] = metrics.get("accept_rate", 0.0)
    return row


def append_results_row(csv_path: str, row: Dict[str, Any]):
    directory = os.path.dirname(csv_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=RESULT_FIELDS, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def interactive_mode(args, model, tokenizer, sam):
    """Interactive mode for running multiple inferences without reloading SAM"""
    
    print("\n" + "="*80)
    print("INTERACTIVE MODE - SAM LOADED")
    print("="*80)
    print("You can now run multiple inferences with different parameters.")
    print("\nCommands:")
    print("  • :prompt <text>                - Set the prompt for generation")
    print("  • :set len_threshold <value>    - Change len_threshold")
    print("  • :set len_bias <value>         - Change len_bias")
    print("  • :set disable_dyn <true/false> - Toggle dynamic SAM")
    print("  • :set disable_eagle <true/false> - Toggle EAGLE/tree fallback")
    print("  • :set max_new_tokens <value>   - Change max_new_tokens")
    print("  • :set samd_n_predicts <value>  - Change n_predicts")
    print("  • :run or :generate             - Run inference with current prompt & settings")
    print("  • :show                         - Show current settings & prompt")
    print("  • :baseline                     - Run baseline generation with current prompt")
    print("  • :quit or :exit                - Exit interactive mode")
    print("="*80)
    print("\nWorkflow: Set parameters → Set prompt →  :run")
    
    # Show initial settings
    print(f"\nInitial settings:")
    print(f"  len_threshold: {args.len_threshold}")
    print(f"  len_bias: {args.len_bias}")
    print(f"  disable_dyn: {args.disable_dyn}")
    print(f"  disable_eagle: {args.disable_eagle}")
    print(f"  samd_n_predicts: {args.samd_n_predicts}")
    print(f"  max_new_tokens: {args.max_new_tokens}")
    
    # Current prompt
    current_prompt = None
    
    while True:
        try:
            print("\n" + "-"*80)
            user_input = input(">>> ").strip()
            
            if not user_input:
                continue
            
            # Exit commands
            if user_input.lower() in [':quit', ':exit', ':q']:
                print("Exiting interactive mode...")
                break
            
            # Set parameter commands
            if user_input.startswith(':set '):
                parts = user_input[5:].split(maxsplit=1)
                if len(parts) >= 2:
                    param = parts[0]
                    value = parts[1]
                    
                    try:
                        if param == 'len_threshold':
                            args.len_threshold = int(value)
                            print(f"✓ Set len_threshold to {args.len_threshold}")
                        elif param == 'len_bias':
                            args.len_bias = int(value)
                            print(f"✓ Set len_bias to {args.len_bias}")
                        elif param == 'disable_dyn':
                            args.disable_dyn = str2bool(value)
                            print(f"✓ Set disable_dyn to {args.disable_dyn}")
                        elif param == 'disable_eagle':
                            args.disable_eagle = str2bool(value)
                            print(f"✓ Set disable_eagle to {args.disable_eagle}")
                        elif param == 'max_new_tokens':
                            args.max_new_tokens = int(value)
                            print(f"✓ Set max_new_tokens to {args.max_new_tokens}")
                        elif param == 'samd_n_predicts':
                            args.samd_n_predicts = int(value)
                            print(f"✓ Set samd_n_predicts to {args.samd_n_predicts}")
                        else:
                            print(f"✗ Unknown parameter: {param}")
                            print("  Available: len_threshold, len_bias, disable_dyn, disable_eagle, max_new_tokens, samd_n_predicts")
                    except ValueError as e:
                        print(f"✗ Invalid value for {param}: {value} ({e})")
                else:
                    print("✗ Usage: :set <parameter> <value>")
                continue
            
            # Show settings command
            if user_input == ':show':
                print(f"\nCurrent settings:")
                print(f"  len_threshold: {args.len_threshold}")
                print(f"  len_bias: {args.len_bias}")
                print(f"  disable_dyn: {args.disable_dyn}")
                print(f"  disable_eagle: {args.disable_eagle}")
                print(f"  samd_n_predicts: {args.samd_n_predicts}")
                print(f"  max_new_tokens: {args.max_new_tokens}")
                print(f"  max_cache_len: {args.max_cache_len}")
                if current_prompt:
                    print(f"  prompt: '{current_prompt}'")
                else:
                    print(f"  prompt: <not set>")
                continue
            
            # Set prompt command
            if user_input.startswith(':prompt '):
                current_prompt = user_input[8:].strip()
                if current_prompt:
                    print(f"✓ Prompt set to: '{current_prompt}'")
                    print(f"  Use ':run' to generate")
                else:
                    print("✗ Empty prompt")
                continue
            
            # Run/Generate command
            if user_input.lower() in [':run', ':generate', ':gen']:
                if not current_prompt:
                    print("✗ No prompt set. Use ':prompt <text>' to set a prompt first")
                    continue
                
                print(f"\n{'='*80}")
                print(f"Running inference...")
                print(f"{'='*80}")
                print(f"Prompt: {current_prompt}")
                print(f"Settings: threshold={args.len_threshold}, bias={args.len_bias}, "
                      f"dyn={'off' if args.disable_dyn else 'on'}, eagle={'off' if args.disable_eagle else 'on'}")
                
                inputs = tokenizer([current_prompt], padding=True, return_tensors="pt").to(args.device)
                
                # Run generation
                samd_metrics = samd_generate_wordgroup(args, inputs, model, tokenizer, sam)
                
                # Optionally save results
                save_result = input("\nSave this result to CSV? (y/n): ").strip().lower()
                if save_result == 'y':
                    baseline_metrics = None
                    row = build_results_row(args, current_prompt, samd_metrics, baseline_metrics)
                    append_results_row(args.results_csv, row)
                    print(f"✓ Results saved to {args.results_csv}")
                continue
            
            # Baseline command
            if user_input.startswith(':baseline'):
                if not current_prompt:
                    print("✗ No prompt set. Use ':prompt <text>' to set a prompt first")
                    continue
                
                print(f"\nRunning baseline with prompt: {current_prompt}")
                inputs = tokenizer([current_prompt], padding=True, return_tensors="pt").to(args.device)
                baseline_generate(args, inputs, model, tokenizer)
                continue
            
            # If no command matched, show help
            print("✗ Unknown input. Available commands:")
            print("  :prompt <text>  - Set prompt")
            print("  :set <param> <value>  - Set parameter")
            print("  :run            - Run inference")
            print("  :show           - Show settings")
            print("  :baseline       - Run baseline")
            print("  :quit           - Exit")
            print("Type ':show' to see all commands")
        
        except KeyboardInterrupt:
            print("\n(Use ':quit' to exit)")
            continue
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            print("\nYou can continue with another prompt or ':quit' to exit")


def main():
    args = parse_args()
    
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=args.dtype,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"  EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"  Vocab size: {len(tokenizer)}")
    
    if args.sam_path and os.path.exists(args.sam_path):
        sam = load_wordgroup_sam(args.sam_path)
    else:
        print(f"Warning: SAM file not found at {args.sam_path}")
        print("Proceeding without static SAM (will use dynamic SAM only)")
        sam = None
    
    # Check if interactive mode is enabled
    if args.interactive:
        if sam is None:
            print("Error: Interactive mode requires a valid SAM file")
            return
        interactive_mode(args, model, tokenizer, sam)
        return
    
    # Single-shot mode (original behavior)
    if args.prompt:
        prompt = args.prompt
    else:
        prompt = "हिंदुस्तानी शास्त्रीय संगीत"
    
    print(f"\n{'='*80}")
    print(f"PROMPT")
    print(f"{'='*80}")
    print(prompt)
    print(f"{'='*80}\n")
    
    inputs = tokenizer(
        [prompt],
        padding=True,
        return_tensors="pt"
    ).to(args.device)
    
    print(f"Input tokens: {inputs.input_ids.shape[-1]}")
    
    baseline_metrics = None
    if not args.skip_baseline:
        baseline_metrics = baseline_generate(args, inputs, model, tokenizer)
    else:
        print("Skipping baseline generation (speedup will be 0)")
    
    if sam is not None:
        samd_metrics = samd_generate_wordgroup(args, inputs, model, tokenizer, sam)
    else:
        print("Skipping SAM generation (no SAM loaded)")


if __name__ == '__main__':
    main()