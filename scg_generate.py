#!/usr/bin/env python3
"""
scg_generate.py — Entry point for Samanvaya-Constrained Generation (SCG).

Three generation modes
----------------------
baseline    : Standard autoregressive decoding. No constraint.
constrained : SCG with opportunistic masking (DOMINO-style).
adaptive    : SCG + AdaSD entropy gate (T_G=0.5 initial, updated online).

Supports two models
-------------------
airavata : ai4bharat/Airavata  (LLaMA-based, Hindi instruction-tuned)
param    : bharatgenai/Param-1-2.9B-Instruct  (Param architecture)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from samanvaya_logits_processor import SamanvayaLogitsProcessor

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    "airavata": {
        "model_id": "ai4bharat/Airavata",
        "trust_remote_code": False,
        "vocab_scan": "vocab_scan_airavata.json",
        # Airavata is fine-tuned with the Tulu v2 instruction format.
        # Roles: <|system|>, <|user|>, <|assistant|> (plain text, not special tokens).
        "prompt_template": (
            "<|system|>\n"
            "आप एक सहायक हैं जो हिंदी में उत्तर देते हैं।\n"
            "<|user|>\n"
            "{instruction}\n"
            "<|assistant|>\n"
        ),
    },
    "param": {
        "model_id": "bharatgenai/Param-1-2.9B-Instruct",
        "trust_remote_code": True,
        "vocab_scan": "vocab_scan_param.json",
        # Param uses ChatML-style prompt
        "prompt_template": (
            "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        ),
    },
}

# ---------------------------------------------------------------------------
# Model loading (cached across calls in the same process)
# ---------------------------------------------------------------------------

_model_cache: dict[str, tuple] = {}


def load_model(model_key: str, device: str = "cuda", dtype=torch.bfloat16):
    """Load and cache model + tokenizer."""
    if model_key in _model_cache:
        return _model_cache[model_key]

    cfg = MODELS[model_key]
    print(f"Loading {cfg['model_id']} …", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_id"],
        trust_remote_code=cfg["trust_remote_code"],
    )

    # Param-1's custom _init_rope() only accepts rope_scaling=None | {"type":
    # "linear"|"dynamic", "factor": float}.  The model's config has
    # {"rope_theta": 10000.0, "rope_type": "default"} — "default" means no
    # special scaling, equivalent to rope_scaling=None.  Set it to None so the
    # model falls through to standard RoPE and avoids KeyError.
    from transformers import AutoConfig
    model_cfg = AutoConfig.from_pretrained(
        cfg["model_id"], trust_remote_code=cfg["trust_remote_code"]
    )
    if (
        hasattr(model_cfg, "rope_scaling")
        and isinstance(model_cfg.rope_scaling, dict)
        and model_cfg.rope_scaling.get("rope_type", "") == "default"
    ):
        model_cfg.rope_scaling = None

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        trust_remote_code=cfg["trust_remote_code"],
        torch_dtype=dtype,
        device_map=device,
        config=model_cfg,
    )
    model.eval()
    _model_cache[model_key] = (model, tokenizer)
    print(f"Loaded {cfg['model_id']}", flush=True)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------

def generate(
    prompt: str,
    model_key: str = "airavata",
    mode: str = "baseline",
    max_new_tokens: int = 256,
    device: str = "cuda",
    T_G_init: float = 0.5,
    workspace: Path = Path(__file__).parent,
) -> dict:
    """
    Generate a Hindi response and return results + stats.

    Parameters
    ----------
    prompt       : Raw instruction text (Hindi or English).
    model_key    : "airavata" or "param".
    mode         : "baseline", "constrained", or "adaptive".
    max_new_tokens: Maximum tokens to generate.
    device       : PyTorch device string.
    T_G_init     : Initial entropy gate threshold (adaptive mode only).
    workspace    : Directory containing vocab_scan_*.json files.

    Returns
    -------
    dict with keys: response, tokens_per_sec, gen_stats, parser_compliance
    """
    assert mode in ("baseline", "constrained", "adaptive"), f"Unknown mode: {mode}"
    assert model_key in MODELS, f"Unknown model: {model_key}"

    cfg = MODELS[model_key]
    model, tokenizer = load_model(model_key, device)

    # Format prompt
    full_prompt = cfg["prompt_template"].format(instruction=prompt)
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(device)

    # Build logits processor
    logits_processors = []
    scg_processor: Optional[SamanvayaLogitsProcessor] = None

    if mode in ("constrained", "adaptive"):
        vocab_scan_path = workspace / cfg["vocab_scan"]
        scg_processor = SamanvayaLogitsProcessor(
            tokenizer=tokenizer,
            vocab_scan=vocab_scan_path,
            use_entropy_gate=(mode == "adaptive"),
            T_G_init=T_G_init,
            eos_token_id=tokenizer.eos_token_id,
        )
        scg_processor.reset_stats()
        logits_processors.append(scg_processor)

    # Generate
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,         # greedy decoding for reproducibility
            temperature=1.0,
            logits_processor=logits_processors if logits_processors else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0

    # Decode only the new tokens
    new_ids = output_ids[0, input_ids.shape[1]:]
    response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    n_new = len(new_ids)
    tokens_per_sec = n_new / elapsed if elapsed > 0 else 0.0

    # Flush any remaining buffer so the parser sees the final word
    if scg_processor is not None:
        scg_processor._flush_buffer()

    result = {
        "response": response,
        "prompt": prompt,
        "formatted_prompt": full_prompt,   # needed for cross-PPL computation
        "mode": mode,
        "model": model_key,
        "new_tokens": n_new,
        "elapsed_sec": round(elapsed, 3),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "gen_stats": scg_processor.generation_stats() if scg_processor else {},
        "parser_compliance": (
            scg_processor.parser.compliance_rate() if scg_processor else None
        ),
    }

    # If running baseline but want FSM compliance measurement, run parser
    # over the generated response post-hoc.
    if mode == "baseline":
        from samanvaya_parser import SamanvayaParser
        post_parser = SamanvayaParser()
        for word in response.split():
            post_parser.update(word)
        result["parser_compliance"] = post_parser.compliance_rate()
        result["parser_stats"] = post_parser.stats()

    return result


# ---------------------------------------------------------------------------
# Cross-perplexity (fluency cost measurement)
# ---------------------------------------------------------------------------

def compute_cross_ppl(
    response_text: str,
    formatted_prompt: str,
    model_key: str,
    device: str = "cuda",
) -> float:
    """
    Compute perplexity of response_text under the unconstrained model.

    Method: teacher-forced forward pass of (prompt + response).  Loss is
    computed only on response tokens (prompt positions masked with -100).

    Interpretation: PPL_SCG / PPL_baseline ≈ 1 means the constraint imposed
    minimal fluency cost.  PPL_SCG >> PPL_baseline means SCG forced the model
    into low-probability token sequences.
    """
    model, tokenizer = load_model(model_key, device)

    full_text = formatted_prompt + response_text
    full_ids   = tokenizer(full_text,        return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    prompt_ids = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=True).input_ids
    prompt_len = prompt_ids.shape[1]

    if full_ids.shape[1] <= prompt_len:
        return float("nan")

    labels = full_ids.clone()
    labels[0, :prompt_len] = -100

    with torch.no_grad():
        out = model(full_ids, labels=labels)

    return round(torch.exp(out.loss).item(), 3)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SCG generation CLI")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Hindi instruction prompt")
    parser.add_argument("--model", choices=list(MODELS), default="airavata")
    parser.add_argument("--mode", choices=["baseline", "constrained", "adaptive"],
                        default="constrained")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu", type=int, default=None,
                        help="CUDA device index (e.g. 1 for GPU 1)")
    parser.add_argument("--T-G", type=float, default=0.5,
                        help="Initial entropy gate threshold (adaptive mode)")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if args.gpu is not None else args.device
    workspace = Path(__file__).parent

    result = generate(
        prompt=args.prompt,
        model_key=args.model,
        mode=args.mode,
        max_new_tokens=args.max_new_tokens,
        device=device,
        T_G_init=args.T_G,
        workspace=workspace,
    )

    print("\n" + "=" * 60)
    print(f"Model   : {result['model']}")
    print(f"Mode    : {result['mode']}")
    print(f"Prompt  : {result['prompt'][:80]}")
    print(f"Response: {result['response']}")
    print(f"Tokens  : {result['new_tokens']} @ {result['tokens_per_sec']:.1f} tok/s")
    print(f"Compliance: {result['parser_compliance']:.2%}")
    if result["gen_stats"]:
        gs = result["gen_stats"]
        print(f"Stats   : constrained_steps={gs['constrained_steps']}, "
              f"opp_saves={gs['opportunistic_saves']}, "
              f"mask_applied={gs['mask_applied']}, "
              f"intervention_rate={gs['intervention_rate']:.2%}")


if __name__ == "__main__":
    main()
