#!/usr/bin/env python3
"""
samanvaya_logits_processor.py — HuggingFace LogitsProcessor for SCG.

Design
------
Word-boundary buffer:
  Both models use SentencePiece BPE with ▁ prefix for word-initial tokens.
  We accumulate subword token IDs; when a ▁-prefixed token arrives it signals
  a word boundary — we flush the buffer (decode → word → parser.update) and
  start a fresh buffer with the new token.

Key timing insight (one-ahead masking):
  The LogitsProcessor sees input_ids[-1] = the token just committed, and
  scores = logits for the NEXT token to generate.  When a ▁-token arrives,
  we flush the old word AND immediately decode the new (single-token) word
  in the buffer to check whether it would trigger a constraint.  If so, we
  apply masking for the NEXT token (continuation or following word-start).
  This prevents the "masks arrive one step late" problem.

Opportunistic masking (DOMINO):
  Check greedy token first. Only mask if it violates the constraint.

Entropy gate (AdaSD, optional):
  Skip masking when model entropy H ≤ T_G (model is confident).
  T_G updated online as rolling mean of entropies at constrained steps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from transformers import LogitsProcessor, PreTrainedTokenizerBase

from samanvaya_parser import (
    A_ENDING,
    ALL_AUX,
    EE_ENDING,
    MUST_CASE_TRIGGERS,
    MUST_CASE_VALID,
    NON_VERB_AUX_TRIGGERS,
    PHRASE_VALID_NEXT,
    SamanvayaParser,
    _COMMON_NOUN_EXCLUSIONS,
    _NON_VERB_SUFFIXES,
)

NEG_INF = float("-inf")

# Import state constants
FREE = "FREE"
MUST_AUX = "MUST_AUX"
MUST_CASE = "MUST_CASE"
MUST_CONT = "MUST_CONT"


def _word_would_trigger(word: str) -> str:
    """
    Return the FSM state that parser would enter AFTER processing `word` from
    FREE state, without actually changing any parser state.
    """
    word = word.rstrip("।!?.,;:\"')")
    if not word:
        return FREE
    if word in MUST_CASE_TRIGGERS:
        return MUST_CASE
    if word in PHRASE_VALID_NEXT:
        return MUST_CONT
    # Require minimum 3 chars: 2-char tokens like 'गा', 'जा', 'ला' are typically
    # subword prefixes (first sub-token of गांधी, जाना, लाखों) not past participles.
    if (
        len(word) >= 3
        and (word.endswith(A_ENDING) or word.endswith(EE_ENDING))
        and word not in ALL_AUX
        and word not in NON_VERB_AUX_TRIGGERS
        and word not in _COMMON_NOUN_EXCLUSIONS
        and not any(word.endswith(sfx) for sfx in _NON_VERB_SUFFIXES)
    ):
        return MUST_AUX
    return FREE


class SamanvayaLogitsProcessor(LogitsProcessor):
    """
    Injects Samanvaya LWG rules into the LLM logit distribution.

    Parameters
    ----------
    tokenizer        : The model's tokenizer.
    vocab_scan       : Parsed vocab_scan_{model}.json dict (or path to it).
    use_entropy_gate : Enable AdaSD entropy gate.
    T_G_init         : Initial entropy threshold (default 0.5 nats).
    eos_token_id     : EOS token; always allowed even in constrained state.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        vocab_scan: dict | str | Path,
        use_entropy_gate: bool = False,
        T_G_init: float = 0.5,
        eos_token_id: Optional[int] = None,
        trace: bool = False,
    ) -> None:
        self.tokenizer = tokenizer

        if isinstance(vocab_scan, (str, Path)):
            with open(vocab_scan, encoding="utf-8") as f:
                vocab_scan = json.load(f)

        svt = vocab_scan["state_valid_tokens"]
        self._must_aux_ids: set[int] = set(svt["MUST_AUX"])
        self._must_case_ke_ids: set[int] = set(svt["MUST_CASE_KE"])
        self._must_case_ki_ids: set[int] = set(svt["MUST_CASE_KI"])

        self._spiece_prefix = "▁"
        tok_vocab = tokenizer.get_vocab()
        self._word_to_ids: dict[str, set[int]] = {}
        for tok_str, tok_id in tok_vocab.items():
            word = tok_str.lstrip(self._spiece_prefix)
            self._word_to_ids.setdefault(word, set()).add(tok_id)

        self._word_start_ids: set[int] = {
            tok_id
            for tok_str, tok_id in tok_vocab.items()
            if tok_str.startswith(self._spiece_prefix)
        }

        self.eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
        self.parser = SamanvayaParser()
        self._token_buffer: list[int] = []
        self._prompt_length: int = -1

        # "Pending" constraint: set when the current buffer's word would trigger
        # a new obligation if flushed. Used for one-ahead masking.
        self._pending_state: str = FREE
        self._pending_trigger: Optional[str] = None  # for MUST_CASE

        self.use_entropy_gate = use_entropy_gate
        self.T_G = T_G_init
        self._entropy_window: list[float] = []

        self.total_steps: int = 0
        self.constrained_steps: int = 0
        self.opportunistic_saves: int = 0
        self.mask_applied: int = 0
        self.entropy_gate_skips: int = 0

        self.trace = trace
        self._trace_log: list[dict] = []

    # ------------------------------------------------------------------
    # LogitsProcessor protocol
    # ------------------------------------------------------------------

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        self.total_steps += 1

        if self._prompt_length < 0:
            self._prompt_length = input_ids.shape[1]
            if self.trace:
                self._trace_log.append({"step": self.total_steps, "action": "INIT"})
            return scores

        last_token_id = int(input_ids[0, -1].item())
        last_token_str = self.tokenizer.decode([last_token_id])

        parser_state_before = self.parser.state
        pending_before = self._pending_state

        self._on_new_token(last_token_id)

        eff_state = self.parser.state if self.parser.is_constrained() else self._pending_state

        if eff_state == FREE:
            if self.trace:
                self._trace_log.append({
                    "step": self.total_steps,
                    "committed_token": last_token_str,
                    "committed_id": last_token_id,
                    "fsm_before": parser_state_before,
                    "fsm_after": self.parser.state,
                    "pending": self._pending_state,
                    "eff_state": FREE,
                    "buffer": self._decode_buffer(),
                    "action": "PASS",
                })
            return scores

        self.constrained_steps += 1
        valid_ids = self._resolve_valid_token_ids(eff_state)

        if not valid_ids:
            if self.trace:
                self._trace_log.append({
                    "step": self.total_steps,
                    "committed_token": last_token_str,
                    "committed_id": last_token_id,
                    "fsm_before": parser_state_before,
                    "fsm_after": self.parser.state,
                    "pending": self._pending_state,
                    "eff_state": eff_state,
                    "buffer": self._decode_buffer(),
                    "action": "PASS_NO_VALID_IDS",
                })
            return scores

        probs = torch.softmax(scores[0], dim=-1)
        greedy_id = int(scores[0].argmax().item())
        greedy_str = self.tokenizer.decode([greedy_id])
        greedy_prob = float(probs[greedy_id].item())

        if greedy_id == self.eos_token_id:
            self._flush_buffer()
            if self.trace:
                self._trace_log.append({
                    "step": self.total_steps,
                    "committed_token": last_token_str,
                    "committed_id": last_token_id,
                    "fsm_before": parser_state_before,
                    "fsm_after": self.parser.state,
                    "pending": self._pending_state,
                    "eff_state": eff_state,
                    "buffer": self._decode_buffer(),
                    "action": "PASS_EOS",
                    "greedy_str": greedy_str,
                    "greedy_prob": greedy_prob,
                })
            return scores

        if greedy_id in valid_ids:
            self.opportunistic_saves += 1
            if self.trace:
                self._trace_log.append({
                    "step": self.total_steps,
                    "committed_token": last_token_str,
                    "committed_id": last_token_id,
                    "fsm_before": parser_state_before,
                    "fsm_after": self.parser.state,
                    "pending": self._pending_state,
                    "eff_state": eff_state,
                    "buffer": self._decode_buffer(),
                    "action": "OPP_SAVE",
                    "greedy_str": greedy_str,
                    "greedy_prob": greedy_prob,
                })
            return scores

        # Entropy gate (AdaSD)
        entropy = None
        if self.use_entropy_gate:
            entropy = self._compute_entropy(scores[0])
            if entropy <= self.T_G:
                self.entropy_gate_skips += 1
                if self.trace:
                    self._trace_log.append({
                        "step": self.total_steps,
                        "committed_token": last_token_str,
                        "committed_id": last_token_id,
                        "fsm_before": parser_state_before,
                        "fsm_after": self.parser.state,
                        "pending": self._pending_state,
                        "eff_state": eff_state,
                        "buffer": self._decode_buffer(),
                        "action": "ENTROPY_SKIP",
                        "greedy_str": greedy_str,
                        "greedy_prob": greedy_prob,
                        "entropy": round(entropy, 4),
                        "T_G": round(self.T_G, 4),
                    })
                return scores
            self._entropy_window.append(entropy)
            if len(self._entropy_window) > 50:
                self._entropy_window.pop(0)
            self.T_G = sum(self._entropy_window) / len(self._entropy_window)

        # Apply vocabulary mask
        self.mask_applied += 1
        masked_scores = scores.clone()
        mask = torch.full((scores.shape[1],), NEG_INF, device=scores.device)
        for tid in valid_ids:
            if tid < mask.shape[0]:
                mask[tid] = 0.0
        masked_scores[0] = scores[0] + mask
        forced_id = int(masked_scores[0].argmax().item())
        forced_str = self.tokenizer.decode([forced_id])
        forced_prob = float(probs[forced_id].item())

        if self.trace:
            self._trace_log.append({
                "step": self.total_steps,
                "committed_token": last_token_str,
                "committed_id": last_token_id,
                "fsm_before": parser_state_before,
                "fsm_after": self.parser.state,
                "pending": self._pending_state,
                "eff_state": eff_state,
                "buffer": self._decode_buffer(),
                "action": "MASK_APPLIED",
                "greedy_str": greedy_str,
                "greedy_id": greedy_id,
                "greedy_prob": round(greedy_prob, 4),
                "forced_str": forced_str,
                "forced_id": forced_id,
                "forced_prob": round(forced_prob, 4),
                "entropy": round(entropy, 4) if entropy is not None else None,
            })

        return masked_scores

    # ------------------------------------------------------------------
    # Word-boundary buffer management
    # ------------------------------------------------------------------

    def _on_new_token(self, token_id: int) -> None:
        """
        Process the most recently committed token.

        When a ▁-prefixed token arrives:
          1. Flush old buffer → call parser.update(old_word)
          2. Start new buffer with the ▁-token
          3. Decode the new (possibly single-token) buffer word and check
             whether it would trigger a new constraint (pending state).
        """
        self._pending_state = FREE
        self._pending_trigger = None

        is_word_start = token_id in self._word_start_ids or len(self._token_buffer) == 0

        if is_word_start and self._token_buffer:
            # Flush old word → update parser
            self._flush_buffer()
            self._token_buffer = [token_id]

            # Handle: if parser is now constrained from the just-flushed word,
            # check if this new ▁-token's word already satisfies the constraint
            # (single-token word case). If yes, immediately flush and go FREE.
            if self.parser.is_constrained():
                new_word = self._decode_buffer().rstrip("।!?.,;:\"')")
                if new_word and new_word in self.parser.valid_next_words():
                    self._flush_buffer()  # → satisfies constraint → parser advances
                    # No pending state needed; parser just advanced.
                    return
                # else: constraint still active; masking handled via parser.state
                return

            # Parser is FREE (old word didn't trigger). Now check if the NEW
            # buffer word itself would trigger a new constraint.
            new_word = self._decode_buffer().rstrip("।!?.,;:\"')")
            if new_word:
                candidate_state = _word_would_trigger(new_word)
                if candidate_state != FREE:
                    self._pending_state = candidate_state
                    if candidate_state == MUST_CASE:
                        self._pending_trigger = new_word  # "के" or "की"

        elif is_word_start:
            # Buffer was empty (start of generation or after early flush).
            # Don't compute pending here: the first token of a word may be a
            # subword prefix (e.g. ▁महा as first sub-token of महात्मा) and
            # would fire bogus MUST_AUX for the continuation token.
            # Constraints are still enforced via parser.state once the full
            # word is flushed on the *next* word-start token arrival.
            self._token_buffer = [token_id]
        else:
            # Continuation token: append to current buffer
            self._token_buffer.append(token_id)
            # Update pending state based on current (partial) word
            # A partial word can't yet be confirmed as a trigger, so clear pending.
            self._pending_state = FREE
            self._pending_trigger = None

    def _flush_buffer(self) -> None:
        """Decode current buffer → one Hindi word → call parser.update(word)."""
        if not self._token_buffer:
            return
        word_text = self.tokenizer.decode(self._token_buffer, skip_special_tokens=True).strip()
        if word_text:
            self.parser.update(word_text)
        self._token_buffer = []

    def _decode_buffer(self) -> str:
        """Decode current buffer without flushing."""
        if not self._token_buffer:
            return ""
        return self.tokenizer.decode(self._token_buffer, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # Valid token ID resolution
    # ------------------------------------------------------------------

    def _resolve_valid_token_ids(self, eff_state: str) -> set[int]:
        """Return set of valid token IDs for the given effective FSM state."""

        if eff_state == MUST_AUX:
            return self._must_aux_ids

        if eff_state == MUST_CASE:
            # Trigger may come from parser (already flushed) or pending
            trigger = (
                self.parser._case_trigger
                if self.parser.state == MUST_CASE
                else self._pending_trigger
            )
            if trigger == "के":
                return self._must_case_ke_ids
            if trigger == "की":
                return self._must_case_ki_ids
            return set()

        if eff_state == MUST_CONT:
            # For parser-driven MUST_CONT, use parser's valid_next_words
            # For pending MUST_CONT, use _word_to_ids for the pending word's continuations
            if self.parser.state == MUST_CONT:
                valid_words = self.parser.valid_next_words()
            else:
                # Pending MUST_CONT: the pending word is in _pending_trigger context
                pending_word = self._decode_buffer().rstrip("।!?.,;:\"')")
                valid_words = PHRASE_VALID_NEXT.get(pending_word, frozenset())

            ids: set[int] = set()
            for word in valid_words:
                ids.update(self._word_to_ids.get(word, set()))
                ids.update(self._word_to_ids.get(self._spiece_prefix + word, set()))
            return ids & self._word_start_ids

        return set()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_entropy(logits: torch.Tensor) -> float:
        probs = torch.softmax(logits, dim=-1)
        return -(probs * (probs + 1e-10).log()).sum().item()

    def get_trace_log(self) -> list[dict]:
        return list(self._trace_log)

    def generation_stats(self) -> dict:
        return {
            "total_steps": self.total_steps,
            "constrained_steps": self.constrained_steps,
            "opportunistic_saves": self.opportunistic_saves,
            "mask_applied": self.mask_applied,
            "entropy_gate_skips": self.entropy_gate_skips,
            "intervention_rate": (
                self.mask_applied / self.total_steps if self.total_steps > 0 else 0.0
            ),
            "opp_save_rate": (
                self.opportunistic_saves / self.constrained_steps
                if self.constrained_steps > 0 else 0.0
            ),
            "parser_stats": self.parser.stats(),
        }

    def reset_stats(self) -> None:
        self.total_steps = 0
        self.constrained_steps = 0
        self.opportunistic_saves = 0
        self.mask_applied = 0
        self.entropy_gate_skips = 0
        self._token_buffer = []
        self._entropy_window = []
        self._prompt_length = -1
        self._pending_state = FREE
        self._pending_trigger = None
        self._trace_log = []
        self.parser.reset()
