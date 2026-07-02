# Project: Samanvaya-Constrained Generation (SCG) for Hindi LLMs

## Who You Are Talking To
Pranav Shinde, MTech student at IIT Bombay. MTP Stage II project under Prof. Ganesh Ramakrishnan. Target conference: ACL ARR October 2026 → EMNLP. Guide has reviewed progress and given a specific direction (see below).

---

## One-Line Summary
Inject Samanvaya Local Word Group (LWG) rules into Hindi LLM generation at **inference time only** — no training changes, no tokenizer retraining — and measure whether this improves linguistic coherence of generated Hindi text.

---

## What Has Been Done (Thesis Stage II)

The thesis (mtp2_23m033.pdf) proposed **WGA-SAM**: Word-Group-Aware Suffix Automaton decoding. It extended SAM-Decoding (Hu et al., 2024) by adding `word_boundaries: List[bool]` to the suffix automaton, stopping drafts at LWG boundaries during speculative decoding.

**Result: It failed.**
- WGA-SAM speedup: 2.76× vs unconstrained SAM: 3.14×
- MAT (Mean Accepted Tokens): 4.925 vs 5.117
- Root cause: hard LWG boundary stopping shortens drafts. A 7-token draft with 4 accepted beats a 3-token fully-correct draft. The verifier (Airavata 7B) was never trained on LWG structure, so LWG-aligned drafts have no acceptance advantage.

**This approach is abandoned.** Do not revisit SAM-based approaches.

---

## Guide's Direction (Current Constraint)

The guide reviewed 5 papers and explicitly said:
> "Look on the **inference side** only. Do not make changes to tokenization or pretraining from scratch."

The 5 papers the guide shared (all in this directory):
1. `adaSd.pdf` — Adaptive Speculative Decoding (entropy-based adaptive stopping, no training)
2. `adaspec.pdf` — Multilingual speculative decoding with language-specific drafter
3. `group tree optimization.pdf` — GTO: draft tree RL training (training-side, reference only)
4. `guiding llms right way.pdf` — DOMINO: grammar-constrained generation (inference-side, **main reference**)
5. `learning to draft.pdf` — LTD: RL-based adaptive decoding (training-side, reference only)

**Papers with no GitHub repo (must implement from scratch):** DOMINO, AdaSD, Future Validity (not yet in directory).

---

## The New Method: Samanvaya-Constrained Generation (SCG)

**Core idea (from DOMINO + AdaSD):**
- Samanvaya LWG rules define a finite state machine (FSM) over Hindi word sequences
- At each decode step, check if the model is in a "constrained" FSM state
- If constrained, use **opportunistic masking** (DOMINO): check greedy token first; only apply vocabulary mask if model's choice violates the rule
- Optionally gate constraint with **entropy threshold** (AdaSD): if model is confident (low entropy), trust it; if uncertain (high entropy), enforce the constraint

**Why this works:** In FREE FSM states (~75-85% of decode steps), zero overhead. Mask only applies in MUST_AUX / MUST_CASE / MUST_CONT states when the model would otherwise violate a rule.

---

## Samanvaya LWG Rules (from wordgrouping_rules.py)

The rules operate at whitespace-delimited Hindi word level:

| Rule | Trigger | Obligation |
|---|---|---|
| Verb-Auxiliary chain | Word ends in ा (A_ENDING) or ी (EE_ENDING) | Next word must be in AUX_AFTER_EE or AUX_AFTER_A |
| Compound postposition | Word is "के" or "की" | Next word must be a RULE3_MULTIWORDS continuation |
| Fixed phrase | Word is first token of RULE1_PHRASES or RULE3_MULTIWORDS | All subsequent words must complete the phrase |
| Case marker left-attach | Word is in ATTACH_TO_LEFT | Should attach to previous word (enforced via masking previous token's follow-up) |

Key constants from `wordgrouping_rules.py`:
- `ATTACH_TO_LEFT`: से, में, का, के, की, को, पर, ने, भी, ही, द्वारा, वाला, वाली, वाले...
- `AUX_AFTER_EE`: जाती, गई, जाएगी, है, हैं, थी, थे, था
- `AUX_AFTER_A`: जाता, गया, जाएगा, है, हैं, थी, थे, था
- `RULE3_MULTIWORDS`: रहे_हैं, रहा_है, के_लिए, के_बाद, के_साथ, बारे_में, ...
- `RULE1_PHRASES`: fixed expressions like हाल_ही_में, दे_दिया, ...

---

## Files To Be Implemented (4-Day Plan)

```
wordgrouping_rules.py       ← EXISTING, do not modify
build_vocab_scan.py         ← Day 1 AM: scan tokenizer, output vocab_scan.json
vocab_scan.json             ← Day 1 PM: auto-generated
samanvaya_parser.py         ← Day 1 PM / Day 2 AM: FSM state machine
samanvaya_logits_processor.py ← Day 2: HuggingFace LogitsProcessor
scg_generate.py             ← Day 3 AM: entry point for all 3 methods
eval_scg.py                 ← Day 3 PM: evaluation harness
hindi_eval_50.jsonl         ← Day 3: evaluation data (source from MT-Bench-Hindi)
results/eval.json           ← Day 4: output
results/summary_table.md    ← Day 4: final comparison
```

---

## Implementation: build_vocab_scan.py

Scans Airavata tokenizer (32k vocab), classifies each token:
- `CASE_MARKER` — token is exactly a Samanvaya case marker (से, में, etc.)
- `AUX_VERB` — token is an auxiliary verb (है, थी, गया, etc.)
- `PHRASE_START` — token is the first word of a RULE1/RULE3 phrase
- `PHRASE_CONT` — token is a continuation word of a phrase
- `CASE_MARKER_MERGED` — token has a case marker merged as suffix (e.g. "रामसे")
- `FREE` — no Samanvaya role

Also pre-computes `state_valid_tokens`:
```json
{
  "MUST_AUX": [list of token IDs for all auxiliary verbs],
  "MUST_CASE": [list of token IDs for phrase continuations and case markers]
}
```

Also computes **WGTF** (Word Group Tokenization Fidelity) = fraction of Samanvaya vocabulary words that are single tokens in Airavata. Hypothesis: below 50%, which justifies the runtime constraint approach.

Model to use: `ai4bharat/airavata` (LLaMA-based, Hindi instruction-tuned, 7B parameters).

---

## Implementation: samanvaya_parser.py

Finite state machine. States:
- `FREE` — no active obligation
- `MUST_AUX` — last word ended in ा/ी, auxiliary must follow
- `MUST_CASE` — generated के/की standalone, compound postposition must follow
- `MUST_CONT` — mid fixed-phrase, next word predetermined
- `PARTIAL` — collecting subword tokens, word not complete yet

Key methods:
- `update(word: str) -> str` — process one complete Hindi word, return new state
- `is_constrained() -> bool` — True if in MUST_AUX/MUST_CASE/MUST_CONT
- `valid_next_words() -> set` — set of words valid in current state
- `compliance_rate() -> float` — completions / (completions + violations)

State transitions:
```
FREE + word_ends_ा_or_ी → MUST_AUX
FREE + word_in_PHRASE_MAP → MUST_CONT
FREE + word_is_"के"_or_"की" → MUST_CASE

MUST_AUX + word_in_AUX_SET → FREE (completion)
MUST_AUX + other_word → FREE (violation, re-evaluate word as FREE)

MUST_CASE + valid_continuation → FREE (completion)
MUST_CASE + other → FREE (violation)

MUST_CONT + expected_next_word → MUST_CONT (if more words needed) or FREE (if phrase complete)
MUST_CONT + wrong_word → FREE (violation)
```

---

## Implementation: samanvaya_logits_processor.py

HuggingFace `LogitsProcessor` subclass. Critical engineering detail: **word-boundary buffer**.

Airavata uses SentencePiece BPE. Word-start tokens have `▁` prefix. Algorithm:
1. When a `▁`-prefixed token arrives, flush token buffer → decode complete previous word → call `parser.update(word)`
2. Append new token to fresh buffer
3. Check parser state; if constrained, apply opportunistic masking

Opportunistic masking (DOMINO):
```python
if greedy_token in set(valid_token_ids):
    return scores  # model already chose correctly, no intervention
```

Entropy gate (AdaSD, optional):
```python
H = -(probs * (probs + 1e-10).log()).sum().item()
if H <= self.T_G:
    return scores  # model confident, trust it
# else: apply mask
self.T_G = mean(recent_rejected_entropies)  # online update
```

Stats tracked per generation:
- `total_steps` — total decode steps
- `constrained_steps` — steps where parser was in constrained state
- `opportunistic_saves` — times model was already valid (no mask needed)
- `mask_applied` — times mask was actually applied
- `entropy_gate_skips` — times entropy gate suppressed constraint

---

## Implementation: scg_generate.py

Three generation modes:
1. `baseline` — no LogitsProcessor, standard autoregressive
2. `constrained` — SamanvayaLogitsProcessor with opportunistic masking
3. `adaptive` — SamanvayaLogitsProcessor + AdaSD entropy gate (T_G=0.5 initial)

Returns: response text, tokens/s, LWG compliance stats.

---

## Evaluation Metrics

| Metric | What it shows | Source |
|---|---|---|
| LWG Compliance Rate | % verb-aux chains, case markers, phrases correctly formed | `parser.compliance_rate()` |
| Intervention Rate | % decode steps where mask was applied | `mask_applied / total_steps` |
| Opportunistic Save Rate | % constrained steps where model was already valid | tracks DOMINO effectiveness |
| WGTF | Fraction of Samanvaya vocab as single tokens in tokenizer | `build_vocab_scan.py` one-time |
| Tokens/second | Throughput — overhead measure | wall-clock timing |
| ChrF | Translation quality vs reference | sacrebleu |

**Baseline comparison table (target):**
| Method | LWG Compliance | Tokens/s | ChrF |
|---|---|---|---|
| Baseline (AR) | ~45-60% | 100% | baseline |
| SCG (constrained) | ~82-90% | ~93-97% | +1-3 |
| AdaSCG (entropy gate) | ~75-85% | ~97-99% | +0.5-2 |

---

## Key Design Decisions Made

1. **No training changes** — guide's explicit instruction. Everything is inference-side.
2. **Use opportunistic masking** — from DOMINO: check greedy first, mask only if invalid. This keeps overhead near zero.
3. **FSM not CFG** — Samanvaya rules are finite-state, not context-free. No need for DOMINO's full CFG parser.
4. **vocab_scan.json offline** — pre-compute token categorization once. O(1) lookup at inference time.
5. **Word buffer for subword alignment** — accumulate tokens until `▁`-prefix token signals word boundary. Only then evaluate parser.
6. **Three evaluation modes** — baseline / constrained / adaptive, run on same 50 prompts for direct comparison.

---

## Proposed Paper Narrative

**Problem:** Hindi LLMs violate Samanvaya LWG rules because BPE tokenization fragments linguistic units (WGTF diagnostic shows X% tokenizer coverage). Training objective (next-token prediction) has no LWG-level signal.

**Method:** SCG injects Samanvaya rules at decode time using a lightweight FSM + opportunistic masking. Inspired by DOMINO's constrained generation architecture, adapted for the finite-state grammar of Hindi morphosyntax. AdaSD's entropy gate makes constraint application adaptive.

**Results:** LWG compliance increases from X% to Y% with Z% mean intervention rate (i.e., model naturally complies most of the time; constraint is a lightweight corrective). Throughput overhead: W%. ChrF improvement: +N.

**Contribution framing:** First inference-time enforcement of Samanvaya LWG rules in Hindi LLM generation. WGTF metric as diagnostic for tokenizer-LWG misalignment (analogous to GTO's training-inference mismatch insight, applied at tokenization level).

---

## Papers Summary (All in This Directory)

**DOMINO** (`guiding llms right way.pdf`): Grammar-constrained generation via opportunistic masking + pre-computed token trees. No GitHub. We borrow: opportunistic masking concept (~10 lines). We skip: full CFG parser, subterminal trees, count-based speculative model.

**AdaSD** (`adaSd.pdf`): Entropy-based adaptive speculative stopping. T_G = mean entropy of rejected tokens, updated online. No GitHub. We borrow: entropy gate formula (~15 lines). We skip: KDE-based Bayesian JS-distance acceptance.

**AdaSpec** (`adaspec.pdf`): Multilingual speculative decoding with language-specific drafter + vocabulary simplification. GitHub available. Reference for future: Hindi-specific EAGLE drafter training.

**GTO** (`group tree optimization.pdf`): Draft tree RL training via Group-based PPO. GitHub: hsj576/GTO. Training-side. We borrow: "training-inference mismatch" framing for paper motivation. WGTF metric is our tokenization-level analog of GTO's insight.

**LTD** (`learning to draft.pdf`): RL co-adaptive depth+size policies, reward = throughput. GitHub: zhihao/Learning-to-Draft. Training-side. Reference for throughput reward formula: λ_c = L_A / T_total.

---

## Pending

- [ ] Get "Future Validity is the Missing Statistic" paper from guide (Φ-estimation for grammar-faithful speculative decoding — relevant to Component 3)
- [ ] Source `hindi_eval_50.jsonl` (50 Hindi prompts + references from MT-Bench-Hindi or Flores-200)
- [ ] Confirm Airavata 7B is accessible on remote server (model ID: `ai4bharat/airavata`)
- [ ] Implement `build_vocab_scan.py` (Day 1)
- [ ] Implement `samanvaya_parser.py` (Day 1-2)
- [ ] Implement `samanvaya_logits_processor.py` (Day 2)
- [ ] Implement `scg_generate.py` + `eval_scg.py` (Day 3)
- [ ] Run evaluation + AdaSD entropy gate (Day 4)
