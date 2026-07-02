# Samanvaya Rules: Inference-Side Injection Plan

**Scope:** Inference-side only — no tokenization changes, no pretraining, no fine-tuning.  
**Goal:** Inject Samanvaya LWG rules into language generation at decode time.  
**Timeline:** 1 week.  
**Note:** "Future Validity is the Missing Statistic" paper not yet in directory — plan accounts for its core concept based on title and context.

---

## What the Guide's Papers Are Collectively Saying

Reading all five papers together, the guide is pointing at a specific research direction:

| Paper | Core technique | What it solves for us |
|---|---|---|
| DOMINO (Guiding LLMs) | Constrained decoding via CFG + pre-computed token trees + speculative prediction | LWG rules → runtime token masks, zero overhead via speculation |
| AdaSD | Entropy-based adaptive draft stopping + Bayesian JS-distance verification | Decide WHERE and WHEN to apply Samanvaya constraints adaptively |
| Future Validity | Φ-estimation of grammar-faithful spec decoding; future-valid acceptance criterion | Ensure accepted speculative tokens don't violate future LWG completions |
| GTO | Draft tree reward = expected acceptance length; eliminate training-decoding misalignment | Rank draft branches by LWG coherence (inference-time reranking) |
| LTD | RL co-adaptive depth + size policies; reward = throughput per cycle | Learn when LWG constraints improve throughput vs hurt it |

The message is clear: **constrained speculative decoding where the grammar is Samanvaya's LWG rules**.

---

## Core Problem: Token–Word Boundary Alignment

Before any plan, the fundamental challenge:

Samanvaya rules operate at the **word level** (whitespace-delimited Hindi words). LLMs operate at the **subword token level**. The case marker "से" might be:
- Token ID 1234 exactly (single token) ← easy to detect
- Split as: "से" + continuation token ← harder
- Merged with the noun before it: "रामसे" (rare but possible) ← hardest

**Solution (from DOMINO's subterminal tree concept):**  
Pre-compute a **Hindi vocabulary scan table**: for each token ID in Airavata's vocabulary, determine which Samanvaya category it belongs to or could belong to:

```python
token_category = {
    1234: "CASE_MARKER",      # exact "से"
    5678: "CASE_MARKER_PARTIAL",  # "से" as suffix of a longer token
    9012: "AUX_VERB",         # exact "है"
    ...
}
```

This pre-computation runs once offline and enables O(1) lookup at each decode step.

---

## The Method: Samanvaya-Constrained Generation (SCG)

Three components, in order of implementation priority:

### Component 1: SamanvayaParser (State Machine)

Tracks LWG parse state across tokens. States correspond to what must happen next:

```
FREE       → no active LWG obligation; model generates freely
MUST_AUX   → last word ended in ा/ी/े, an auxiliary verb MUST follow
MUST_CASE  → after "के", next must be a compound postposition (लिए/बाद/साथ...)
MUST_CONT  → mid fixed-phrase (e.g., generated "हाल", "ही" must follow)
PARTIAL    → in the middle of a multi-token word, can't evaluate LWG yet
```

Transitions are deterministic given the decoded word and current state:

```python
class SamanvayaParser:
    def update(self, decoded_word: str) -> None:
        if self.state == "FREE":
            if decoded_word.endswith(EE_ENDING) or decoded_word.endswith(A_ENDING):
                self.state = "MUST_AUX"
                self.pending_word = decoded_word
            elif decoded_word in {"के", "की", "के"}:
                self.state = "MUST_CASE"
            elif decoded_word in FIXED_PHRASE_PREFIXES:
                self.state = "MUST_CONT"
                self.phrase_progress = [decoded_word]
        elif self.state == "MUST_AUX":
            if decoded_word in AUX_AFTER_EE or decoded_word in AUX_AFTER_A:
                self.state = "FREE"  # LWG complete
            else:
                self.state = "FREE"  # LWG broken — log violation
        # ... etc.
    
    def is_constrained(self) -> bool:
        return self.state != "FREE" and self.state != "PARTIAL"
    
    def valid_next_words(self) -> set[str]:
        """Return set of words that would be valid in current state."""
        if self.state == "MUST_AUX":
            return AUX_AFTER_EE | AUX_AFTER_A
        if self.state == "MUST_CASE":
            return {p[1] for p in RULE3_MULTIWORDS if p[0] in {"के", "की"}}
        ...
```

### Component 2: SamanvayaLogitsProcessor (Runtime Masking)

Plugs into HuggingFace's `generate()` via `LogitsProcessorList`. Uses **opportunistic masking** (DOMINO's key optimization): let the model propose first, only recompute mask if the proposal is invalid.

```python
from transformers import LogitsProcessor
import torch

class SamanvayaLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, parser: SamanvayaParser, 
                 vocab_scan: dict, opportunistic: bool = True):
        self.tokenizer = tokenizer
        self.parser = parser
        self.vocab_scan = vocab_scan        # pre-computed offline
        self.opportunistic = opportunistic
        self._token_buffer = []             # accumulates tokens within a word
    
    def __call__(self, input_ids: torch.LongTensor, 
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        # Update parser with last generated token
        last_token = input_ids[0, -1].item()
        self._update_parser_state(last_token)
        
        # If not in a constrained state: return scores unchanged
        if not self.parser.is_constrained():
            return scores
        
        if self.opportunistic:
            # First check if model's greedy choice is already valid
            greedy_token = scores.argmax(-1).item()
            if self._is_valid_token(greedy_token):
                return scores  # no intervention needed
        
        # Apply mask: allow only tokens from valid_next_words
        valid_token_ids = self._get_valid_token_ids()
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask[0, valid_token_ids] = True
        scores = scores.masked_fill(~mask, float('-inf'))
        return scores
    
    def _get_valid_token_ids(self) -> list[int]:
        """Pre-computed lookup: state → valid token IDs."""
        valid_words = self.parser.valid_next_words()
        return [
            tok_id for tok_id, category in self.vocab_scan.items()
            if self.vocab_scan[tok_id]['word'] in valid_words
        ]
    
    def _update_parser_state(self, token_id: int):
        decoded = self.tokenizer.decode([token_id])
        # Handle partial-word tokens (bridge tokens in DOMINO terminology)
        if decoded.startswith('▁') or token_id in WORD_START_TOKENS:
            # Word boundary — flush buffer and evaluate complete word
            if self._token_buffer:
                complete_word = self.tokenizer.decode(self._token_buffer)
                self.parser.update(complete_word)
            self._token_buffer = [token_id]
        else:
            self._token_buffer.append(token_id)
```

### Component 3: Grammar-Faithful Speculative Acceptance (Future Validity concept)

When using a drafter (EAGLE-2 or any draft model) with speculative decoding, the standard acceptance criterion (match target model distribution) is insufficient for grammar-constrained generation. A speculative token might be statistically accepted by the target model but put the generation in a state where NO valid LWG completion exists.

This is the core problem the "Future Validity" paper addresses. The fix, adapted from the paper's concept:

**Φ-modified acceptance criterion:**

```python
def grammar_faithful_accept(draft_token, target_prob, draft_prob, 
                             parser_state_after_draft) -> bool:
    """
    Accept draft token only if:
    1. Standard spec decoding criterion (target_prob / draft_prob) holds, AND
    2. The parser state after accepting this token still has valid continuations (future-valid).
    """
    # Standard speculative sampling acceptance
    alpha = min(1.0, target_prob / draft_prob)
    if random.random() > alpha:
        return False
    
    # Future validity check: does the parser state allow valid continuation?
    phi = parser_future_validity(parser_state_after_draft)
    if phi == 0.0:  # no valid continuation exists
        return False  # reject even if statistically accepted
    
    return True

def parser_future_validity(parser_state) -> float:
    """
    Returns Phi = probability that a valid LWG completion exists from this state.
    For deterministic states (MUST_AUX, MUST_CASE): Phi = 1 if valid_next_words() ≠ empty
    For FREE state: Phi = 1 (always future-valid)
    """
    if parser_state == "FREE":
        return 1.0
    valid = parser.valid_next_words()
    return 1.0 if valid else 0.0  # can be made probabilistic later
```

This is implementable without the full paper because the concept is: **before accepting a speculative token, simulate the parser state transition and check if valid continuations still exist**.

---

## Implementation Plan: Day by Day

### Day 1: Vocabulary Pre-computation

**Task:** Build `vocab_scan.json` — mapping from every Airavata token ID to its Samanvaya category.

```python
# build_vocab_scan.py
from transformers import AutoTokenizer
from wordgrouping_rules import ATTACH_TO_LEFT, AUX_AFTER_EE, AUX_AFTER_A, RULE3_MULTIWORDS

tokenizer = AutoTokenizer.from_pretrained("ai4bharat/airavata")
vocab = tokenizer.get_vocab()  # dict: str → int

vocab_scan = {}
for word, tok_id in vocab.items():
    clean = word.replace('▁', '').strip()  # remove sentencepiece prefix
    
    if clean in ATTACH_TO_LEFT:
        vocab_scan[tok_id] = {'word': clean, 'category': 'CASE_MARKER'}
    elif clean in AUX_AFTER_EE or clean in AUX_AFTER_A:
        vocab_scan[tok_id] = {'word': clean, 'category': 'AUX_VERB'}
    elif any(clean == phrase[0] for phrase in RULE3_MULTIWORDS):
        vocab_scan[tok_id] = {'word': clean, 'category': 'PHRASE_START'}
    elif any(clean == phrase[1] for phrase in RULE3_MULTIWORDS):
        vocab_scan[tok_id] = {'word': clean, 'category': 'PHRASE_CONT'}
    else:
        # Check if this token is a suffix containing a case marker
        for marker in ATTACH_TO_LEFT:
            if clean.endswith(marker) and len(clean) > len(marker):
                vocab_scan[tok_id] = {'word': clean, 'category': 'CASE_MARKER_MERGED'}
                break

# Pre-compute constrained state → valid token ID sets
state_valid_tokens = {
    'MUST_AUX': [tid for tid, v in vocab_scan.items() 
                 if v['category'] == 'AUX_VERB'],
    'MUST_CASE': [tid for tid, v in vocab_scan.items() 
                  if v['category'] in ('PHRASE_CONT', 'CASE_MARKER')],
}

import json
json.dump({'vocab_scan': vocab_scan, 'state_valid_tokens': state_valid_tokens},
          open('vocab_scan.json', 'w'), ensure_ascii=False)
```

**Deliverable:** `vocab_scan.json` with coverage statistics (how many Samanvaya words are exactly represented as single tokens vs multi-token).

---

### Day 2: SamanvayaParser + Unit Tests

**Task:** Implement the state machine from `wordgrouping_rules.py` logic, test on known examples.

```python
# samanvaya_parser.py
from wordgrouping_rules import (
    ATTACH_TO_LEFT, AUX_AFTER_EE, AUX_AFTER_A,
    RULE3_MULTIWORDS, RULE1_PHRASES, EE_ENDING, A_ENDING
)

PHRASE_PREFIXES = {p[0]: p for p in RULE1_PHRASES + RULE3_MULTIWORDS}

class SamanvayaParser:
    def __init__(self):
        self.state = "FREE"
        self.phrase_progress = []
        self.violations = 0
        self.completions = 0
    
    def update(self, word: str) -> str:
        """Process a complete word. Returns new state."""
        prev_state = self.state
        
        if self.state == "MUST_AUX":
            if word in AUX_AFTER_EE or word in AUX_AFTER_A:
                self.completions += 1
                self.state = "FREE"
            else:
                self.violations += 1
                # May still enter new LWG with this word
                self.state = "FREE"
                self.update(word)  # re-evaluate as FREE
        
        elif self.state == "MUST_CASE":
            if word in {p[1] for p in RULE3_MULTIWORDS if p[0] in ("के", "की")}:
                self.completions += 1
            else:
                self.violations += 1
            self.state = "FREE"
        
        elif self.state == "MUST_CONT":
            expected = PHRASE_PREFIXES.get(self.phrase_progress[0])
            pos = len(self.phrase_progress)
            if expected and pos < len(expected) and word == expected[pos]:
                self.phrase_progress.append(word)
                if len(self.phrase_progress) == len(expected):
                    self.completions += 1
                    self.state = "FREE"
                    self.phrase_progress = []
                # else stay in MUST_CONT
            else:
                self.violations += 1
                self.state = "FREE"
                self.phrase_progress = []
        
        elif self.state == "FREE":
            if word in PHRASE_PREFIXES:
                phrase = PHRASE_PREFIXES[word]
                if len(phrase) > 1:
                    self.state = "MUST_CONT"
                    self.phrase_progress = [word]
                # single-word phrases: nothing to enforce
            elif word.endswith(EE_ENDING) or word.endswith(A_ENDING):
                self.state = "MUST_AUX"
            # Note: case markers in FREE state are violations too (should be attached left)
            # but we handle that at prev-word level
        
        return self.state
    
    def valid_next_words(self) -> set:
        if self.state == "MUST_AUX":
            return AUX_AFTER_EE | AUX_AFTER_A
        if self.state == "MUST_CASE":
            return {p[1] for p in RULE3_MULTIWORDS}
        if self.state == "MUST_CONT":
            expected = PHRASE_PREFIXES.get(self.phrase_progress[0])
            pos = len(self.phrase_progress)
            return {expected[pos]} if expected and pos < len(expected) else set()
        return set()  # FREE: no restriction
    
    def is_constrained(self) -> bool:
        return self.state not in ("FREE", "PARTIAL")
    
    def compliance_rate(self) -> float:
        total = self.completions + self.violations
        return self.completions / total if total > 0 else 1.0
```

**Unit tests:**
```python
def test_parser():
    p = SamanvayaParser()
    # "राम_ने" — ने is case marker, should attach left
    # Test: aux verb chain
    p.update("जा")    # ends in ा → MUST_AUX
    assert p.state == "MUST_AUX"
    p.update("रहा")   # not in AUX sets → violation
    assert p.violations == 1
    
    p2 = SamanvayaParser()
    p2.update("जाती")  # ends in ी
    assert p2.state == "MUST_AUX"
    p2.update("है")    # in AUX_AFTER_EE → completion
    assert p2.completions == 1
    assert p2.state == "FREE"
```

---

### Day 3: LogitsProcessor Integration + First Run

**Task:** Wire up the processor and run constrained generation on 10 MT-Bench-Hindi questions.

```python
# scg_generate.py
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from samanvaya_parser import SamanvayaParser
from samanvaya_logits_processor import SamanvayaLogitsProcessor
import json

model = AutoModelForCausalLM.from_pretrained("ai4bharat/airavata")
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/airavata")
vocab_scan = json.load(open('vocab_scan.json'))

def generate_constrained(prompt: str, max_new_tokens: int = 200) -> str:
    parser = SamanvayaParser()
    processor = SamanvayaLogitsProcessor(
        tokenizer=tokenizer,
        parser=parser,
        vocab_scan=vocab_scan,
        opportunistic=True  # key: let model choose freely when possible
    )
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        logits_processor=LogitsProcessorList([processor]),
        do_sample=True,
        temperature=0.7,
    )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], 
                                skip_special_tokens=True)
    
    print(f"LWG compliance: {parser.compliance_rate():.2%}")
    print(f"Violations: {parser.violations}, Completions: {parser.completions}")
    return response
```

**Day 3 goal:** Get a first working constrained generation. Measure:
- LWG compliance rate (baseline unconstrained vs constrained)
- Overhead: tokens/s constrained vs unconstrained
- Qualitative check: do responses look more grammatically coherent?

---

### Day 4: Speculative Decoding Integration

**Task:** Add EAGLE-2 as drafter with grammar-faithful acceptance (Future Validity concept).

Two options for the speculative component:

**Option A: HuggingFace Assisted Generation (faster to implement)**
```python
# Uses assistant model for speculation, but without grammar-faithful acceptance
from transformers import AutoModelForCausalLM

assistant_model = AutoModelForCausalLM.from_pretrained("eagle2-airavata-draft")  
# if not available, use a smaller Hindi model

outputs = model.generate(
    **inputs,
    assistant_model=assistant_model,
    logits_processor=LogitsProcessorList([processor]),
    max_new_tokens=200,
)
```

**Option B: Custom Grammar-Faithful Speculative Loop (more novel, implements Future Validity)**
```python
def grammar_faithful_speculative_decode(
    model, draft_model, tokenizer, 
    prompt_ids, parser, vocab_scan,
    max_new_tokens=200, num_draft_tokens=5
):
    generated = list(prompt_ids[0])
    
    while len(generated) - len(prompt_ids[0]) < max_new_tokens:
        context = torch.tensor([generated])
        
        # 1. Draft model generates k tokens
        draft_tokens = []
        draft_probs = []
        draft_parser_states = []
        
        temp_parser = copy.deepcopy(parser)
        for _ in range(num_draft_tokens):
            with torch.no_grad():
                draft_logits = draft_model(context).logits[0, -1]
            
            # Apply Samanvaya constraints to drafter too
            if temp_parser.is_constrained():
                valid_ids = vocab_scan['state_valid_tokens'][temp_parser.state]
                mask = torch.full_like(draft_logits, float('-inf'))
                mask[valid_ids] = 0.0
                draft_logits = draft_logits + mask
            
            draft_probs_t = torch.softmax(draft_logits, dim=-1)
            draft_tok = torch.multinomial(draft_probs_t, 1).item()
            
            draft_tokens.append(draft_tok)
            draft_probs.append(draft_probs_t[draft_tok].item())
            draft_parser_states.append(copy.deepcopy(temp_parser))
            
            # Update temp parser
            decoded_word = tokenizer.decode([draft_tok]).replace('▁', '').strip()
            if decoded_word:
                temp_parser.update(decoded_word)
            
            context = torch.tensor([generated + draft_tokens])
        
        # 2. Target model verifies all draft tokens in one forward pass
        with torch.no_grad():
            target_logits = model(torch.tensor([generated + draft_tokens])).logits[0]
        
        # 3. Grammar-faithful acceptance (Future Validity criterion)
        accepted = 0
        for i, (draft_tok, draft_prob) in enumerate(zip(draft_tokens, draft_probs)):
            target_prob_dist = torch.softmax(target_logits[len(generated) - 1 + i], dim=-1)
            target_prob = target_prob_dist[draft_tok].item()
            
            # Standard speculative sampling acceptance
            alpha = min(1.0, target_prob / (draft_prob + 1e-10))
            
            # Future validity check
            future_valid = (draft_parser_states[i].valid_next_words() or 
                          not draft_parser_states[i].is_constrained())
            
            if random.random() < alpha and future_valid:
                generated.append(draft_tok)
                parser.update(tokenizer.decode([draft_tok]).replace('▁', '').strip())
                accepted += 1
            else:
                # Rejection: resample from corrected distribution
                corrected = torch.clamp(target_prob_dist - torch.tensor(draft_probs[i]), min=0)
                if corrected.sum() > 0:
                    corrected /= corrected.sum()
                    # Apply grammar constraint to corrected distribution
                    if parser.is_constrained():
                        valid_ids = vocab_scan['state_valid_tokens'].get(parser.state, [])
                        mask = torch.zeros_like(corrected)
                        mask[valid_ids] = 1.0
                        corrected *= mask
                        if corrected.sum() > 0:
                            corrected /= corrected.sum()
                    
                    new_tok = torch.multinomial(corrected, 1).item()
                    generated.append(new_tok)
                    parser.update(tokenizer.decode([new_tok]).replace('▁', '').strip())
                break
        
        if accepted == 0:
            pass  # Already handled rejection above
    
    return generated
```

**Day 4 goal:** Working speculative decode loop with grammar-faithful acceptance. Measure speedup ratio over autoregressive baseline.

---

### Day 5: AdaSD Entropy Gate at LWG Boundaries

**Task:** Add adaptive threshold for when to apply LWG constraints (from AdaSD).

Key insight from AdaSD: token entropy predicts acceptance. High entropy at a specific position → that token is uncertain → we SHOULD constrain it if in a relevant state. Low entropy → model is confident → constraint intervention may distort distribution unnecessarily.

```python
class AdaptiveSamanvayaProcessor(SamanvayaLogitsProcessor):
    def __init__(self, *args, initial_T_G=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.T_G = initial_T_G       # entropy generation threshold
        self.rejected_entropies = [] # online update buffer
    
    def __call__(self, input_ids, scores):
        # Update parser state
        last_token = input_ids[0, -1].item()
        self._update_parser_state(last_token)
        
        # Compute entropy of current distribution
        probs = torch.softmax(scores[0], dim=-1)
        H = -(probs * (probs + 1e-10).log()).sum().item()
        
        # Only constrain if:
        # 1. Parser is in constrained state AND
        # 2. Entropy is above threshold (model is uncertain, needs guidance)
        if self.parser.is_constrained() and H > self.T_G:
            valid_token_ids = self._get_valid_token_ids()
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask[0, valid_token_ids] = True
            scores = scores.masked_fill(~mask, float('-inf'))
        
        return scores
    
    def update_threshold(self, accepted: bool, entropy: float):
        """Called after each verification step (AdaSD update rule)."""
        if not accepted:
            self.rejected_entropies.append(entropy)
            if len(self.rejected_entropies) > 0:
                self.T_G = sum(self.rejected_entropies) / len(self.rejected_entropies)
```

**Logic:** In a MUST_AUX state (model generated verb ending in ा/ी), if H is low (model confidently predicts an auxiliary), we trust it and let it generate freely. If H is high (model is confused), we apply the mask to force a valid auxiliary.

---

### Day 6: Evaluation

**Metrics to measure:**

```python
# eval_scg.py
import json
from rouge_score import rouge_scorer

def evaluate(method_name, generate_fn, test_questions, reference_answers):
    results = {
        'method': method_name,
        'lwg_compliance': [],
        'tokens_per_sec': [],
        'rouge_l': [],
        'speedup_vs_ar': []
    }
    
    scorer = rouge_scorer.RougeScorer(['rougeL'])
    
    for question, reference in zip(test_questions, reference_answers):
        t0 = time.time()
        response, parser = generate_fn(question)
        t1 = time.time()
        
        n_tokens = len(tokenizer(response)['input_ids'])
        
        results['lwg_compliance'].append(parser.compliance_rate())
        results['tokens_per_sec'].append(n_tokens / (t1 - t0))
        results['rouge_l'].append(scorer.score(reference, response)['rougeL'].fmeasure)
    
    return {k: sum(v)/len(v) for k, v in results.items() if isinstance(v, list)}
```

**Baselines to compare:**
1. Airavata 7B autoregressive (no constraints, no spec decoding)
2. Airavata 7B + EAGLE-2 spec decoding (no LWG constraints)
3. **SCG: Airavata 7B + Samanvaya LogitsProcessor** (constrained, no spec decoding)
4. **SCG-SD: Airavata 7B + Samanvaya constraints + EAGLE-2 speculation** (full method)
5. **AdaSCG: SCG-SD + entropy gate** (adaptive version)

**Evaluation dataset:** MT-Bench-Hindi (200 questions) or a subset of 50 questions if time is limited.

---

### Day 7: Analysis + Write-up

Key questions to answer:
- What fraction of generation steps are in a constrained state? (Expected: ~15–25%)
- What fraction of those constraints are "opportunistically satisfied" (model already valid)?
- What is the overhead per constrained step vs unconstrained?
- Does LWG compliance correlate with human preference ratings?

---

## What Each Paper Gives Us (Concrete Reuse)

### DOMINO (no GitHub): Implemented from scratch
- Opportunistic masking: ✓ in `SamanvayaLogitsProcessor.__call__`
- Pre-computed token trees: ✓ in `vocab_scan.json` (Day 1)
- Speculative prediction: ✓ partially in Option B decode loop
- **What we skip:** full CFG parser machinery (DOMINO handles C, JSON, XML; ours is simpler)

### AdaSD (no GitHub): Implemented from scratch
- Entropy-based generation threshold T_G: ✓ in `AdaptiveSamanvayaProcessor`
- Online T_G update from rejected token entropies: ✓ in `update_threshold()`
- Bayesian JS-distance acceptance: ✗ (too complex for week-1; use standard speculative sampling)
- **Algorithm 1 from AdaSD paper** is self-contained enough to implement without repo

### Future Validity (no GitHub, paper not yet read): Concept-level implementation
- Future-validity check before accepting speculative tokens: ✓ in `grammar_faithful_speculative_decode`
- Φ-estimation: simplified as binary "does valid continuation exist?"
- Full Φ-estimation (probability-weighted): implement once paper is available

### GTO (GitHub: hsj576/GTO): Use for inspiration, not code
- Draft tree reward concept: used for Day 4 draft reranking
- PPO training: skip (inference-only week)

### LTD (GitHub: zhihao/Learning-to-Draft): Reference implementation
- Throughput reward signal: used as evaluation metric (λ_c = L_A / T_total)
- Co-adaptive policies: skip (inference-only week)

---

## File Structure

```
/home/pranavjs/Desktop/claude/
├── wordgrouping_rules.py          # existing — input to all below
├── samanvaya_parser.py            # Day 2: state machine
├── samanvaya_logits_processor.py  # Day 3: HF LogitsProcessor integration
├── build_vocab_scan.py            # Day 1: pre-compute token→category mapping
├── vocab_scan.json                # Day 1 output: pre-computed lookup
├── scg_generate.py                # Day 3: constrained generation entry point
├── scg_speculative.py             # Day 4: grammar-faithful spec decoding loop
├── eval_scg.py                    # Day 6: evaluation harness
└── results/
    ├── baseline_ar.json
    ├── baseline_eagle2.json
    ├── scg_constrained.json
    └── scg_speculative.json
```

---

## Expected Outcomes

| Method | LWG Compliance | Tokens/s | vs AR Baseline |
|---|---|---|---|
| AR (no constraints) | ~40–60% (natural) | 100% | 1.0× |
| SCG (constrained only) | ~85–95% | ~90–98% | 0.9–1.0× |
| SCG-SD (constrained + spec) | ~85–95% | ~180–250% | 1.8–2.5× |
| AdaSCG (entropy-adaptive) | ~80–90% | ~200–270% | 2.0–2.7× |

The key result: **LWG compliance increases significantly with near-zero throughput overhead** because DOMINO's opportunistic masking only intervenes when necessary.

---

## Critical Risks and Fallbacks

**Risk 1:** Airavata's tokenizer fragments Hindi words such that case markers are never single tokens.  
**Fallback:** Detect word boundaries by decoding the full buffer when a space/▁ token is generated, then apply parser retroactively.

**Risk 2:** Grammar constraints reduce fluency (model forced into suboptimal word choices).  
**Fallback:** Set entropy threshold T_G high enough that constraints are rarely applied; measure compliance vs quality tradeoff.

**Risk 3:** EAGLE-2 drafter is not available for Airavata 7B.  
**Fallback:** Use HuggingFace's built-in assisted generation with a smaller Hindi model (e.g., Param1-2.9B as drafter).

**Risk 4:** "Future Validity" paper has insights we're missing that change the approach.  
**Action:** Get the paper from guide ASAP; its Φ-estimation likely improves the acceptance criterion in Day 4.
