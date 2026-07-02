# New Method Proposal: Beyond WGA-SAM

**Status:** WGA-SAM is underperforming (2.76× vs 3.14× for unconstrained SAM).  
**Root cause:** Hard LWG boundary stopping reduces draft length (MAT 4.925 vs 5.117) without compensating benefit.  
**Source papers:** AdaSD, AdaSpec, GTO, DOMINO, LTD — all read and synthesized below.

---

## Diagnosis

The fundamental flaw in WGA-SAM: it treats LWG boundaries as **mandatory stopping points**. This is analogous to what the LTD paper calls "static time allocation" — a rigid policy that cannot adapt to context. The LTD paper proves empirically that maximizing acceptance length ≠ maximizing throughput. WGA-SAM sacrifices both.

Three compounding problems:
1. **Draft truncation penalty**: A 7-token draft with 4 tokens accepted beats a 3-token fully-correct draft.
2. **Static boundary enforcement**: Every LWG boundary is equally treated regardless of confidence.
3. **Drafter-verifier misalignment**: The verifier (Airavata 7B) was never trained on LWG structure, so even LWG-aligned drafts have no acceptance advantage.

The fix requires eliminating the hard stop while preserving the linguistic insight. The five papers provide four different paths.

---

## Method 1: Entropy-Gated LWG Stopping (2 days, no training)

**Source:** AdaSD (Lu et al., 2026)

**Core idea:** Replace the mandatory `if word_boundaries[pos]: break` with a soft entropy gate. Only stop at an LWG boundary when the draft model is genuinely uncertain — i.e., when continuing would likely produce a rejected token anyway.

### How it works

AdaSD shows that the entropy of the draft model's token distribution at step t (H(q_t)) is a strong predictor of whether that token will be accepted:
- **Low entropy** → model is confident → token likely accepted → continue past LWG boundary
- **High entropy** → model is uncertain → token likely rejected → stop at LWG boundary (natural exit)

This converts LWG boundaries from hard stops into **smart checkpoints**: the model only exits at an LWG boundary if it's already uncertain. When confident, it drafts through.

### Algorithm

```python
def gen_draft_adaptive(self, index, start_token, T_G=0.0):
    """
    Entropy-gated LWG-aware draft generation.
    T_G: entropy threshold (updated online from rejected token entropies).
    """
    pos = index
    draft = []
    
    while pos is not None and len(draft) < self.max_draft_len:
        # Get next token options from SAM state
        continuations = self.states[pos].next  # dict[token_id -> next_state]
        
        if not continuations:
            break
        
        if len(continuations) == 1:
            # Single continuation: zero entropy, always continue past LWG boundary
            token_id = next(iter(continuations))
            H = 0.0
        else:
            # Multiple continuations: compute entropy over SAM-reachable tokens
            # Use EAGLE-2 logits restricted to this continuation set
            logits = self.eagle_model.get_logits(context)  # shape: [vocab]
            mask = torch.zeros(vocab_size)
            mask[list(continuations.keys())] = 1.0
            masked_probs = softmax(logits * mask)  # renormalize over valid continuations
            H = -(masked_probs * masked_probs.log()).sum().item()
            token_id = sample(masked_probs)
        
        draft.append(token_id)
        pos = continuations[token_id]
        
        # ENTROPY-GATED LWG STOP
        if self.word_boundaries[pos]:
            if H > T_G:
                break  # uncertain at LWG boundary → stop (natural exit)
            # else: confident → continue past LWG boundary
    
    return draft

def update_T_G(self, rejected_entropies: list) -> float:
    """Online update: T_G = mean entropy of rejected tokens."""
    return sum(rejected_entropies) / len(rejected_entropies) if rejected_entropies else 0.5
```

### For Dynamic SAM (dominant contributor at 64% acceptance)

Dynamic SAM builds online from generation context. Entropy can be estimated from:
- **SAM branching factor**: if `len(state.next) == 1`, entropy = 0 (always continue)
- **EAGLE-2 logits**: when multiple SAM continuations exist, use the restricted logit distribution

### Expected outcome

- When Dynamic SAM has a unique continuation (common for high-frequency n-gram patterns): draft extends past LWG boundary → longer drafts → MAT increases
- When SAM branches or falls back to EAGLE-2: LWG boundary acts as a natural stopping gate
- Result: MAT should increase from 4.925 toward 5.0–5.1, recovering most of the gap to unconstrained SAM

### Implementation effort

**Day 1:**
- Modify `gen_draft` in `wga_sam.py` to accept `T_G` parameter and implement entropy gate
- Initialize T_G = 0.5 (reasonable default from AdaSD experiments)
- Collect rejected token entropies during inference for online update

**Day 2:**
- Run MT-Bench-Hindi evaluation with the new method
- Ablate over T_G ∈ {0.3, 0.5, 0.7, 1.0} (all no-retraining)
- Compare: WGA-SAM vs Entropy-Gated-WGA-SAM vs unconstrained SAM

---

## Method 2: LWG-Biased Endpos Candidate Reranking (1 day, no training)

**Source:** GTO (Hu et al., 2026), combined with existing multi-draft SAM

**Core idea:** WGA-SAM already generates k candidate sequences from `endpos_candidates`. Instead of truncating ALL drafts at LWG boundaries, keep full-length drafts but **rerank** candidates to prefer those whose accepted prefix ends at an LWG boundary.

This is zero-cost at inference — we just change the scoring function for which k candidates to send to the verifier.

### Algorithm

```python
def score_candidate(self, candidate_sequence, acceptance_probs):
    """
    Score = sum of acceptance probabilities + λ * LWG completion bonus.
    """
    base_score = sum(acceptance_probs)  # expected acceptance length
    
    # LWG bonus: count how many token positions are LWG boundaries
    lwg_bonus = sum(
        1.0 for i, tok_id in enumerate(candidate_sequence)
        if self.word_boundaries[self.state_at_position(i)]
    )
    
    return base_score + self.lambda_lwg * lwg_bonus

def gen_draft_candidates_reranked(self, index, start_token, k=5):
    """
    Generate k candidates WITHOUT truncation, then rerank by LWG score.
    """
    # Generate full-length candidates (no LWG truncation)
    candidates = self.gen_draft_candidates_unrestricted(index, start_token, k * 3)
    
    # Estimate acceptance probs using EAGLE-2 model probabilities
    scored = [(c, self.score_candidate(c, self.eagle_probs(c))) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    return [c for c, _ in scored[:k]]
```

### λ_lwg tuning

- λ_lwg = 0: identical to unconstrained SAM (upper bound on speedup)
- λ_lwg → ∞: identical to WGA-SAM (lower bound — our current broken method)
- λ_lwg ∈ {0.1, 0.3, 0.5}: the useful range — slight LWG preference without truncation

### Expected outcome

MAT stays near unconstrained SAM levels (≥5.0) but the accepted sequences are linguistically cleaner. This is the easiest path to match or exceed unconstrained SAM speedup.

---

## Method 3: Hindi-EAGLE with LWG Training (1–2 weeks, training required)

**Source:** AdaSpec (Do et al., 2026) + existing plan.md Direction C

**Core idea:** The current EAGLE-2 drafter was trained on English ShareGPT data. AdaSpec shows that language-specific training data is the single biggest factor for non-English speculative decoding. Train a new EAGLE-2 drafter on:
1. Self-synthesized Hindi instruction data (using Airavata 7B itself)
2. Annotated with LWG boundaries from `wordgrouping_rules.py`
3. With LWG boundary auxiliary loss added to EAGLE-2's training objective

### Training objective

```
L_total = L_reg + w_cls * L_cls + λ_lwg * L_boundary
```

Where:
- `L_reg`: SmoothL1 loss on feature embeddings (standard EAGLE-2)
- `L_cls`: CrossEntropy on token prediction (standard EAGLE-2)
- `L_boundary`: Binary cross-entropy on boundary prediction
  - For each token position t: BCE(p_t, is_last_token_in_LWG(t))
  - This teaches the drafter to predict where LWG boundaries are

### Data synthesis pipeline

```python
# Step 1: Generate Hindi instruction-response pairs
responses = airavata_7b.generate(hindi_prompts, n=10000, temperature=1.0)

# Step 2: Annotate with LWG boundaries
for response in responses:
    grouped = group_sentence(response)          # from wordgrouping_rules.py
    boundaries = get_boundary_mask(grouped)     # bool list, True at last token of each LWG

# Step 3: Train EAGLE-2 with extended objective
trainer = EAGLETrainer(
    base_model=airavata_7b,
    loss_fn=LWGAwareLoss(lambda_lwg=0.1)
)
```

### Vocabulary simplification (from AdaSpec)

AdaSpec shows that using only the top-k most frequent tokens for the target language reduces LM head computation. For Hindi:
```python
# Count token frequencies in Hindi corpus
token_freqs = Counter(tokenize(hindi_corpus))
top_8k_hindi_tokens = token_freqs.most_common(8000)

# Create restricted LM head projection
W_LM_hindi = W_LM_full[top_8k_hindi_tokens, :]  # shape: [8000, d]
```

This directly reduces draft model forward pass time.

### Expected outcome

AdaSpec reports 2.3× speedup over EAGLE-2 for language-specific training. Applied to Hindi, a properly Hindi-tuned EAGLE-2 should achieve higher acceptance rates than the current English-trained EAGLE-2, potentially pushing MAT from ~5.0 toward 5.5–6.0.

---

## Method 4: LWG-Aware Throughput RL (2–3 weeks, most complete)

**Source:** LTD (Zhang et al., 2026) + GTO (Hu et al., 2026)

**Core idea:** Reformulate the LWG-aware draft stopping as an RL problem where the reward is directly throughput per cycle. Two co-adaptive policies replace all hard constraints:

- **Depth policy π_D**: at each draft step, decide CONTINUE or STOP
- **Size policy π_V**: after draft tree is built, select how many candidates to send for verification

### State representation with LWG awareness

```python
state = [
    current_depth,              # how deep in draft tree
    context_length,             # total sequence length so far
    token_probs_at_frontier,    # W^2 probability scores (from EAGLE-2)
    lwg_completion_fraction,    # tokens since last LWG boundary / avg LWG length
    entropy_at_frontier,        # uncertainty signal (from AdaSD)
]
```

The key addition over vanilla LTD: `lwg_completion_fraction`. This tells the policy whether we're in the middle of an LWG (dangerous to stop) or at a natural completion point (safe to stop).

### Reward

```
R = L_A / T_total = accepted_tokens / (draft_time + verify_time)
```

Same as LTD — throughput per cycle. No LWG reward manipulation: the policies LEARN that stopping mid-LWG wastes draft time (because the verifier rejects partial LWGs more often for Hindi).

### Training

Phase 1: Pre-train on Airavata 7B + existing SAM drafts (PPO, 100k steps)
Phase 2: Joint co-adaptation (alternating policy updates, 2 iterations)

### Expected outcome

LTD achieves 36.4% improvement over EAGLE-3 baseline. Applied with LWG-awareness in the state, this method should:
- Always match or beat unconstrained SAM (3.14×)
- Potentially exceed it by 15–25% for Hindi-specific content
- Degrade gracefully on multilingual/mixed content

---

## Recommended New Direction: Two-Stage Plan

### Immediate (this week): Entropy-Gated WGA-SAM + Candidate Reranking

Combine Method 1 and Method 2 — both are no-training modifications:

```
New draft pipeline:
1. Try Dynamic SAM first
   - At each LWG boundary: check entropy gate
   - If H(q_t) ≤ T_G: continue past boundary (Method 1)
   - If H(q_t) > T_G: stop at boundary (natural exit)
2. Generate k candidates using endpos_candidates
3. Rerank candidates by (expected_acceptance + λ_lwg * lwg_completion_bonus) (Method 2)
4. Fallback to EAGLE-2 for positions not covered by SAM
```

This removes the hard truncation while preserving LWG preference via soft reranking. Expected to recover most of the 3.14× gap.

**Code changes needed:**
- `wga_sam.py: gen_draft()` — add entropy check at LWG boundary
- `wga_sam.py: gen_draft_candidates()` — add reranking step
- Add `compute_entropy()` helper using masked EAGLE-2 logits
- Add `update_T_G()` online update from rejected token stats

### Medium-term (2–4 weeks): Hindi-EAGLE Training

Run AdaSpec-style self-synthesis:
1. Generate 20k Hindi instruction-response pairs from Airavata 7B
2. Apply `wordgrouping_rules.py` to annotate LWG boundaries
3. Fine-tune EAGLE-2 drafter with L_boundary auxiliary loss
4. Evaluate with the entropy-gated stopping from Stage 1

### Long-term (1–2 months): LWG-Aware RL Policy

Implement LTD-style co-adaptive policies with LWG state features. This is the full contribution for the EMNLP 2026 submission.

---

## Why This Direction Is Novel

| Paper | Contribution | Our extension |
|---|---|---|
| AdaSD | Entropy-based adaptive stopping | LWG boundaries as entropy checkpoints, not hard stops |
| AdaSpec | Language-specific drafter training | Hindi-specific EAGLE with LWG boundary loss |
| GTO | Draft tree reward aligns with decoding policy | LWG coherence bonus in draft tree reward |
| DOMINO | Minimally invasive constrained generation | LWG grammar as DOMINO constraint (future) |
| LTD | Throughput RL with co-adaptive policies | LWG_completion_fraction in RL state |

The key novelty: **every existing method treats linguistic structure as external to the decoding optimization loop**. We embed LWG structure inside the optimization objective — either as entropy checkpoints, candidate reranking signals, training signals, or RL state features.

---

## Comparison Table: Old vs New

| Metric | WGA-SAM (old) | Entropy-Gated (Method 1) | Reranking (Method 2) | Hindi-EAGLE (Method 3) |
|---|---|---|---|---|
| Requires training | No | No | No | Yes (1–2 weeks) |
| Draft truncation | Hard stop at all LWG boundaries | Stop only when uncertain | No truncation | No truncation |
| LWG respect | Forced | Adaptive | Preference via reranking | Learned |
| Expected MAT | 4.925 | ~5.0–5.1 | ~5.0–5.2 | ~5.5–6.0 |
| Expected speedup | 2.76× | ~2.95–3.05× | ~3.0–3.1× | ~3.3–3.8× |
| Implementation time | Done | 1–2 days | 1 day | 1–2 weeks |

---

## Immediate 2-Day Implementation Plan

### Day 1 — Entropy-Gated Stopping

**Morning (3h):**
1. Add `compute_draft_entropy()` to `wga_sam.py`:
   - For SAM states with 1 continuation: entropy = 0.0
   - For SAM states with k>1 continuations: compute entropy over EAGLE-2 logits masked to valid tokens
2. Modify `gen_draft()`: replace `if word_boundaries[pos]: break` with entropy gate
3. Add `T_G` as a parameter with default 0.5

**Afternoon (3h):**
4. Add online `T_G` update in the main decode loop (collect rejected entropies → take mean)
5. Quick smoke test on 5 MT-Bench-Hindi prompts
6. Confirm draft lengths increase (MAT > 4.925)

### Day 2 — Evaluation and Ablation

**Morning (3h):**
1. Run full MT-Bench-Hindi evaluation (200 questions × Airavata 7B verifier)
2. Compare: WGA-SAM | Entropy-Gated | Unconstrained SAM across all metrics

**Afternoon (3h):**
3. Ablate T_G: {0.0 (= unconstrained), 0.3, 0.5, 0.7, 1.0 (= WGA-SAM), adaptive}
4. Optional: add candidate reranking (Method 2) as additional experiment
5. Write up results, update plan.md checkboxes

### Success criteria for this week

- [ ] Entropy-gated WGA-SAM MAT ≥ 5.0 (recovers from 4.925 toward unconstrained 5.117)
- [ ] Speedup ≥ 3.0× (vs WGA-SAM's 2.76×, approaching unconstrained 3.14×)
- [ ] Zero accuracy regression (acceptance distribution preserved — method is still lossless)
- [ ] T_G adaptive update converges within 20 steps
