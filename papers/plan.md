# Research Plan: Word-Group-Informed Multi-Token Generation for Indian Languages

**Project:** Extension of MTP Stage II — Speculative Multi-Token Generation with Linguistically-Informed Constraints  
**Target Venue:** EMNLP 2026 (ACL ARR Oct cycle) or ECIR 2027  
**Primary Language:** Hindi (Devanagari), extensible to Marathi and other Indic languages  
**Supervisor:** Prof. Ganesh Ramakrishnan, IIT Bombay

---

## 1. Background and Motivation

### 1.1 What Has Been Done (MTP Stage II)

The thesis introduced the **Word-Group-Aware Suffix Automaton (WGA-SAM)** framework, which integrates Samanvaya Local Word Groups (LWGs) as hard boundary constraints into the retrieval-based speculative decoding pipeline (SAM-Decoding + EAGLE-2). The core idea: the drafter's `gen_draft` algorithm terminates at the nearest LWG boundary, ensuring every proposed draft is a linguistically complete semantic unit.

**Key results on MT-Bench-Hindi (200 questions, Airavata 7B verifier):**

| Configuration | MAT | Tokens/s | Speedup |
|---|---|---|---|
| EAGLE-2 (baseline) | 3.391 | 156.78 | 1.488× |
| EAGLE-2 + SAM | 5.117–5.983 | 324–330 | 3.08–3.14× |
| EAGLE-2 + WGA-SAM (ours) | 4.925 | 290–299 | 2.76–2.85× |

**Core finding:** WGA-SAM produces linguistically sound drafts but is ~10% slower in speedup than unconstrained SAM due to **draft truncation** — shorter LWG-bounded drafts yield fewer tokens per verification step even though acceptance rate per token is higher.

**Root cause of the gap:**
1. Static SAM acceptance rate is only 2–4% (corpus too small — 19k WikiChat rows)
2. Dynamic SAM is the dominant contributor (64% acceptance rate) but is blind to word-group structure
3. The verifier LLM never "learns" about word groups — structural knowledge is siloed in the drafter

### 1.2 The Central Hypothesis

> **Word groups are not just syntactic boundaries — they are the fundamental units of semantic planning in Indian languages. A model that internalizes word-group structure will exhibit better context understanding, more coherent inference trajectories, and higher draft acceptance rates.**

The thesis proved the first claim at the retrieval level. The next phase tests whether this hypothesis holds when word-group awareness is baked into: (a) tokenization, (b) the draft model's training objective, and (c) the inference decoding algorithm itself.

---

## 2. Problem Statement

Standard multi-token generation methods applied to Hindi/Indian languages fail in three ways:

1. **Tokenization fragmentation:** BPE-based subword tokenizers split Hindi words arbitrarily, breaking apart morphologically meaningful units (e.g., राम_ने split across multiple tokens with no respect for the NE+case-marker boundary).
2. **Drafter blindness:** Draft models (EAGLE, smaller LMs) are trained on raw token sequences and cannot distinguish intra-group from inter-group boundaries, leading to proposals that cross semantic unit boundaries.
3. **Verifier unawareness:** The target LLM has no signal to prefer word-group-coherent completions, so even if the drafter proposes valid drafts, the verifier's probability distribution may not align with them.

The goal is to address all three levels — tokenization, training, and inference — in a unified framework.

---

## 3. Proposed Directions

The plan is organized into four directions in order of implementation complexity and expected impact. Each direction is independent and can produce standalone results.

---

### Direction A: Solving the Static Corpus Coverage Problem (Near-term, ~4 weeks)

**Problem:** The Static SAM's 2–4% acceptance rate is the single largest bottleneck. This stems from training on only 19k WikiChat rows. The linguistic constraints in WGA-SAM are sound but underutilized because the Static SAM rarely fires.

**Approach: Large-scale LWG-annotated corpus construction**

1. **Source expansion:** Extend beyond WikiChat to the full Indic-Instruct dataset (IndicLLMSuite), OPUS Hindi corpora, CC-100 Hindi, and SANGRAHA. Target 500k–2M sentences.
2. **Scale `wordgrouping_rules.py`:** The existing rule-based pipeline processes each sentence in O(n) time. Run it at scale to produce an LWG-annotated JSONL corpus of the form `{"tokens": [...], "word_group_boundaries": [...]}`.
3. **Build a large WGA-Static SAM:** Rebuild the Static SAM on this corpus. Expected outcome: the Static SAM's match rate rises from ~2% to 20–40%, making it a genuine contributor alongside the Dynamic SAM.
4. **Ablation:** Report Static SAM acceptance rate vs corpus size to characterize the scaling law.

**Success metric:** Static SAM acceptance rate exceeds 15%. Overall WGA-SAM speedup approaches or exceeds unconstrained SAM.

**Files to modify:**
- [wordgrouping_rules.py](wordgrouping_rules.py) — add batch processing and parallelization
- SAM construction scripts — rebuild with the larger corpus

---

### Direction B: Word-Group-Aware Dynamic SAM (Adaptive Linguistic Drafter, ~6 weeks)

**Problem:** The Dynamic SAM learns query-specific token repetitions but remains blind to word-group structure. It cannot constrain its proposals to LWG boundaries because it has no mechanism to detect word-group boundaries on-the-fly for newly generated text.

**Approach: Online LWG Boundary Detector**

Train a lightweight boundary classifier that, given the current token sequence generated so far, predicts for each position whether it is a word-group boundary. This classifier runs alongside the Dynamic SAM.

**Sub-approach B1: Rule-based online annotator**

The existing `wordgrouping_rules.py` already implements a deterministic pipeline. Apply it online to the accepted tokens after each speculative step:
1. After accepting tokens from the verifier, run `group_sentence()` on the new text to detect boundaries.
2. Feed boundary flags into the Dynamic SAM's `add_state` loop so newly added states carry boundary information.
3. The Dynamic SAM's `gen_draft` then terminates at detected boundaries.

This converts the "offline-only" constraint into a partially online one at near-zero latency cost (rule-based, no model inference).

**Sub-approach B2: Learned boundary classifier**

Train a small BiLSTM or linear-probe on top of the verifier's hidden states to predict `word_group_boundaries[i]` for each accepted token. This classifier:
- Is called once per verification step on the accepted prefix
- Outputs boundary flags that update the Dynamic SAM
- Can generalize to new vocabulary and complex morphology beyond the rule set

**Training data:** The LWG-annotated corpus from Direction A. Binary classification per token: is this the last token of an LWG?

**Architecture:** Single linear layer on top of the last accepted hidden state from Airavata/Param — minimal overhead (~0.1ms per step).

**Success metric:** Dynamic SAM acceptance rate improves from 64% to >70% while maintaining linguistic validity. Drafts no longer cross word-group boundaries even for unseen text patterns.

---

### Direction C: LWG-Informed Draft Model Training (Medium-term, ~8 weeks)

**Problem:** The EAGLE-2 autoregression head is trained purely to predict the next token's embedding. It has no awareness of word-group structure, so its tree-of-candidates often proposes continuations that split LWGs.

This direction teaches the draft model to internalize LWG structure through auxiliary training objectives, replacing the rigid hard-stop constraint with a learned soft preference.

#### C1: Boundary-Prediction Auxiliary Loss

Augment the EAGLE-2 training objective with a boundary prediction head:

```
L_Total = L_reg + w_cls * L_cls + λ₁ * L_boundary + λ₂ * L_contrastive
```

Where:
- `L_reg` = SmoothL1 loss on feature prediction (existing)
- `L_cls` = CrossEntropy on token prediction (existing)
- `L_boundary` = BinaryCrossEntropy on predicting `word_group_boundaries[i]` for each generated token position
- `L_contrastive` = pull hidden representations of tokens within the same LWG closer together, push tokens from different LWGs apart

The boundary prediction head is a 2-class linear classifier on the autoregression head's output at each step. During inference, the EAGLE head now proposes tokens AND predicts whether the proposed token ends an LWG. This allows the drafter to self-terminate at predicted boundaries rather than relying solely on the rule-based lookup.

**Training setup:**
- Base model: Airavata 7B (or Param1 2.9B) as the backbone
- Data: LWG-annotated WikiChat + expanded corpus from Direction A
- Boundary labels derived from `wordgrouping_rules.py` output
- Same training procedure as EAGLE-2 but with additional boundary head

#### C2: On-Policy Self-Distillation with LWG Rewards (GKD/SDFT Style)

Inspired by [Agarwal et al., 2024] (GKD) and [Shenfeld et al., 2026] (SDFT):

**Teacher-Student Setup:**
- **Teacher policy**: Airavata 7B receiving input + Samanvaya word-group structure as a conditioning variable `c`
- **Student policy**: Airavata 7B receiving only the input (no word-group conditioning)
- Both policies share weights; the conditioning variable is prepended as a special prefix or embedded separately

**On-Policy Training Signal:**
1. Student generates trajectories autoregressively
2. Teacher evaluates student trajectories and provides token-level probability feedback
3. Student is trained to minimize KL divergence from teacher ON ITS OWN GENERATED OUTPUT (not on reference text)

**Linguistically-Informed Loss:**

```
L_Total = L_KD + λ₁ * L_boundary + λ₂ * L_contrastive
```

- `L_KD` = token-level KL divergence (student vs teacher)
- `L_boundary` = boundary prediction loss (binary classification per token)
- `L_contrastive` = within-group cohesion loss on embeddings

**Why this is better than standard KD:** The student is trained on its own trajectory, which avoids exposure bias. The teacher's word-group conditioning gives it a better probability distribution over completions that respect LWG structure. The student learns to implicitly model this structure without needing word-group annotations at inference time.

**Expected outcome:** The fine-tuned model generates text that is intrinsically more likely to produce LWG-complete continuations, increasing the acceptance rate when this model is used as either the verifier or the drafter.

---

### Direction D: Constrained Decoding at Inference Time (~6 weeks, parallel with C)

This direction uses word-group constraints directly in the inference algorithm rather than in training. It is complementary to C and can be applied to any frozen LLM.

#### D1: Logit-Masking Constrained Decoding

At each token generation step, consult the current partial word-group context:
- If the current token is **not** the end of an LWG (determined by a fast rule-based or classifier-based predictor), restrict the next token's logit distribution to the set of tokens that are plausible continuations of the current partial LWG.
- If the current token **is** the end of an LWG, allow the full distribution.

**Implementation:**
1. Build a trie of all valid LWG token sequences from the annotated corpus
2. At each step, traverse the trie using the current partial LWG to obtain the set of valid next tokens
3. Mask out invalid tokens (set logits to −∞) before softmax

**Tradeoff:** This is a hard constraint — it guarantees every generated sequence decomposes into valid LWGs, but may hurt performance if the trie coverage is incomplete (unseen LWG patterns get rejected). Use a soft version: add a bonus `α * IsValidGroupToken(t)` to logit rather than masking.

#### D2: LWG-Aware Lookahead Decoding

Lookahead decoding generates multiple candidate branches in parallel using n-gram predictions. Extend it with word-group awareness:
- The lookahead branch generation is constrained to not cross LWG boundaries
- Candidate selection uses an LWG-coherence score as a tiebreaker: among candidates with equal prefix match length, prefer the one that ends on a word-group boundary

#### D3: Word-Group-Constrained Beam Search

For generation tasks where quality matters more than latency (e.g., machine translation):
- Beam search where each hypothesis maintains an LWG boundary tracker
- Incomplete LWG hypotheses receive a small penalty `β` per non-boundary position
- Complete LWG hypotheses receive a bonus `γ`

This directly tests the hypothesis that word-group-coherent beams produce better MT outputs (measured by BLEU/chrF).

---

### Direction E: LWG-Aware Tokenization (Longer-term, ~12 weeks)

This is the most fundamental intervention — modifying the tokenizer to produce tokens that align with LWG boundaries. Inspired by the thesis's future work and [Liu et al., 2025] (SuperBPE) and [Schmidt et al., 2025] (BoundlessBPE).

#### E1: Constrained-Merge BPE

Modify the BPE merge algorithm: two adjacent pre-tokens can be merged **only if** they belong to the same pre-annotated Samanvaya word group. Specifically:
- Pre-process the training corpus with `wordgrouping_rules.py` to annotate LWG boundaries
- During BPE training, add a constraint: `merge(a, b)` is only performed if the boundary flag between `a` and `b` is `False` (i.e., they are within the same LWG)

This ensures the resulting vocabulary never contains merge tokens that cross an LWG boundary.

#### E2: Group-as-Pretoken

Instead of whitespace-separated words as BPE pre-tokens, use Samanvaya word groups as the indivisible base units:
- `राम_ने` → single pre-token (not splittable by BPE)
- `घर_में` → single pre-token
- BPE then operates on sequences of these group-level pre-tokens, learning merges at the group level

**Expected impact:** Each BPE token corresponds to at most one complete LWG. The model's vocabulary is semantically denser. Multi-token generation per step is more meaningful — one "token" step corresponds to one or more complete semantic units.

#### E3: Score-Modified BPE

Add a linguistic bonus to the BPE merge scoring function:

```
Score(a, b) = Freq(a, b) + α * IsInSameGroup(a, b)
```

where `IsInSameGroup(a, b)` returns 1 if the pair (a, b) always co-occurs within the same LWG in the annotated corpus. This is a soft version — it biases merges toward LWG-coherent pairs without hard prohibition.

**Evaluation:** Retrain Airavata or a smaller baseline (Param 2.9B or Qwen 3 8B) on the new tokenized corpus. Measure: vocabulary alignment with LWGs, downstream task performance on MT-Bench-Hindi, generation coherence.

---

## 4. Experimental Plan

### 4.1 Models (Baselines and Verifiers)

| Model | Parameters | Use |
|---|---|---|
| Airavata | 7B | Primary verifier (Hindi-specific, OpenHaathi base) |
| Param1-Instruct | 2.9B | Small Hindi verifier / draft model candidate |
| Krutrim 2 | 12B | Hindi-focused verifier |
| Qwen 3 | 8B/14B | Multilingual baseline |
| Qwen 3.5 | 9B | Multilingual baseline |
| Gemma 3 | 12B | Multilingual baseline |
| Llama 4 Scout | — | Large multilingual verifier |
| Llama 3.1-8B-Instruct | 8B | Strong English-baseline comparison |

For speculative decoding experiments, EAGLE-2 weights are available for Llama 3.1. For Airavata and Param, EAGLE-2 training must be done from scratch using the WGA-annotated corpus.

### 4.2 Datasets

**Training / SAM construction:**
- Indic-Instruct WikiChat (Hindi) — 19k rows (current)
- Full Indic-Instruct suite — 500k+ rows (Direction A expansion)
- CC-100 Hindi / SANGRAHA (for tokenizer training in Direction E)

**Evaluation:**
- **MT-Bench-Hindi** (200 multi-turn questions, 8 categories) — primary benchmark, measures conversational speedup and quality
- **Flores-200 Hindi dev set** — machine translation quality (BLEU, chrF)
- **IndicSentEval** or similar — sentence completion / coherence measurement
- **Samanvaya word-grouping test set** — direct measure of LWG boundary prediction accuracy

### 4.3 Metrics

**Efficiency (Speculative Decoding):**
- Mean Accepted Tokens (MAT) per verification step
- Tokens/second (wall-clock throughput)
- Speedup ratio vs. autoregressive baseline

**Quality (Generation):**
- BLEU, chrF (vs. reference or vs. original LLM output)
- Word-group boundary violation rate (% of generated sequences that cross an LWG boundary mid-token)
- Human evaluation: fluency, grammaticality, coherence (Hindi speakers)

**Linguistic Alignment:**
- LWG boundary F1: precision and recall of predicted boundaries vs. rule-based annotations
- Surprisal analysis: perplexity of the model on LWG-complete vs. LWG-incomplete sequences

### 4.4 Ablations

For each direction, maintain the following ablation structure:

1. **No word grouping:** Standard EAGLE-2 + unconstrained SAM (thesis baseline)
2. **Hard boundary (static only):** WGA-SAM with Static SAM constraints only (thesis result)
3. **Hard boundary (static + dynamic):** Full WGA-SAM from thesis
4. **+Direction A:** Larger static corpus
5. **+Direction B:** Online dynamic boundary annotation
6. **+Direction C:** Boundary-aware EAGLE training
7. **+Direction D:** Constrained decoding at inference
8. **Full system:** All directions combined

---

## 5. Implementation Roadmap

### Phase 1 (Weeks 1–4): Foundation and Corpus Scaling

**Goal:** Address the static corpus coverage bottleneck and establish reliable baselines across multiple models.

- [ ] Scale `wordgrouping_rules.py` to process 500k+ sentences efficiently (batch + multiprocessing)
- [ ] Download and pre-process extended Hindi corpus (Indic-Instruct full + SANGRAHA subset)
- [ ] Run LWG annotation pipeline at scale; produce large `grouped.jsonl`
- [ ] Build new WGA-Static SAM from large corpus; measure acceptance rate improvement
- [ ] Set up EAGLE-2 training pipeline for Airavata and Param1
- [ ] Evaluate Llama 4 Scout + existing EAGLE weights on MT-Bench-Hindi as new baseline
- [ ] Reproduce thesis results on new hardware/model versions to confirm baseline numbers

**Deliverable:** Acceptance rate table showing Static SAM hit rate vs corpus size; updated Table 5.1 with Llama 4 Scout baseline.

### Phase 2 (Weeks 5–8): Online LWG Detection and Dynamic SAM Integration

**Goal:** Make the Dynamic SAM word-group aware using the rule-based online annotator.

- [ ] Implement online rule-based boundary detector: after each verification step, run `group_sentence()` on accepted text and update Dynamic SAM with boundary flags
- [ ] Modify Dynamic SAM's `add_state` and `gen_draft` to be boundary-aware (mirror WGA-Static SAM logic)
- [ ] Train lightweight boundary classifier (linear probe on Airavata hidden states) using Direction A corpus
- [ ] Evaluate: compare dynamic SAM acceptance rate before vs. after LWG awareness
- [ ] Report: word-group boundary violation rate in Dynamic SAM drafts before vs. after

**Deliverable:** Updated speculative decoding results with LWG-aware Dynamic SAM; violation rate analysis.

### Phase 3 (Weeks 7–12): EAGLE Training with Boundary Auxiliary Loss

**Goal:** Integrate word-group awareness into the EAGLE autoregression head training.

- [ ] Implement boundary prediction head on top of EAGLE-2 autoregression head architecture
- [ ] Modify training loss to include `L_boundary` (binary CE for each token's boundary flag)
- [ ] Train on LWG-annotated corpus; tune λ₁ weight for boundary loss
- [ ] Implement contrastive loss `L_contrastive` for within-group token representation cohesion (optional, test value)
- [ ] Evaluate EAGLE boundary F1 and its impact on draft acceptance rate
- [ ] Run full speculative decoding pipeline with new EAGLE head; compare to thesis results

**Deliverable:** EAGLE-2 variant with boundary auxiliary loss; ablation showing impact of each loss component.

### Phase 4 (Weeks 10–14): RL / Self-Distillation Direction

**Goal:** Teach the base LLM to prefer LWG-coherent continuations through on-policy distillation.

- [ ] Implement GKD-style training: teacher = Airavata with LWG prefix; student = Airavata without
- [ ] Implement LWG-conditioned prefix: encode word-group annotations as a structured prefix (e.g., `<LWG>राम_ने</LWG> <LWG>घर_में</LWG>...`)
- [ ] Generate student trajectories on-policy; compute teacher feedback at each student-generated token
- [ ] Train student with `L_KD + λ₁*L_boundary + λ₂*L_contrastive`
- [ ] Evaluate fine-tuned model: MAT improvement, BLEU on Flores-200, boundary violation rate

**Deliverable:** LWG-aware fine-tuned Airavata/Param; comparison of LWG-conditioned vs. standard generation.

### Phase 5 (Weeks 12–16): Constrained Decoding and Integration

**Goal:** Implement inference-time constrained decoding (logit masking or soft bonus) and combine all components.

- [ ] Build LWG trie from annotated corpus for fast lookup
- [ ] Implement soft logit bonus (`L_valid_group`) at inference time
- [ ] Evaluate: constrained decoding speedup/quality tradeoff
- [ ] Evaluate LWG-aware beam search on Flores-200 MT task
- [ ] Run full combined system: WGA-SAM + online boundary detector + boundary-aware EAGLE + constrained decoding
- [ ] Run on multiple verifiers: Airavata, Param1, Krutrim 2, Qwen 3 8B

**Deliverable:** Full system results across all models; final Table showing each component's contribution.

### Phase 6 (Weeks 15–20): Tokenization and Generalization

**Goal:** Test LWG-aware tokenization and extend to Marathi.

- [ ] Implement constrained-merge BPE (E1): modify tokenizer training to block cross-boundary merges
- [ ] Implement group-as-pretoken BPE (E2): pre-group corpus before BPE training
- [ ] Evaluate new tokenizers on: vocabulary-LWG alignment, downstream perplexity, MT-Bench-Hindi
- [ ] Extend `wordgrouping_rules.py` with Marathi-specific rules (if Samanvaya annotations available)
- [ ] Run WGA-SAM on Marathi with Airavata or Krutrim 2 as verifier
- [ ] Cross-lingual analysis: do LWG constraints transfer across Hindi↔Marathi?

**Deliverable:** Tokenization ablation results; Marathi evaluation showing cross-lingual generalizability.

---

## 6. Key Design Decisions and Tradeoffs

### 6.1 Hard vs. Soft LWG Constraints

**Hard constraints** (current WGA-SAM approach): Draft is always terminated at an LWG boundary. Guarantees linguistic validity but reduces draft length → lower MAT.

**Soft constraints** (proposed): Add an LWG-completion bonus to the acceptance criterion or scoring. A draft that ends mid-LWG is not rejected outright — instead, partial credit is given for the accepted prefix up to the last LWG boundary. This balances throughput and linguistic coherence.

**Recommendation:** Implement soft constraints in Direction D first. If they match or exceed hard constraints while improving MAT, migrate WGA-SAM to soft mode.

### 6.2 Rule-Based vs. Model-Based Boundary Detection

The current `wordgrouping_rules.py` is deterministic and fast but limited in coverage (fixed lexicons and morphological rules). A learned classifier generalizes better but adds latency.

**Recommendation:** Use rule-based detection for Dynamic SAM updates (Phase 2) since it is zero-latency. Deploy learned classifier only if the rule-based approach still shows <50% recall on boundary detection for newly generated text.

### 6.3 Draft Length vs. Acceptance Rate

The thesis identified a fundamental tradeoff: shorter LWG-bounded drafts have higher per-token acceptance rate but lower MAT. To break this tradeoff:

Option A: **Generate drafts of multiple LWGs.** Instead of stopping at the first LWG boundary, generate drafts that span 2–3 complete LWGs. This increases draft length while maintaining linguistic validity.

Option B: **Multi-LWG tree drafting.** At each LWG boundary, branch into multiple candidate next-LWGs (from different corpus positions). Present the verifier with a trie of complete LWG sequences rather than a single linear draft.

Option A is simpler and should be implemented first. Option B is the advanced multi-draft workflow already described in the thesis (Section 4.4.2) but needs extension to multi-LWG depth.

### 6.4 Which RL Framework to Use

- **GKD/SDFT** (Direction C2): Low implementation overhead, reuses existing fine-tuning infrastructure. Best for teaching the base model structural preferences. No reward model needed.
- **GRPO/PPO** with LWG reward: More flexible but computationally expensive. Requires a reward function that scores LWG compliance of the full generated sequence. Reserve for if GKD shows limited impact.

**Recommendation:** Start with GKD-style self-distillation. If the boundary prediction F1 from C1 is already high, C2 may not add much.

---

## 7. Expected Contributions

1. **Large-scale LWG-annotated Hindi corpus**: The first large-scale corpus annotated with Samanvaya word-group boundaries, enabling training of boundary-aware models.

2. **LWG-aware Dynamic SAM**: Extension of the WGA-SAM framework to handle on-the-fly generated text, closing the key limitation identified in the thesis.

3. **Boundary-aware EAGLE training**: EAGLE autoregression head trained with boundary prediction auxiliary loss, eliminating the need for post-hoc constraint application.

4. **On-policy LWG distillation**: First application of self-distillation with linguistically-informed rewards for Indian language generation, teaching models to internalize word-group structure.

5. **LWG-constrained tokenization**: Empirical study of how word-group-aligned BPE tokenization affects model perplexity and generation coherence for Hindi/Marathi.

6. **Cross-lingual evaluation**: Demonstration that word-group-aware speculative decoding generalizes beyond Hindi to Marathi.

---

## 8. Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| Large corpus annotation is slow | Medium | Parallelize `wordgrouping_rules.py`; scale to 8+ CPU cores |
| EAGLE training on Airavata/Param is compute-intensive | High | Use Param 2.9B as the primary model for training experiments; only fine-tune Airavata for final results |
| Boundary classifier F1 is low on novel text | Medium | Fall back to rule-based; expand rule set with more morphological patterns |
| RL/distillation training is unstable | Medium | Use GKD (simpler, more stable than PPO); anneal λ₁ gradually |
| Llama 4 token quality is poor for Hindi | Low | Already identified as a risk in the write-up; fall back to Airavata as primary verifier |
| LWG-constrained tokenizer hurts downstream performance | Medium | Run perplexity checks before full training; constrained merging may reduce vocabulary coverage |

---

## 9. Conference Submission Strategy

**Target:** October 2026 ACL ARR cycle (→ EMNLP 2026 or NAACL 2027)  
**Backup:** ECIR 2027

**Minimum viable paper (4 months):**
- Direction A (corpus scaling) + Directions B and C1 results
- Full ablation table with MAT, tokens/s, speedup, and boundary violation rate
- Comparison across Airavata, Param1, Qwen 3, Llama 3.1

**Full paper (6+ months):**
- All of Directions A–D
- Marathi cross-lingual results
- Human evaluation on MT-Bench-Hindi
- Direction E tokenization ablation (if complete)

**Framing:** "From Drafter to Model: Integrating Samanvaya Word Groups into Every Layer of Speculative Decoding for Indian Languages"

---

## 10. Related Papers in This Directory

The following reference papers are present and relevant:

- [adaSd.pdf](adaSd.pdf) — Adaptive speculative decoding; relevant to soft constraint integration in Direction D
- [adaspec.pdf](adaspec.pdf) — Another adaptive spec decoding approach; may inform the tradeoff analysis in Section 6.1
- [group tree optimization.pdf](group tree optimization.pdf) — Tree-based optimization; relevant to multi-LWG tree drafting in Section 6.3 Option B
- [guiding llms right way.pdf](guiding llms right way.pdf) — LLM guidance at inference time; relevant to Direction D constrained decoding
- [learning to draft.pdf](learning to draft.pdf) — Draft model training approaches; directly relevant to Direction C

---

## 11. Immediate Next Steps

**This week (before July 5, 2026):**
1. Set up Llama 4 Scout + EAGLE evaluation on MT-Bench-Hindi — verify token quality for Hindi (per write-up todo)
2. Run Param1-Instruct + EAGLE-2 on MT-Bench-Hindi to establish second baseline
3. Begin corpus download and LWG annotation pipeline scaling (Direction A)
4. Read the 5 reference papers in the directory and annotate key techniques relevant to each Direction

**Within 2 weeks:**
5. Complete large corpus LWG annotation; start WGA-Static SAM rebuild
6. Profile `wordgrouping_rules.py` throughput; identify bottlenecks
7. Design the online boundary detection integration for Dynamic SAM (Direction B)
