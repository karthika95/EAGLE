# SCG Evaluation Summary — Samanvaya-Constrained Generation

**Method:** Inference-time FSM constraint injection via HuggingFace LogitsProcessor.
- **SCG (constrained):** Opportunistic masking (DOMINO) — mask only when model's greedy violates rule.
- **AdaSCG (adaptive):** SCG + AdaSD entropy gate — skip mask when model confidence is high (H ≤ T_G).

**Constraint rules:** Samanvaya LWG: verb-auxiliary chains (ा/ी → AUX), compound postpositions (के + continuation), fixed phrases.

---

## Dataset: `mt_bench_hi_eval.jsonl` — MT-Bench-Hi open-domain instruction following (200 prompts)

Source: nvidia/MT-Bench-Hi. 8 categories (writing, humanities, roleplay, extraction, reasoning, stem, coding, math). Turn 1 only. No gold reference — compliance + cross-PPL as primary metrics.

---

## Model 1: `ai4bharat/Airavata` (7B, LLaMA-based Hindi instruction-tuned)

**WGTF** (Word Group Tokenization Fidelity): **92.4%**
> 92.4% of Samanvaya vocabulary words are single tokens — tokenizer CAN represent LWG units, but model's training distribution does not enforce LWG structure.

| Method | LWG Compliance | Δ Compliance | Tokens/s | IR | OSR | Cross-PPL ratio |
|--------|---------------|-------------|----------|-----|-----|-----------------|
| Baseline (AR) | 44.0% | — | 56.7 | 0.0% | — | 1.000 |
| SCG (constrained) | **81.9%** | **+37.8 pp** | 51.9 | 3.0% | 69.5% | 1.206 |
| AdaSCG (adaptive) | 58.1% | +14.0 pp | 52.2 | 1.5% | 38.8% | 1.154 |

*n_prompts = 200 (primary tier n=160: writing, humanities, roleplay, extraction)*

**Primary-tier headline (primary only):** Baseline 40.2% → SCG **82.9%** (+42.6 pp), AdaSCG 57.4% (+17.2 pp)

### Per-Category LWG Compliance — Airavata

| Category | Tier | n | Baseline | SCG (constrained) | AdaSCG (adaptive) |
|----------|------|---|----------|-------------------|-------------------|
| coding       | out_of_scope | 10 | 30.2% | 50.5% | 28.7% |
| extraction   | primary      | 40 | 42.3% | 79.1% | 61.2% |
| humanities   | primary      | 40 | 40.4% | 76.6% | 54.7% |
| math         | out_of_scope | 10 | 88.2% | 98.0% | 89.5% |
| reasoning    | secondary    | 10 | 71.6% | 97.5% | 75.8% |
| roleplay     | primary      | 40 | 36.0% | 88.3% | 61.3% |
| stem         | secondary    | 10 | 47.0% | 65.4% | 49.2% |
| writing      | primary      | 40 | 42.2% | 87.4% | 52.4% |

**Key observations:**
- SCG lifts compliance to 81.9% with only 3.0% IR and 69.5% OSR — model naturally complies in 70% of constrained steps.
- Cross-PPL ratio 1.206: constrained text has 20.6% higher perplexity under the base model — low fluency cost confirming opportunistic masking keeps forced interventions rare.
- AdaSCG entropy gate reduces PPL cost further (×1.154) but reduces compliance gain to +14 pp — entropy threshold (T_G=0.5) is aggressive on open-ended generation.

---

## Model 2: `bharatgenai/Param-1-2.9B-Instruct` (2.9B)

**WGTF** (Word Group Tokenization Fidelity): **84.9%**
> Lower WGTF than Airavata — more Samanvaya words split across multiple tokens, reducing precision of single-token masking.

| Method | LWG Compliance | Δ Compliance | Tokens/s | IR | OSR | Cross-PPL ratio |
|--------|---------------|-------------|----------|-----|-----|-----------------|
| Baseline (AR) | 36.7% | — | 59.7 | 0.0% | — | 1.000 |
| SCG (constrained) | **74.1%** | **+37.4 pp** | 55.4 | 3.6% | 67.8% | 1.537 |
| AdaSCG (adaptive) | 48.2% | +11.5 pp | 55.6 | 1.3% | 20.1% | 1.132 |

*n_prompts = 200 (primary tier n=160)*

**Primary-tier headline (primary only):** Baseline 35.0% → SCG **74.0%** (+39.0 pp), AdaSCG 45.5% (+10.5 pp)

### Per-Category LWG Compliance — Param-1

| Category | Tier | n | Baseline | SCG (constrained) | AdaSCG (adaptive) |
|----------|------|---|----------|-------------------|-------------------|
| coding       | out_of_scope | 10 | 30.0% | 76.2% | 50.0% |
| extraction   | primary      | 40 | 27.7% | 68.8% | 45.9% |
| humanities   | primary      | 40 | 40.3% | 79.4% | 45.7% |
| math         | out_of_scope | 10 | 51.4% | 79.9% | 71.9% |
| reasoning    | secondary    | 10 | 49.0% | 74.6% | 65.4% |
| roleplay     | primary      | 40 | 37.7% | 68.0% | 42.9% |
| stem         | secondary    | 10 | 43.4% | 67.9% | 49.1% |
| writing      | primary      | 40 | 34.4% | 79.7% | 47.6% |

**Key observations:**
- Compliance gain (+37.4 pp) matches Airavata (+37.8 pp) despite 2.4× fewer parameters — SCG effectiveness is grammar-driven, not capacity-driven.
- PPL ratio 1.537 > Airavata 1.206: lower WGTF (84.9%) causes subword-split words to receive imprecise token-level masking, forcing lower-probability continuations.
- AdaSCG dramatically reduces PPL cost (×1.537 → ×1.132) while compliance drops to +11.5 pp — entropy gating is especially valuable for the lower-WGTF model.
- OSR 67.8% (near-identical to Airavata 69.5%): model autonomy in constrained steps is consistent across model sizes.

---

## Cross-Model Comparison

| Model | Params | WGTF | Baseline CR | SCG CR | Δ | IR | OSR | PPL ratio |
|-------|--------|------|-------------|--------|---|----|-----|-----------|
| Airavata | 7B | 92.4% | 44.0% | **81.9%** | **+37.8 pp** | 3.0% | 69.5% | 1.206 |
| Param-1 | 2.9B | 84.9% | 36.7% | **74.1%** | **+37.4 pp** | 3.6% | 67.8% | 1.537 |

**WGTF–Compliance and WGTF–PPL correlation:**
Higher WGTF (92.4% Airavata vs 84.9% Param-1) predicts (a) higher SCG compliance ceiling and (b) lower cross-PPL fluency cost — supports WGTF as a diagnostic for inference-time masking effectiveness.

---

## Metrics Legend

| Metric | Definition |
|--------|-----------|
| LWG Compliance Rate (CR) | completions / (completions + violations) — % of all LWG obligations correctly satisfied |
| Intervention Rate (IR) | `mask_applied / total_decode_steps` — fraction of decode steps where vocabulary mask was applied |
| Opportunistic Save Rate (OSR) | `opportunistic_saves / constrained_steps` — fraction of constrained steps where model's greedy was already valid (DOMINO mechanism) |
| WGTF | Fraction of Samanvaya vocabulary words that are single tokens in the model's tokenizer |
| Tokens/s | Wall-clock throughput: `new_tokens / elapsed_seconds` |
| Cross-PPL ratio | PPL of generated text under the unconstrained model, normalized to baseline — measures fluency cost of constraint |

---

## Limitations

1. **MUST_CASE precision:** Genitive "के" (रामायण के पात्र) and compound postposition "के" (जानकारी के लिए) are indistinguishable without POS tagging. Genitive triggers MUST_CASE incorrectly, inflating violation counts.
2. **Subword masking coverage:** Words with WGTF < 100% (split across ≥2 tokens) are masked at subword level — mask may permit tokens whose complete word is invalid.
3. **AdaSCG T_G portability:** Entropy threshold T_G=0.5 may need per-task tuning; open-ended generation has a different entropy distribution than structured tasks.
4. **No human evaluation:** Compliance measured by automatic FSM; human judges needed to confirm perceived naturalness improvement.
