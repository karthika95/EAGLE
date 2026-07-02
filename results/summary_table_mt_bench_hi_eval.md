# SCG Evaluation Summary — MT-Bench-Hi (200 prompts)

**Dataset:** `mt_bench_hi_eval.jsonl` — MT-Bench-Hi (nvidia/MT-Bench-Hi), Hindi instruction following across 8 categories.
**Task:** Open-ended Hindi generation, 2-turn benchmark (turn 1 used). No fixed reference; compliance + fluency metrics.

---

## Model 1: `ai4bharat/Airavata` (7B, LLaMA-based)

**WGTF** (Word Group Tokenization Fidelity): **92.4%**

| Method | LWG Compliance | Δ vs Base | Tokens/s | IR | OSR | PPL (×base) |
|--------|---------------|-----------|----------|-----|-----|-------------|
| baseline     | 44.0% | — | 56.7 | 0.0% | 0.0% | 1.000 |
| constrained  | 81.9% | +37.8 pp | 51.9 | 3.0% | 69.5% | 1.206 |
| adaptive     | 58.1% | +14.0 pp | 52.2 | 1.5% | 38.8% | 1.154 |

*n_prompts = 200*

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

### Primary-tier Headline — Airavata (writing + humanities + roleplay + extraction, n=160)

| Method | LWG Compliance | Δ vs Base | IR | OSR |
|--------|---------------|-----------|-----|-----|
| baseline     | 40.2% | — | 0.0% | 0.0% |
| constrained  | 82.9% | +42.6 pp | 3.1% | 69.5% |
| adaptive     | 57.4% | +17.2 pp | 1.5% | 32.6% |

---

## Model 2: `bharatgenai/Param-1-2.9B-Instruct` (2.9B)

**WGTF** (Word Group Tokenization Fidelity): **84.9%**

| Method | LWG Compliance | Δ vs Base | Tokens/s | IR | OSR | PPL (×base) |
|--------|---------------|-----------|----------|-----|-----|-------------|
| baseline     | 36.7% | — | 59.7 | 0.0% | 0.0% | 1.000 |
| constrained  | 74.1% | +37.4 pp | 55.4 | 3.6% | 67.8% | 1.537 |
| adaptive     | 48.2% | +11.5 pp | 55.6 | 1.3% | 20.1% | 1.132 |

*n_prompts = 200*

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

### Primary-tier Headline — Param-1 (writing + humanities + roleplay + extraction, n=160)

| Method | LWG Compliance | Δ vs Base | IR | OSR |
|--------|---------------|-----------|-----|-----|
| baseline     | 35.0% | — | 0.0% | 0.0% |
| constrained  | 74.0% | +39.0 pp | 3.8% | 69.2% |
| adaptive     | 45.5% | +10.5 pp | 1.4% | 20.2% |

*n_primary = 160*

---

## Cross-Model Comparison — MT-Bench-Hi

| Model | Params | WGTF | Baseline CR | SCG CR | Δ | IR | OSR | PPL ratio |
|-------|--------|------|-------------|--------|---|----|-----|-----------|
| Airavata | 7B | 92.4% | 44.0% | **81.9%** | **+37.8 pp** | 3.0% | 69.5% | 1.206 |
| Param-1 | 2.9B | 84.9% | 36.7% | **74.1%** | **+37.4 pp** | 3.6% | 67.8% | 1.537 |

**Key observations:**
- Both models achieve near-identical compliance gains (+37.8 pp vs +37.4 pp), despite 7B vs 2.9B parameter difference — suggesting SCG effectiveness is grammar-rule-driven, not model-capacity-driven.
- Airavata's higher WGTF (92.4%) keeps PPL cost low (×1.206); Param-1's lower WGTF (84.9%) incurs higher fluency cost (×1.537) — subword-split words receive imprecise token-level masking, forcing lower-probability continuations.
- AdaSCG substantially reduces PPL cost for Param-1 (×1.537 → ×1.132) at the cost of compliance gain (+37.4 → +11.5 pp); for Airavata the trade-off is milder (×1.206 → ×1.154, compliance +37.8 → +14.0 pp).
- OSR is ~67-70% for both models: in ~2 of every 3 constrained steps, the model independently selects a valid token — the constraint acts as a corrective safety net, not a dominant controller.
- Coding/math categories: values are not meaningful for LWG analysis (coding generates English/code, math generates numerals/symbols — FSM sees no Hindi verb/postposition constructs). Reported for completeness only.
