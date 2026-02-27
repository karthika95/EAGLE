#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

# Configuration
devices=0
MODEL_PATH="ai4bharat/Airavata"
MODEL_ID="airavata-samd-wordgroup"
SAM_PATH="downloads/processed_file.pkl"
TREE_MODEL_PATH="downloads/airavata_bs1/state_20"
BENCH_NAME="mt_bench"  # or "spec_bench" 

# SAM parameters
N_PREDICTS=15
LEN_THRESHOLD=3
LEN_BIAS=2
DISABLE_DYN=false
DISABLE_EAGLE=false

# Generation parameters
MAX_NEW_TOKENS=1024
TEMPERATURE=0.0

echo "=================================================="
echo "SAM-D Evaluation with Word-Group-Aware SAM"
echo "=================================================="
echo "Model: $MODEL_PATH"
echo "SAM: $SAM_PATH"
echo "Benchmark: $BENCH_NAME"
echo "Model ID: $MODEL_ID"
echo "=================================================="
echo ""

CUDA_VISIBLE_DEVICES=${devices} \
    python -m evaluation.inference_samd_wordgroup \
    --model-type tulu \
    --template tulu \
    --bench-name ${BENCH_NAME} \
    --model-path ${MODEL_PATH} \
    --model-id ${MODEL_ID} \
    --sam_path ${SAM_PATH} \
    --tree_method eagle2 \
    --tree_model_path ${TREE_MODEL_PATH} \
    --samd_n_predicts ${N_PREDICTS} \
    --samd_len_threshold ${LEN_THRESHOLD} \
    --samd_len_bias ${LEN_BIAS} \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --temperature ${TEMPERATURE} \
    --dtype float16 \
    --attn_implementation sdpa \
    $([ "$DISABLE_DYN" = true ] && echo "--disable_dyn") \
    $([ "$DISABLE_EAGLE" = true ] && echo "--disable_eagle")

echo ""
echo "=================================================="
echo "Evaluation complete!"
echo "Results saved to: evaluation/data/${BENCH_NAME}/model_answer/${MODEL_ID}.jsonl"
echo "=================================================="
