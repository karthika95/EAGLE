#!/bin/bash
# Comprehensive Draft Token Statistics Evaluation
# Evaluates and compares original SAMD vs word-group-aware SAMD

set -e

cd $(dirname $0)/..

# Configuration
DEVICES=0
MODEL_PATH="ai4bharat/Airavata"
MODEL_TYPE="tulu"
BENCH_NAME="mt_bench"

# SAM paths
# downloads/sam_hindi_wordgroup.pkl
# preprocess/sam_airavata_wikichat.pkl
SAM_PATH="preprocess/sam_airavata_wikichat.pkl"
WORDGROUP_SAM_PATH="preprocess/sam_airavata_wikichat.pkl"
TREE_MODEL_PATH="/data/pranav_shinde/pranav/EAGLE/checkpoints/airavata_wikichat/state_20/"

# SAM parameters
N_PREDICTS=15
LEN_THRESHOLD=0
LEN_BIAS=0

# Generation parameters
MAX_NEW_TOKENS=1024
QUESTION_COUNT=200  # Set to full dataset size (e.g., 80) for complete evaluation

# Output directory
OUTPUT_DIR="evaluation/statistics_results"

echo "=================================================="
echo "Draft Token Statistics Evaluation"
echo "=================================================="
echo "Model: $MODEL_PATH"
echo "Benchmark: $BENCH_NAME"
echo "Questions: $QUESTION_COUNT"
echo "SAM: $SAM_PATH"
echo "WordGroup SAM: $WORDGROUP_SAM_PATH"
echo "Output: $OUTPUT_DIR"
echo "=================================================="
echo ""

# Run evaluation
CUDA_VISIBLE_DEVICES=${DEVICES} \
    python -m evaluation.evaluate_draft_statistics \
    --model-path ${MODEL_PATH} \
    --model-type ${MODEL_TYPE} \
    --bench-name ${BENCH_NAME} \
    --sam-path ${SAM_PATH} \
    --wordgroup-sam-path ${WORDGROUP_SAM_PATH} \
    --tree-model-path ${TREE_MODEL_PATH} \
    --samd-n-predicts ${N_PREDICTS} \
    --samd-len-threshold ${LEN_THRESHOLD} \
    --samd-len-bias ${LEN_BIAS} \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --question-count ${QUESTION_COUNT} \
    --output-dir ${OUTPUT_DIR} \
    --dtype float16

echo ""
echo "=================================================="
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=================================================="
