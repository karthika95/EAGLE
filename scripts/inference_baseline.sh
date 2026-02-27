#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

devices=0

# vicuna
CUDA_VISIBLE_DEVICES=${devices} \
    python -m evaluation.inference_baseline \
    --model-type tulu \
    --bench-name mt_bench \
    --model-path ai4bharat/Airavata \
    --model-id airavata-7b
