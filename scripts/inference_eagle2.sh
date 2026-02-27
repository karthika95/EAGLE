#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

devices=0

# vicuna-7b-v1.3
CUDA_VISIBLE_DEVICES=${devices} python -m evaluation.inference_eagle2 \
    --model-type tulu \
    --ea-model-path /data/pranav_shinde/pranav/SAM-Decoding/downloads/airavata_bs1/state_20 \
    --base-model-path ai4bharat/Airavata \
    --model-id airavata-eagle2 \
    --bench-name mt_bench \
    --temperature 0
