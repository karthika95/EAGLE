#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

python -m evaluation.speed \
    --file-path evaluation/data/mt_bench/model_answer/airavata-samd-wordgroup.jsonl

    
# evaluation/data/mt_bench/model_answer/airavata-eagle2-temperature-0.0.jsonl
# evaluation/data/mt_bench/model_answer/airavata-samd-wordgroup.jsonl

# python -m evaluation.speed \
#     --file-path evaluation/data/spec_bench/model_answer/vicuna-7b-v1.3-samd-token_recycle.jsonl

# python -m evaluation.speed \
#     --file-path evaluation/data/spec_bench/model_answer/vicuna-7b-v1.3-samd-eagle2.jsonl
