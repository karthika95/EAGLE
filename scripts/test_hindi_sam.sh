#!/bin/bash

# Test word-group-aware SAM inference for Hindi
# This script runs inference that respects word group boundaries

# Default paths - modify as needed
MODEL_PATH="ai4bharat/Airavata"
SAM_PATH="/data/pranav_shinde/pranav/SAM-Decoding/downloads/processed_file.pkl"
N_PREDICTS=15
MAX_NEW_TOKENS=256
DRAFT_PATH="/data/pranav_shinde/pranav/SAM-Decoding/downloads/airavata_bs1/state_20"
LEN_THRESHOLD=5
LEN_BIAS=5
DISABLE_DYN="False"
DISABLE_EAGLE="False"
INTERACTIVE=false  # Set to true for interactive mode


# Default Hindi prompt (only used in non-interactive mode)
PROMPT="हिंदुस्तानी शास्त्रीय संगीत"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --interactive|-i)
            INTERACTIVE=true
            shift
            ;;
        --prompt|-p)
            PROMPT="$2"
            shift 2
            ;;
        --len-threshold)
            LEN_THRESHOLD="$2"
            shift 2
            ;;
        --len-bias)
            LEN_BIAS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -i, --interactive       Run in interactive mode (load SAM once, run multiple inferences)"
            echo "  -p, --prompt TEXT       Set the prompt (non-interactive mode only)"
            echo "  --len-threshold N       Set len_threshold (default: 5)"
            echo "  --len-bias N            Set len_bias (default: 5)"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Interactive mode allows you to:"
            echo "  • Load the large SAM file only once (saves ~18 minutes per run)"
            echo "  • Run multiple inferences with different parameters"
            echo "  • Change len_threshold, len_bias, and other settings on the fly"
            echo ""
            echo "Performance tip:"
            echo "  Convert .pkl to .pt format for 4-6x faster loading:"
            echo "  bash scripts/convert_sam_to_torch.sh"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

if [ "$INTERACTIVE" = true ]; then
    echo "===================================================================="
    echo "Starting INTERACTIVE MODE"
    echo "===================================================================="
    echo "Model: $MODEL_PATH"
    echo "SAM: $SAM_PATH"
    echo ""
    echo "Loading SAM (this will take ~18 minutes, but only happens once)..."
    echo "After loading, you can run unlimited inferences with different parameters!"
    echo "===================================================================="
else
    echo "Testing word-group-aware SAM inference..."
    echo "Model: $MODEL_PATH"
    echo "SAM: $SAM_PATH"
    echo "Prompt: $PROMPT"
fi
echo ""

# Build command with conditional flags
CMD="CUDA_VISIBLE_DEVICES=0 python tests/test_samd_hindi_wordgroup.py \
    --model_path \"$MODEL_PATH\" \
    --sam_path \"$SAM_PATH\" \
    --samd_n_predicts \"$N_PREDICTS\" \
    --max_new_tokens \"$MAX_NEW_TOKENS\" \
    --tree_method \"eagle2\" \
    --tree_model_path \"$DRAFT_PATH\" \
    --dtype \"float16\" \
    --device \"cuda\" \
    --len_threshold \"$LEN_THRESHOLD\" \
    --len_bias \"$LEN_BIAS\" \
    --disable_dyn \"$DISABLE_DYN\" \
    --disable_eagle \"$DISABLE_EAGLE\""

# Add interactive flag if enabled
if [ "$INTERACTIVE" = true ]; then
    CMD="$CMD --interactive"
else
    # Add prompt and skip_baseline only in non-interactive mode
    CMD="$CMD --prompt \"$PROMPT\" --skip_baseline"
fi

# Execute the command
eval $CMD

echo ""
if [ "$INTERACTIVE" = true ]; then
    echo "Interactive session ended."
else
    echo "Inference complete!"
fi
