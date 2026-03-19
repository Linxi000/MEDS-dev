#!/bin/bash
# Example script to unify DAPO-Math-17K and MATH levels 3-5 using Qwen-Math template

set -e

# Paths (modify these according to your environment)
DAPO_MATH_PATH="${HOME}/data/dapo-math-17k.parquet"
MATH_LIGHTEVAL_PATH="${HOME}/data/MATH-lighteval.parquet"
OUTPUT_PATH="${HOME}/data/unified_math_25k.parquet"

# Run the unification script
python examples/data_preprocess/unified_math_data.py \
    --dapo_path "${DAPO_MATH_PATH}" \
    --math_path "${MATH_LIGHTEVAL_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --level_filter 3 4 5

echo "Unified dataset created at: ${OUTPUT_PATH}"
echo "You can now use this dataset for training DAPO with Qwen-Math template"

