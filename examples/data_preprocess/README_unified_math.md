# Unified Math Dataset Processing

This directory contains scripts to create a unified math dataset following the Qwen-Math template as described in the DAPO paper.

## Overview

The unified dataset combines:
- **DAPO-Math-17K-processed**: ~17K math problems from DAPO corpus
- **MATH levels 3-5**: Math problems from Hendrycks MATH dataset (levels 3-5 only)

**Total**: ~25K math problems

## Qwen-Math Template

All problems are processed using the Qwen-Math template:

```
<|im_start|>system
Please reason step by step, and put your final answer within \boxed{}. <|im_end|>
<|im_start|>user
{problem}
<|im_end|>
<|im_start|>assistant
```

## Usage

### 1. Prepare Your Data

You need two input files:
- DAPO-Math-17K-processed (parquet or jsonl format)
- MATH-lighteval (parquet or jsonl format)

### 2. Run the Unification Script

```bash
python examples/data_preprocess/unified_math_data.py \
    --dapo_path /path/to/dapo-math-17k.parquet \
    --math_path /path/to/MATH-lighteval.parquet \
    --output_path /path/to/unified_math_25k.parquet \
    --level_filter 3 4 5
```

### 3. Use in Training

Update your training script to use the unified dataset:

```bash
python3 -m recipe.dapo.main_dapo \
    data.train_files="${HOME}/data/unified_math_25k.parquet" \
    data.val_files="${HOME}/data/aime-2024.parquet" \
    # ... other configs
```

## Data Format

The unified dataset has the following format:

```python
{
    "data_source": "math_dapo" or "DigitalLearningGmbH/MATH-lighteval",
    "prompt": [
        {
            "role": "system",
            "content": "Please reason step by step, and put your final answer within \\boxed{}."
        },
        {
            "role": "user",
            "content": "{problem statement}"
        },
        {
            "role": "assistant",
            "content": ""
        }
    ],
    "ability": "MATH",
    "reward_model": {
        "ground_truth": "{extracted_answer}",
        "style": "rule-Qwen-Math"
    },
    "extra_info": {
        # Dataset-specific metadata
    }
}
```

## Key Features

1. **Unified Template**: All problems use the same Qwen-Math template format
2. **Answer Extraction**: Ground truth answers are extracted from \boxed{} format
3. **Level Filtering**: Only MATH levels 3-5 are included
4. **Backward Compatible**: Reward functions handle both data sources

## Reward Function

The unified dataset works with both scoring methods:
- `math_dapo.compute_score()`: More lenient scoring (used first)
- `math.compute_score()`: Strict scoring (fallback)

See `verl/utils/reward_score/__init__.py` for details.

## Example Workflow

```bash
# 1. Download datasets
huggingface-cli download lighteval/MATH --local-dir ~/data/MATH-lighteval

# 2. Get DAPO-Math-17K-processed (from your source)

# 3. Create unified dataset
python examples/data_preprocess/unified_math_data.py \
    --dapo_path ~/data/dapo-math-17k.parquet \
    --math_path ~/data/MATH-lighteval/train.parquet \
    --output_path ~/data/unified_math_25k.parquet

# 4. Train DAPO
python3 -m recipe.dapo.main_dapo \
    data.train_files="${HOME}/data/unified_math_25k.parquet" \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    # ... other configs
```

## Citation

If you use this unified dataset processing, please cite:

```bibtex
@article{yu2025dapo,
  title={DAPO: Direct Alignment of Preference Optimization via Policy Gradient},
  author={Yu, Xiang and Yang, Aiyuan and Bai, Hongjiang and others},
  year={2025}
}
```

