#!/bin/bash

export TOKENIZERS_PARALLELISM=False
VERL_MODEL_PATH=/mindopt/nlp_nas_2/weiwen/Dynamic_Verl/checkpoints/math_rl/MATH-Qwen2.5-Math-7B-GRPO-longli-dynamicrollout/global_step_112/actor
MODEL_PATH=/mindopt/nlp_nas_2/weiwen/Dynamic_Verl/checkpoints/math_rl/MATH-Qwen2.5-Math-7B-GRPO-longli-dynamicrollout/global_step_112/actor/huggingface
OUTPUT_DIR=/mindopt/nlp_nas_2/weiwen/Dynamic_Verl/checkpoints/math_rl/MATH-Qwen2.5-Math-7B-GRPO-longli-dynamicrollout/global_step_112/actor/huggingface

# mkdir -p "$OUTPUT_DIR"
# echo "converting $VERL_MODEL_PATH"

# python merge.py \
#     --local_dir ${VERL_MODEL_PATH}


echo "Evaluating $MODEL_PATH"

math_test_path=data/math_test/MATH
aime2025_test_path=data/math_test/AIME2025
amc23_test_path=data/math_test/amc23

datasets="${math_test_path},${aime2025_test_path},${amc23_test_path}"

python eval.py \
  --model_name="$MODEL_PATH" \
  --datasets="$datasets" \
  --split="test" \
  --output_dir="$OUTPUT_DIR" \
  --batch_size=1000 \
  --max_tokens=4096 \
  --num_gpus=2 \
  --temperature=0.6 \
  --top_p=0.95 \
  --num_generation=16