#!/bin/bash

export TOKENIZERS_PARALLELISM=False

# 基础路径
BASE_PATH=$1

# 数据集路径
math_test_path=../data/math_train/test_math.parquet
aime2025_test_path=../data/math_train/test_aime2025.parquet
aime2024_test_path=../data/math_train/test_aime2024.parquet
amc23_test_path=../data/math_train/test_amc23.parquet
gpqa_test_path=../data/math_train/test_gpqa.parquet
math500_test_path=../data/math_train/test_math500.parquet

datasets="${aime2025_test_path},${amc23_test_path},${gpqa_test_path},${aime2024_test_path},${math500_test_path},${math_test_path}"
datasets="${aime2025_test_path},${amc23_test_path},${gpqa_test_path},${aime2024_test_path},${math500_test_path}"
# datasets="${amc23_test_path}"
# datasets="${math_test_path}"

# 循环评测不同的 global_step
for step in $(seq $2 7 $3); do
    echo "=========================================="
    echo "Processing global_step_${step}"
    echo "=========================================="
    
    VERL_MODEL_PATH="${BASE_PATH}/global_step_${step}/actor"
    MODEL_PATH="${VERL_MODEL_PATH}/huggingface"
    # MODEL_PATH="$BASE_PATH"
    # 检查路径是否存在
    if [ ! -d "$MODEL_PATH" ]; then
        echo "Warning: $MODEL_PATH does not exist, skipping..."
        continue
    fi
    
    # echo "Evaluating $MODEL_PATH"
    
    # 如果需要先执行 merge，取消注释下面的代码
     python merge.py --local_dir $VERL_MODEL_PATH
    
    # 执行评测（如果需要重新生成结果，取消注释）
    python eval.py \
      --model_name="$MODEL_PATH" \
      --datasets="$datasets" \
      --split="test" \
      --output_dir="$MODEL_PATH" \
      --batch_size=1000 \
      --max_tokens=16384 \
      --num_gpus=2 \
      --temperature=0.6 \
      --top_p=0.95 \
      --num_generation=16
    
    # 计算各个数据集的指标
    echo "Calculating metrics for global_step_${step}..."
    python cal_metric.py --file_path "${MODEL_PATH}/math500-test-temp_0.6-top_p_0.95-top_k_-1.jsonl"
    # python cal_metric.py --file_path "${MODEL_PATH}/math-test-temp_0.6-top_p_0.95-top_k_-1.jsonl"
    python cal_metric.py --file_path "${MODEL_PATH}/aime2024-test-temp_0.6-top_p_0.95-top_k_-1.jsonl"
    python cal_metric.py --file_path "${MODEL_PATH}/aime2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl"
    python cal_metric.py --file_path "${MODEL_PATH}/amc23-test-temp_0.6-top_p_0.95-top_k_-1.jsonl"
    python cal_metric.py --file_path "${MODEL_PATH}/gpqa-test-temp_0.6-top_p_0.95-top_k_-1.jsonl"
    
    echo "Finished processing global_step_${step}"
    echo ""
done

echo "All evaluations completed!"