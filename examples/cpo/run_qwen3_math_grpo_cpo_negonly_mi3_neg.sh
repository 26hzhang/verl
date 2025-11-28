#!/bin/bash
# run on 8xH100
# make sure your current working directory is the root of the project
swanlab login --api-key MzMWBJ6Bra1CfZ9cNqEUu
set -x

# 显示所有环境变量
echo "Environment Variables:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  RANK: $RANK"
echo "  NPROC_PER_NODE: $NPROC_PER_NODE"


READY_DIR="/mnt/workspace/workgroup_dev/weiwen/ray_ready"  # 共享目录，所有Worker节点会在这里写入"ready"文件

# **Master Node (RANK=0) Clears READY_DIR Before Starting**
if [ "$RANK" -eq 0 ]; then
    echo "Master node: Cleaning up READY_DIR ($READY_DIR)..."
    rm -rf $READY_DIR  # Remove old files
    mkdir -p $READY_DIR  # Recreate directory
fi

# **Ensure READY_DIR exists for workers**
mkdir -p $READY_DIR

# 判断当前节点是否为 Master
if [ "$RANK" -eq 0 ]; then
    echo "This is the Master node. Starting Ray head node..."
    ray start --head --port=$MASTER_PORT
else
    echo "This is a Worker node. Waiting for the Master node to be ready..."
    
    # **Worker Nodes Wait for Master**
    MAX_RETRIES=60
    WAIT_TIME=10
    for ((i=1; i<=MAX_RETRIES; i++)); do
        echo "Checking if Master node is available at $MASTER_ADDR:$MASTER_PORT (Attempt $i/$MAX_RETRIES)..."
        (echo > /dev/tcp/$MASTER_ADDR/$MASTER_PORT) >/dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "Master node is available! Connecting as a Worker node..."
            break
        else
            echo "Master node is not available yet. Sleeping for $WAIT_TIME seconds..."
            sleep $WAIT_TIME
        fi
    done

    # **Start Ray Worker (Only if not already running)**
    ray start --address="$MASTER_ADDR:$MASTER_PORT"

    # **Worker Marks Itself as Ready**
    touch "$READY_DIR/worker_$RANK.ready"
    echo "Worker $RANK is ready."

    # **Keep Worker Running Until Master Completes Training**
    echo "Worker node $RANK: Keeping Ray process alive..."
    # Worker节点不需要提交任务，只需要等待
    while true; do
        sleep 30
        nvidia-smi
    done
fi


ulimit -n 65535
PROJECT_DIR="$(pwd)"
# loss_agg_mode="token-mean"
use_kl_in_reward=False
kl_coef=0.001
use_kl_loss=True
kl_loss_coef=0.001

clip_ratio_low=0.2
clip_ratio_high=0.2

max_prompt_length=2048
max_response_length=16384

train_prompt_bsz=1024
ppo_mini_batch_size=256


do_think=False

enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=10

enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 3))
overlong_penalty_factor=0.1



math_train_path=/mnt/workspace/workgroup_dev/weiwen/cut2_dynamic/data/math_train/train_math.parquet
math_test_path=/mnt/workspace/workgroup_dev/weiwen/cut2_dynamic/data/math_train/test_math.parquet
aime2025_test_path=/mnt/workspace/workgroup_dev/weiwen/cut2_dynamic/data/math_train/test_aime2025.parquet
aime2024_test_path=/mnt/workspace/workgroup_dev/weiwen/cut2_dynamic/data/math_train/test_aime2024.parquet
gpqa_test_path=/mnt/workspace/workgroup_dev/weiwen/cut2_dynamic/data/math_train/test_gpqa.parquet
amc23_test_path=/mnt/workspace/workgroup_dev/weiwen/cut2_dynamic/data/math_train/test_amc23.parquet
math500_test_path=/mnt/workspace/workgroup_dev/weiwen/cut2_dynamic/data/math_train/test_math500.parquet
project_name='math_rl_qwen3-1.7b'
exp_name='Qwen3-1.7B_math-vllm-n8_1103_cpo_neg0.1_negonly_mi3_'
CKPTS_DIR=/mnt/workspace/workgroup_dev/weiwen/verl_checkpoints/${project_name}/${exp_name}
train_files="['$math_train_path']"
test_files="['$math500_test_path', '$aime2024_test_path', '$aime2025_test_path', '$amc23_test_path', '$gpqa_test_path']"


# **Master Waits for All Workers**_
if [ "$RANK" -eq 0 ]; then
    echo "Waiting for all Worker nodes to be ready..."

    EXPECTED_WORKERS=$((WORLD_SIZE - 1))  # 计算Worker节点数量
    while true; do
        READY_COUNT=$(ls -1 $READY_DIR | wc -l)
        echo "Current ready workers: $READY_COUNT / $EXPECTED_WORKERS"
        if [ "$READY_COUNT" -ge "$EXPECTED_WORKERS" ]; then
            echo "All workers are ready. Proceeding with training..."
            break
        fi
        sleep 5
    done
    TRAINING_NODES=$WORLD_SIZE  # Equivalent to WORLD_SIZE
    TRAINING_GPUS_PER_NODE=$NPROC_PER_NODE  # Equivalent to NPROC_PER_NODE
        echo "Training Configuration:"
    echo "  TRAINING_NODES: $TRAINING_NODES"
    echo "  TRAINING_GPUS_PER_NODE: $TRAINING_GPUS_PER_NODE"
    
    echo "Starting training on the Master node..."
    
    python -m verl.trainer.main_ppo \
        algorithm.adv_estimator=cpo \
        trainer.project_name=${project_name} \
        trainer.experiment_name=${exp_name} \
        actor_rollout_ref.model.path=../models/Qwen3-1.7B \
        +algorithm.gold_as_hint=True \
        +algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
        +algorithm.cpo_lambda=2 \
        +algorithm.pos_alpha=0 \
        +algorithm.neg_alpha=0.1 \
        +actor_rollout_ref.actor.cpo_lambda=2 \
        +actor_rollout_ref.actor.pos_alpha=0 \
        +actor_rollout_ref.actor.neg_alpha=0.1 \
        trainer.critic_warmup=0 \
        trainer.logger='["swanlab"]' \
        trainer.n_gpus_per_node=$TRAINING_GPUS_PER_NODE \
        trainer.nnodes=$TRAINING_NODES \
        trainer.save_freq=7 \
        trainer.test_freq=7 \
        trainer.log_val_generations=7 \
        data.train_batch_size=${train_prompt_bsz} \
        data.max_prompt_length=${max_prompt_length} \
        data.max_response_length=${max_response_length} \
        data.filter_overlong_prompts=True \
        +data.thinking=${do_think} \
        data.truncation='error' \
        data.return_raw_chat=True \
        +actor_rollout_ref.actor.loss_position='all' \
        +actor_rollout_ref.actor.wrap_method="negonly_mi3" \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
        actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
        algorithm.kl_ctrl.kl_coef=${kl_coef} \
        actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20000 \
        actor_rollout_ref.rollout.max_num_batched_tokens=20000\
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.rollout.multi_turn.enable=False\
        actor_rollout_ref.rollout.multi_turn.max_assistant_turns=1\
        algorithm.use_kl_in_reward=${use_kl_in_reward} \
        data.train_files="$train_files" \
        data.val_files="$test_files" \
        trainer.default_local_dir="${CKPTS_DIR}" \
        trainer.total_epochs=20 $@

fi
    # actor_rollout_ref.rollout.max_num_batched_tokens=30000\ vllm专用
# nohup bash run_qwen2.5_math_grpo_cut.sh.gsh  > train-0901.log 2>&1 &
