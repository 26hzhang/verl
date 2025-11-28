#!/bin/bash
# run on 8xH100
# make sure your current working directory is the root of the project
swanlab login --api-key iPPnOFirR3dXCBgOBPqiB
set -x

# 显示所有环境变量
echo "Environment Variables:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  RANK: $RANK"
echo "  NPROC_PER_NODE: $NPROC_PER_NODE"

PROJECT_DIR="$(pwd)"

READY_DIR=$PROJECT_DIR/all_outputs/ray_ready  # 共享目录，所有Worker节点会在这里写入"ready"文件

# **Master Node (RANK=0) Clears READY_DIR Before Starting**
if [ "$RANK" -eq 0 ]; then
    echo "Master node: Cleaning up READY_DIR ($READY_DIR)..."
    rm -rf $READY_DIR  # Remove old files
    mkdir -p $READY_DIR  # Recreate directory
fi

# **Ensure READY_DIR exists for workers**
mkdir -p $READY_DIR


ulimit -n 65535
# loss_agg_mode="token-mean"
use_kl_in_reward=False
kl_coef=0.001
use_kl_loss=True
kl_loss_coef=0.001

clip_ratio_low=0.2
clip_ratio_high=0.2

max_prompt_length=2048
max_response_length=8192

train_prompt_bsz=1024
ppo_mini_batch_size=256

do_think=False

enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=10

enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 3))
overlong_penalty_factor=0.1

# "token-mean", "seq-mean-token-sum", or "seq-mean-token-mean"
loss_agg_mode="seq-mean-token-mean"

math_train_path=$PROJECT_DIR/data/math_train/train_math.parquet
math_test_path=$PROJECT_DIR/data/math_train/test_math.parquet
aime2025_test_path=$PROJECT_DIR/data/math_train/test_aime2025.parquet
aime2024_test_path=$PROJECT_DIR/data/math_train/test_aime2024.parquet
gpqa_test_path=$PROJECT_DIR/data/math_train/test_gpqa.parquet
amc23_test_path=$PROJECT_DIR/data/math_train/test_amc23.parquet
math500_test_path=$PROJECT_DIR/data/math_train/test_math500.parquet
project_name='math_cpo_rl_qwen3-1.7b'
exp_name='Qwen3-1.7B_math-vllm-n8_1103_cpo_neg0.1_negonly_mi3_8192'
CKPTS_DIR=$PROJECT_DIR/all_outputs/verl_ckpts/${project_name}/${exp_name}
train_files="['$math_train_path']"
test_files="['$math500_test_path', '$aime2024_test_path', '$aime2025_test_path', '$amc23_test_path', '$gpqa_test_path']"
    
echo "Starting training on the Master node..."
    
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=cpo \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    trainer.critic_warmup=0 \
    trainer.logger='["swanlab","console"]' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=7 \
    trainer.test_freq=7 \
    trainer.log_val_generations=7 \
    actor_rollout_ref.model.path=$PROJECT_DIR/model_weights/Qwen3-1.7B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    +algorithm.gold_as_hint=True \
    +algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    +algorithm.cpo_lambda=2 \
    +algorithm.pos_alpha=0 \
    +algorithm.neg_alpha=0.1 \
    +actor_rollout_ref.actor.cpo_lambda=2 \
    +actor_rollout_ref.actor.pos_alpha=0 \
    +actor_rollout_ref.actor.neg_alpha=0.1 \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.thinking=${do_think} \
    +actor_rollout_ref.actor.loss_position='all' \
    +actor_rollout_ref.actor.wrap_method="negonly_mi3" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20000 \
    actor_rollout_ref.rollout.max_num_batched_tokens=20000\
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
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
