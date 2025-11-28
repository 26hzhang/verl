#!/bin/bash
# run on 8xH100
# make sure your current working directory is the root of the project
export VLLM_USE_V1=1

swanlab login --api-key iPPnOFirR3dXCBgOBPqiB

set -x
ulimit -n 65535

ray stop
ray start --head --resources='{"drivers": 1}'

loss_agg_mode="token-mean"
use_kl_in_reward=False
kl_coef=0.001
use_kl_loss=True
kl_loss_coef=0.001

max_prompt_length=2048
max_response_length=16384

train_prompt_bsz=1024
ppo_mini_batch_size=256

do_think=True

cpo_lambda=2.0
pos_alpha=0.0
neg_alpha=1.0
loss_position="all"
wrap_method="seq_kl"

# Hyperparams
DTYPE=bfloat16
MODEL_DTYPE=fp32

PROJECT_DIR="$(pwd)"
math_train_path=$PROJECT_DIR/data/math_train/train_math.parquet
math_test_path=$PROJECT_DIR/data/math_train/test_math.parquet
aime2025_test_path=$PROJECT_DIR/data/math_train/test_aime2025.parquet
aime2024_test_path=$PROJECT_DIR/data/math_train/test_aime2024.parquet
gpqa_test_path=$PROJECT_DIR/data/math_train/test_gpqa.parquet
amc23_test_path=$PROJECT_DIR/data/math_train/test_amc23.parquet
math500_test_path=$PROJECT_DIR/data/math_train/test_math500.parquet
project_name='cpo_qwen3-4b-base'
exp_name='Qwen3-4B-base_math-vllm-n8_1125_cpo_neg1_seqkl_clip_studio'
CKPTS_DIR=$PROJECT_DIR/checkpoints/${project_name}/${exp_name}
train_files="['$math_train_path']"
test_files="['$math500_test_path', '$aime2024_test_path', '$aime2025_test_path', '$amc23_test_path', '$gpqa_test_path']"


PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    +data.thinking=${do_think} \
    data.truncation='error' \
    data.return_raw_chat=True \
    algorithm.adv_estimator=cpo \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    +algorithm.gold_as_hint=True \
    +algorithm.cpo_lambda=${cpo_lambda} \
    +algorithm.pos_alpha=${pos_alpha} \
    +algorithm.neg_alpha=${neg_alpha} \
    actor_rollout_ref.model.path=$PROJECT_DIR/model_weights/Qwen3-4B-Base \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20000 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.dtype=$DTYPE \
    actor_rollout_ref.actor.fsdp_config.model_dtype=$MODEL_DTYPE \
    +actor_rollout_ref.actor.cpo_lambda=${cpo_lambda} \
    +actor_rollout_ref.actor.pos_alpha=${pos_alpha} \
    +actor_rollout_ref.actor.neg_alpha=${neg_alpha} \
    +actor_rollout_ref.actor.loss_position=${loss_position} \
    +actor_rollout_ref.actor.wrap_method=${wrap_method} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=20000 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.dtype=$DTYPE \
    actor_rollout_ref.ref.fsdp_config.model_dtype=$MODEL_DTYPE \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.total_epochs=20 \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab"]' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=7 \
    trainer.test_freq=7 \
    trainer.log_val_generations=7 \
    $@