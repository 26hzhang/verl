# run on 8xH100
# make sure your current working directory is the root of the project
export SWANLAB_PROJECT="mllm_train_sft"
export OPENAI_ENDPOINT="https://idealab.alibaba-inc.com/api/openai/v1"
export OPENAI_API_KEY="afdfd0b8bacc7e7c86b5e59fa6fc027b" # lasa: afdfd0b8bacc7e7c86b5e59fa6fc027b. https://idealab.alibaba-inc.com/api/openai/v1 decision: 956c41bd0f31beaf68b871d4987af4bb http://39.96.211.155:8000/proxy/api/openai/v1/
export DEPLOYMENT_MODEL="qwen3-235b-a22b-instruct-2507"
swanlab login --api-key MzMWBJ6Bra1CfZ9cNqEUu
set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
loss_agg_mode="token-mean"
use_kl_in_reward=False
kl_coef=0.001
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 ))
max_response_length=$((1024 * 3))
train_prompt_bsz=1024
ppo_mini_batch_size=256
do_think=False

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10

enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 3))
overlong_penalty_factor=0.1

math_train_path=data/math_nsr/train_math.parquet
math_test_path=data/math_nsr/test_math.parquet
aime2025_test_path=data/math_nsr/test_aime2025.parquet
amc23_test_path=data/math_nsr/test_amc23.parquet
train_files="['$math_train_path']"
test_files="['$math_test_path', '$aime2025_test_path', '$amc23_test_path']"
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.project_name='math_rl' \
    trainer.experiment_name='Qwen2.5-math-7B_math-vllm-n8_0919_dapo' \
    actor_rollout_ref.model.path=../models/Qwen2.5-Math-7B \
    trainer.critic_warmup=0 \
    trainer.logger='["swanlab"]' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=8 \
    trainer.test_freq=1 \
    trainer.log_val_generations=2 \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    +data.thinking=${do_think} \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=30000 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
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
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    +algorithm.filter_groups.enable=${enable_filter_groups} \
    +algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    +algorithm.filter_groups.metric=${filter_groups_metric} \
    trainer.total_epochs=20 $@
