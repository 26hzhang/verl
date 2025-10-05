# Running verl Entrypoints

## Training Examples
verl provides various training examples organized by algorithm type in the `examples/` directory:

### PPO Training
```bash
# Basic PPO training example
cd examples/ppo_trainer/
# Check specific run scripts like run_qwen2-7b.sh

# PPO with sequence balancing
examples/ppo_trainer/run_qwen2-7b_seq_balance.sh

# PPO with sequence parallelism
examples/ppo_trainer/run_deepseek7b_llm_sp2.sh
```

### GRPO Training
```bash
# GRPO training examples
cd examples/grpo_trainer/
# Check run scripts like run_qwen3-8b.sh

# Multi-modal GRPO with vision-language models
examples/grpo_trainer/run_qwen2_5_vl-7b.sh
```

### Supervised Fine-tuning
```bash
# SFT examples
cd examples/sft/gsm8k/
# Various SFT configurations available:
examples/sft/gsm8k/run_qwen_05_peft.sh          # LoRA fine-tuning
examples/sft/gsm8k/run_qwen_05_sp2_liger.sh     # With Liger kernel
```

### Other Algorithms
```bash
# RLOO training
cd examples/rloo_trainer/

# ReMax training  
cd examples/remax_trainer/

# REINFORCE++ training
cd examples/reinforce_plus_plus_trainer/
```

## Multi-turn and Tool Integration
```bash
# Multi-turn conversations with SGLang
cd examples/sglang_multiturn/

# Tool integration examples available
```

## Configuration Management
```bash
# Print current configuration
python3 scripts/print_cfg.py --cfg job

# Print specific trainer config
python3 scripts/print_cfg.py --cfg job --config-name=ppo_megatron_trainer.yaml

# Generate config files
scripts/generate_trainer_config.sh
```

## Distributed and Cloud Deployment
```bash
# Ray-based distributed training
cd examples/ray/

# SLURM cluster deployment
cd examples/slurm/

# SkyPilot cloud deployment
cd examples/skypilot/

# Split GPU placement
cd examples/split_placement/
```

## Recipe-based Training
Algorithm-specific recipes are available in the `recipe/` directory:
- `recipe/dapo/` - DAPO algorithm
- `recipe/sppo/` - Self-play preference optimization
- `recipe/gspo/` - GSPO algorithm
- `recipe/prime/` - PRIME algorithm

## Diagnostic and Utility Scripts
```bash
# Diagnose system/environment
python3 scripts/diagnose.py

# Initialize random model
python3 scripts/init_random_model.py

# View rollout results
python3 scripts/rollout_viewer.py

# Convert HuggingFace to Megatron format
python3 scripts/converter_hf_to_mcore.py
```

## Installation Scripts
```bash
# Install vLLM, SGLang, and Megatron
scripts/install_vllm_sglang_mcore.sh
```

## Typical Training Workflow
1. Choose algorithm (PPO, GRPO, SFT, etc.)
2. Navigate to corresponding example directory
3. Review and modify run scripts as needed
4. Execute training script
5. Monitor training progress via configured logging (wandb, tensorboard, etc.)