# verl Project Overview

## Purpose
verl (Volcano Engine Reinforcement Learning for LLMs) is a flexible, efficient and production-ready RL training library for large language models (LLMs). It's the open-source version of the HybridFlow paper and is maintained by ByteDance Seed team and the verl community.

## Key Features
- **Easy extension of diverse RL algorithms**: PPO, GRPO, GSPO, ReMax, REINFORCE++, RLOO, PRIME, DAPO, etc.
- **Seamless integration with existing LLM infra**: FSDP, FSDP2, Megatron-LM for training; vLLM, SGLang, HF Transformers for rollout
- **Compatible with popular models**: Qwen-3, Qwen-2.5, Llama3.1, Gemma2, DeepSeek-LLM, etc.
- **Scalability**: Supports up to 671B models and hundreds of GPUs
- **Multi-modal support**: Vision-language models (VLMs) and multi-modal RL
- **Advanced features**: Flash attention 2, sequence packing, LoRA, Liger-kernel

## Tech Stack
- **Language**: Python (>=3.10)
- **Core dependencies**: PyTorch, Ray, Transformers, accelerate, flash-attn
- **Training backends**: FSDP, FSDP2, Megatron-LM
- **Inference engines**: vLLM, SGLang, HuggingFace Transformers
- **Configuration**: Hydra-core for config management
- **Data**: TensorDict, datasets, pandas, pyarrow
- **Experiment tracking**: wandb, swanlab, mlflow, tensorboard

## Architecture
- Hybrid-controller programming model for flexible RL dataflows
- Modular APIs for seamless integration
- Flexible device mapping for efficient resource utilization
- 3D-HybridEngine for efficient actor model resharding