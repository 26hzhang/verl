# verl Project Overview

## Purpose
verl (Volcano Engine Reinforcement Learning for LLMs) is a flexible, efficient, and production-ready RL training library for large language models (LLMs). It's the open-source version of the HybridFlow framework.

## Tech Stack
- **Language**: Python (>=3.10)
- **Core Dependencies**: 
  - PyTorch (with FSDP, FSDP2)
  - Ray (>=2.41.0) for distributed execution
  - Transformers (HuggingFace)
  - TensorDict (0.8.0-0.10.0, excluding 0.9.0)
  - Hydra-core for configuration
  - Optional: vLLM, SGLang, Megatron-LM
- **Build System**: setuptools with pyproject.toml
- **Version Management**: Dynamic version from verl/version/version file

## Key Features
- Multiple RL algorithms: PPO, GRPO, CPO, GSPO, ReMax, RLOO, DAPO, etc.
- Training backends: FSDP, FSDP2, Megatron-LM
- Inference backends: vLLM, SGLang, HF Transformers
- Model support: Qwen-3, Qwen-2.5, Llama3.1, Gemma2, DeepSeek, etc.
- Advanced features: Flash attention 2, sequence packing, sequence parallelism, LoRA
- Scales to 671B models with expert parallelism
- Multi-modal RL support (VLMs)

## Repository Structure
```
verl/
├── verl/                      # Main package
│   ├── trainer/              # Training logic (PPO, GRPO, etc.)
│   ├── workers/              # Actor, critic, rollout workers
│   ├── models/               # Model wrappers and engines
│   ├── utils/                # Utilities (datasets, rewards, etc.)
│   ├── single_controller/    # Ray-based distributed control
│   ├── interactions/         # Interaction protocols
│   ├── cpo/                  # CPO-specific implementation
│   └── experimental/         # Experimental features (agent loop, etc.)
├── examples/                 # Example training scripts
│   ├── ppo_trainer/
│   ├── grpo_trainer/
│   ├── cpo/
│   └── sft/
├── recipe/                   # Training recipes (DAPO, SPPO, etc.)
├── tests/                    # Test suite
├── docs/                     # Documentation
└── scripts/                  # Utility scripts
```

## Main Entry Points
- Training scripts are in `examples/` and root directory
- Trainer implementations in `verl/trainer/`
- Configuration uses Hydra with YAML files in `verl/trainer/config/`
