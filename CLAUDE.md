# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

本文件为 Claude Code (claude.ai/code) 在此代码库中工作提供指导。

## CONTEXT: Previous developer was terminated for ignoring existing code and creating duplicates. You must prove you can work within existing architecture.

## MANDATORY PROCESS:

1. Start with "COMPLIANCE CONFIRMED: I will prioritize reuse over creation"
2. Analyze existing code BEFORE suggesting anything new
3. Reference specific files from the provided analysis
4. Include validation checkpoints throughout your response
5. End with compliance confirmation

## RULES (violating ANY invalidates your response):

- ❌ No new files without exhaustive reuse analysis
- ❌ No rewrites when refactoring is possible
- ❌ No generic advice - provide specific implementations
- ❌ No ignoring existing codebase architecture
- ✅ Extend existing services and components
- ✅ Consolidate duplicate code
- ✅ Reference specific file paths
- ✅ Provide migration strategies

## Core Development Commands / 核心开发命令

### Code Quality and Pre-commit / 代码质量与预提交检查
```bash
# Install pre-commit hooks (run once) / 安装预提交钩子（仅需运行一次）
pip install pre-commit && pre-commit install

# Run pre-commit on staged files / 对暂存文件运行预提交检查
pre-commit run

# Run pre-commit on all files / 对所有文件运行预提交检查
pre-commit run --all-files

# Run specific hooks / 运行特定钩子
pre-commit run --all-files ruff                    # Lint and format / 代码检查和格式化
pre-commit run --all-files autogen-trainer-cfg     # Generate config files / 生成配置文件
pre-commit run --all-files check-docstrings        # Check docstring coverage / 检查文档字符串覆盖率
pre-commit run --all-files check-license           # Verify license headers / 验证许可证头部
```

### Testing / 测试
```bash
# CPU unit tests (can run without GPU) / CPU单元测试（无需GPU）
pytest tests/**/test_*_on_cpu.py

# GPU unit tests (requires GPU) / GPU单元测试（需要GPU）
pytest tests/ --ignore=tests/special_npu --ignore=tests/special_distributed --ignore=tests/special_e2e

# Component-specific tests / 组件特定测试
pytest tests/trainer/     # Trainer tests / 训练器测试
pytest tests/models/      # Model tests / 模型测试
pytest tests/workers/     # Worker tests / 工作器测试

# Run single test file / 运行单个测试文件
pytest tests/test_protocol_on_cpu.py -v

# Run specific test with detailed output / 运行特定测试并显示详细输出
pytest tests/trainer/test_ppo.py::TestPPO::test_basic_training -v -s
```

### Configuration Management / 配置管理
```bash
# Generate trainer configuration files (required after config changes) / 生成训练器配置文件（修改配置后必需）
scripts/generate_trainer_config.sh

# Print configuration for debugging / 打印配置用于调试
python3 scripts/print_cfg.py --cfg job
python3 scripts/print_cfg.py --cfg job --config-name=ppo_megatron_trainer.yaml

# Validate configuration / 验证配置
python3 -c "from verl.trainer.config import parse_config; parse_config('ppo_trainer')"
```

### Installation and Development Setup / 安装与开发环境设置
```bash
# Install in development mode / 开发模式安装
pip install -e .[test,vllm]    # For vLLM backend / 用于vLLM后端
pip install -e .[test,sglang]  # For SGLang backend / 用于SGLang后端

# Install third-party dependencies / 安装第三方依赖
scripts/install_vllm_sglang_mcore.sh

# Install for specific use cases / 特定用途安装
pip install -e .[test,vllm,megatron]  # Full installation / 完整安装
pip install -e .[dev]                 # Development tools only / 仅开发工具
```

### Training Commands / 训练命令
```bash
# Basic PPO training / 基础PPO训练
python3 -m verl.trainer.main_ppo \
    data.train_files="['/path/to/train.parquet']" \
    data.val_files="['/path/to/val.parquet']" \
    actor_rollout_ref.model.path="/path/to/model"

# GRPO training / GRPO训练
python3 -m verl.trainer.main_grpo \
    data.train_files="['/path/to/train.parquet']" \
    algorithm.grpo.beta=0.1

# SFT training / 监督微调
python3 -m verl.trainer.main_sft \
    data.train_files="['/path/to/train.parquet']" \
    model.path="/path/to/base_model"

# Multi-node training with Ray / 使用Ray进行多节点训练
ray start --head --port=6379
python3 -m verl.trainer.main_ppo trainer.nnodes=4 trainer.n_gpus_per_node=8
```

### Debugging and Diagnostics / 调试和诊断
```bash
# System diagnostics / 系统诊断
python3 scripts/diagnose.py

# View rollout data / 查看生成数据
python3 scripts/rollout_viewer.py --path /path/to/rollout.jsonl

# Profile training performance / 分析训练性能
python3 -m verl.trainer.main_ppo trainer.enable_profiling=True

# Monitor Ray cluster / 监控Ray集群
ray status
ray dashboard  # Web UI at http://localhost:8265
```

## Architecture Overview / 架构概述

verl implements a **hybrid-controller programming model** for distributed reinforcement learning training of large language models. The architecture separates concerns across multiple layers:

verl 实现了用于大语言模型分布式强化学习训练的**混合控制器编程模型**。该架构在多个层次上分离关注点：

### Core Components / 核心组件

**1. Trainer Layer (`verl/trainer/`) / 训练器层**
- `main_ppo.py`: Entry point with `TaskRunner` class that orchestrates distributed PPO training / 入口点，包含编排分布式PPO训练的`TaskRunner`类
- `ppo/ray_trainer.py`: `RayPPOTrainer` handles the main training loop and worker coordination / `RayPPOTrainer`处理主训练循环和工作器协调
- `ppo/core_algos.py`: Core PPO algorithm implementations / 核心PPO算法实现
- `ppo/reward.py`: Reward function processing and KL penalty computation / 奖励函数处理和KL惩罚计算
- Supports multiple algorithms: PPO, GRPO, SFT, RLOO, ReMax, REINFORCE++ / 支持多种算法

**2. Worker Layer (`verl/workers/`) / 工作器层**
- `fsdp_workers.py`: FSDP/FSDP2 backend implementations with classes:
  - `ActorRolloutRefWorker`: Handles actor model and rollout generation / 处理actor模型和rollout生成
  - `CriticWorker`: Manages value function estimation / 管理价值函数估计
  - `RewardModelWorker`: Processes reward model inference / 处理奖励模型推理
  - `AsyncActorRolloutRefWorker`: Asynchronous rollout generation / 异步rollout生成
- `megatron_workers.py`: Megatron-LM backend implementations / Megatron-LM后端实现
- `roles/`: New modular worker implementations / 新的模块化工作器实现

**3. Single Controller (`verl/single_controller/`) / 单一控制器**
- `ray/`: Ray-based distributed coordination / 基于Ray的分布式协调
- `RayWorkerGroup`: Manages groups of Ray remote workers / 管理Ray远程工作器组
- `ResourcePoolManager`: Handles GPU resource allocation across different roles / 处理不同角色间的GPU资源分配

**4. Models (`verl/models/`) / 模型层**
- Model adapters for different frameworks (HuggingFace, Megatron) / 不同框架的模型适配器
- Support for various model architectures (Qwen, Llama, DeepSeek, etc.) / 支持各种模型架构
- Vision-language model support with processors / 支持视觉-语言模型和处理器

### Training Flow Architecture / 训练流程架构

The training follows this distributed pattern: / 训练遵循以下分布式模式：

1. **TaskRunner** (Ray remote) orchestrates the entire training process / 编排整个训练过程
2. **Resource allocation** maps different roles (Actor, Critic, RewardModel) to GPU pools / 资源分配将不同角色映射到GPU池
3. **Worker initialization** creates specialized workers based on strategy (FSDP vs Megatron) / 工作器初始化根据策略创建专门的工作器
4. **Training loop** coordinates between: / 训练循环协调以下组件：
   - **Actor/Rollout**: Generates responses from prompts / 从提示生成响应
   - **Critic**: Estimates value functions / 估计价值函数
   - **Reward Model**: Scores generated responses / 对生成的响应评分
   - **Reference Policy**: Provides KL penalty baseline (optional) / 提供KL惩罚基线（可选）

### Key Architectural Patterns / 关键架构模式

**Hybrid Backend Support** / 混合后端支持: The same high-level training logic works with different backends:
- FSDP/FSDP2 for PyTorch native distributed training / PyTorch原生分布式训练
- Megatron-LM for large-scale model parallelism / 大规模模型并行
- vLLM/SGLang for efficient inference / 高效推理

**Role-based Worker System** / 基于角色的工作器系统: Workers are organized by their function in the RL pipeline rather than by model type, enabling flexible resource allocation and scaling. / 工作器按照RL流水线中的功能而非模型类型组织，实现灵活的资源分配和扩展。

**Ray Remote Architecture** / Ray远程架构: Heavy use of Ray for distributed coordination, allowing the system to scale across multiple nodes while maintaining a clean programming interface. / 大量使用Ray进行分布式协调，允许系统在多个节点间扩展，同时保持清洁的编程接口。

## Configuration System / 配置系统

Uses Hydra for hierarchical configuration management. Key config areas: / 使用Hydra进行分层配置管理。关键配置区域：

- `verl/trainer/config/`: YAML configuration files / YAML配置文件
- Algorithm parameters, model settings, data processing, resource allocation / 算法参数、模型设置、数据处理、资源分配
- Auto-generated flattened configs in `_generated_*.yaml` files (do not edit directly) / 自动生成的扁平化配置文件（请勿直接编辑）

### Key Configuration Patterns / 关键配置模式

```yaml
# FSDP strategy configuration / FSDP策略配置
actor_rollout_ref:
  actor:
    strategy: fsdp2  # or fsdp, megatron
    fsdp_config:
      param_offload: false
      optimizer_offload: false
  rollout:
    name: vllm  # or sglang, hf_transformers
    tensor_model_parallel_size: 1

# Algorithm configuration / 算法配置
algorithm:
  adv_estimator: gae  # or vanilla
  use_kl_in_reward: true
  kl_ctrl:
    kl_coef: 0.1
    adaptive_kl: true

# Data configuration / 数据配置
data:
  train_batch_size: 1024
  max_prompt_length: 1024
  max_response_length: 512
  filter_overlong_prompts: true
```

## Examples and Recipes Structure / 示例与配方结构

**Examples (`examples/`)**: Ready-to-run training scripts organized by algorithm / 按算法组织的可运行训练脚本
- Each subdirectory contains shell scripts for specific model/dataset combinations / 每个子目录包含特定模型/数据集组合的shell脚本
- Follow naming pattern: `run_<model>_<dataset/variant>.sh` / 遵循命名模式

**Recipes (`recipe/`)**: Algorithm-specific implementations and research recipes / 算法特定实现和研究配方
- Contains implementations of newer algorithms (DAPO, SPPO, PRIME, etc.) / 包含新算法的实现
- Often includes paper reproduction code and specific hyperparameter settings / 通常包含论文复现代码和特定超参数设置

### Example Usage Patterns / 示例使用模式

```bash
# Run Qwen2-7B with reward model / 使用奖励模型运行Qwen2-7B
examples/ppo_trainer/run_qwen2-7b_rm.sh

# Run with sequence balancing for better efficiency / 使用序列平衡提高效率
examples/ppo_trainer/run_qwen2-7b_seq_balance.sh

# Multi-modal training with vision-language model / 视觉-语言模型的多模态训练
examples/grpo_trainer/run_qwen2_5_vl-7b.sh

# Large model training with Megatron / 使用Megatron训练大模型
examples/ppo_trainer/run_qwen2.5-32b.sh
```

## Technical Deep Dives / 技术深入解析

### Worker Implementation Details / 工作器实现细节

**ActorRolloutRefWorker** handles: / 处理：
- Model loading and sharding / 模型加载和分片
- Forward passes for training / 训练的前向传播
- Response generation (rollout) / 响应生成（rollout）
- Reference policy computation / 参考策略计算

**CriticWorker** manages: / 管理：
- Value function model / 价值函数模型
- Advantage computation / 优势计算
- Critic loss calculation / Critic损失计算

**RewardModelWorker** handles: / 处理：
- Reward model inference / 奖励模型推理
- Batch processing of responses / 响应的批处理
- Reward aggregation and normalization / 奖励聚合和归一化

### Memory and Performance Optimization / 内存与性能优化

```python
# Enable gradient checkpointing / 启用梯度检查点
actor_rollout_ref.model.enable_gradient_checkpointing=True

# Use sequence packing for efficiency / 使用序列打包提高效率
data.pack_sequences=True

# FSDP offloading options / FSDP卸载选项
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True

# Memory-efficient attention / 内存高效注意力
actor_rollout_ref.model.use_remove_padding=True
```

### Advanced Features / 高级特性

**Async Rollout** / 异步Rollout: Enables overlapping of generation and training phases / 实现生成和训练阶段的重叠

**3D-HybridEngine**: Efficient model resharding between training and inference / 训练和推理间的高效模型重分片

**Multi-turn Conversations**: Support for conversation context in RL training / 支持RL训练中的对话上下文

**Tool Integration**: Framework for incorporating external tools in RL training / 在RL训练中集成外部工具的框架

## Development Notes / 开发说明

- **License headers required**: All Python files must include Apache-2.0 license headers / 所有Python文件必须包含Apache-2.0许可证头部
- **Config regeneration**: Run `scripts/generate_trainer_config.sh` after modifying trainer configs / 修改训练器配置后运行配置生成脚本
- **Multi-modal support**: Framework supports vision-language models via processor integration / 通过处理器集成支持视觉-语言模型
- **Resource-aware design**: Training and inference can be placed on different GPU sets for optimal utilization / 训练和推理可以放置在不同GPU集合上以优化利用率
- **Code style**: 120 character line limit, use Ruff for formatting / 120字符行限制，使用Ruff格式化
- **Testing**: Always add tests for new features in appropriate `tests/` subdirectories / 总是在适当的测试子目录中为新特性添加测试

## FINAL REMINDER: If you suggest creating new files, explain why existing files cannot be extended. If you recommend rewrites, justify why refactoring won't work.

- 🔍 STEP 1 - ANALYZE CURRENT SYSTEM: Analyze the existing codebase and identify relevant files for the requested feature implementation. Then proceed to Step 2.
- 🎯 STEP 2 - CREATE IMPLEMENTATION PLAN: Based on your analysis from Step 1, create a detailed implementation plan for the requested feature. Then proceed to Step 3.
- 🔧 STEP 3 - PROVIDE TECHNICAL DETAILS: Create the technical implementation details including code changes, API modifications, and integration points. Then proceed to Step 4.
- ✅ STEP 4 - FINALIZE DELIVERABLES: Complete the implementation plan with testing strategies, deployment considerations, and final recommendations.
- 🎯 INSTRUCTIONS: Follow each step sequentially. Complete one step before moving to the next. Use the findings from each previous step to inform the next step.