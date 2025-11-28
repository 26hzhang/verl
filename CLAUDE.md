# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

VERL (Volcano Engine Reinforcement Learning) is a distributed RL training framework for LLMs built on Ray. The codebase implements a **hybrid controller model** that enables efficient switching between training and inference modes on the same GPU resources.

## Essential Commands

### Development Setup
```bash
# Quick iteration (Python-only)
pip install -e .[test,vllm]  # or .[test,sglang]

# Full GPU setup
pip install -e .[test,vllm,gpu]
```

### Pre-commit Hooks (Always Run Before Committing)
```bash
pre-commit install
pre-commit run --all-files

# Critical: After modifying Hydra configs
bash scripts/generate_trainer_config.sh
```

### Testing
```bash
# CPU-only tests
pytest -s tests/*_on_cpu.py

# GPU tests (excluding special categories)
pytest -s --ignore-glob="*test_special_*.py" --ignore-glob='*on_cpu.py' tests/

# Specific test suites
pytest tests/trainer/
pytest tests/workers/
pytest tests/utils/

# Distributed tests (requires 2+ GPUs)
torchrun --standalone --nnodes=1 --nproc-per-node=2 tests/workers/actor/test_special_dp_actor.py
```

### Linting & Formatting
```bash
ruff check --fix .
ruff format .
mypy verl/  # Only strict on specific modules
```

## Architecture Overview

### Core Design Pattern: Hybrid Controller Model

```
Ray Cluster
    ↓
RayPPOTrainer (orchestrator)
    ↓
WorkerGroups (distributed workers)
├── ActorRolloutRefWorker (hybrid: training + inference)
├── CriticWorker (value function)
└── RewardModelWorker (scoring)
    ↓
DataProto (unified data protocol)
```

**Key Insight**: Workers switch between `trainer_mode()` and `rollout_mode()` without reloading weights, enabling efficient memory utilization.

### Critical Abstractions

#### 1. DataProto (`verl/protocol.py`)
Universal data container for inter-component communication:
- `batch`: TensorDict for tensors with same batch size
- `non_tensor_batch`: Dict for numpy arrays
- `meta_info`: Metadata and metrics

Operations: slicing, chunking, padding, serialization, fold/unfold batch dimensions.

#### 2. Worker Groups (`verl/single_controller/`)
Ray-based distributed worker management:
- **ResourcePool**: Tracks GPU allocations across nodes
- **RayWorkerGroup**: Manages Ray actors with dispatch/collect pattern
- **Dispatch Modes**: `ONE_TO_ALL`, `GATHER`, `DP_COMPUTE`, `MEGATRON_COMPUTE`

#### 3. Registry Pattern
Extensible algorithm registration:
```python
@register_adv_est("my_estimator")  # Advantage estimators
@register_policy_loss("my_loss")   # Policy loss functions
```

Registries:
- `ADV_ESTIMATOR_REGISTRY`: GAE, GRPO, REINFORCE++, RLOO, CPO, etc.
- `POLICY_LOSS_REGISTRY`: Vanilla PPO, GSPO, GPG, KL coverage, etc.

### Training Backends

**FSDP** (`verl/workers/fsdp_workers.py`):
- Fully Sharded Data Parallel with FSDP2 support
- Ulysses Sequence Parallel integration
- CPU offloading for params/optimizer
- LoRA support

**Megatron** (`verl/workers/megatron_workers.py`):
- Tensor Parallel (TP), Pipeline Parallel (PP), Data Parallel (DP)
- Expert Model Parallel for MoE models
- Virtual Pipeline Parallel

### Inference Backends

**Rollout Engines** (`verl/workers/rollout/`):
- vLLM: Sync/async modes
- SGLang: Sync/async with server adapter
- HF Transformers: Fallback implementation

**Weight Resharding** (`verl/workers/sharding_manager/`):
Manages weight transformation between training (FSDP/Megatron) and inference (vLLM/SGLang) parallelism strategies.

## Configuration System

### Hydra-based Hierarchical Configs

Main templates:
- `verl/trainer/config/ppo_trainer.yaml`: FSDP-based training
- `verl/trainer/config/ppo_megatron_trainer.yaml`: Megatron-based training

Structure:
```yaml
defaults:
  - actor@actor_rollout_ref.actor: dp_actor
  - rollout@actor_rollout_ref.rollout: rollout
  - model@actor_rollout_ref.model: hf_model
  - critic@critic: dp_critic

algorithm:
  adv_estimator: gae  # or grpo, cpo, etc.
  kl_ctrl:
    type: fixed
    kl_coef: 0.001
```

**Config Dataclasses**: All Hydra configs convert to frozen dataclasses with dict-like interface (`verl/base_config.py`).

**Critical Hook**: `autogen-trainer-cfg` generates flattened YAML references from Hydra configs. Must run after config changes:
```bash
bash scripts/generate_trainer_config.sh
```

## Training Flow

```
1. Rollout Phase
   └─ Switch workers to rollout_mode()
   └─ Generate sequences → DataProto

2. Reward Computation
   └─ Compute rewards and add to DataProto
   └─ Apply KL penalty (if enabled)

3. Advantage Estimation
   └─ Get estimator from registry (GAE/GRPO/CPO/etc.)
   └─ Compute advantages and returns

4. Policy Update
   └─ Switch workers to trainer_mode()
   └─ Actor: Compute policy loss, optimizer step
   └─ Critic: Compute value loss, optimizer step

5. Checkpoint & Metrics
   └─ Save model states
   └─ Log to tensorboard/wandb
```

## Key Extension Points

### Adding New RL Algorithms

**1. Register Advantage Estimator** (`verl/trainer/ppo/core_algos.py`):
```python
@register_adv_est("my_estimator")
def my_advantage_estimator(returns, values, gamma=1.0):
    advantages = compute_my_advantages(returns, values)
    return advantages, {"metric/key": value}
```

**2. Register Policy Loss**:
```python
@register_policy_loss("my_loss")
def my_loss(old_log_prob, log_prob, advantages, response_mask,
            loss_agg_mode, config):
    loss = compute_my_loss(...)
    return loss, {"actor/my_loss": loss.mean()}
```

**3. Update Config**:
```yaml
algorithm:
  adv_estimator: my_estimator
  policy_loss: my_loss
```

### Custom Reward Functions

Create reward function in separate file:
```python
def compute_score(data: DataProto) -> torch.Tensor:
    responses = data.batch["responses"]
    return your_scoring_logic(responses)
```

Reference in config:
```yaml
custom_reward_function:
  path: /path/to/reward.py
  name: compute_score
```

## Code Quality Requirements

### Pre-commit Checks (Enforced)
1. **ruff**: Linting + auto-fixing (120 char line length)
2. **ruff-format**: Code formatting
3. **mypy**: Type checking on strict modules
4. **autogen-trainer-cfg**: Config generation validation
5. **check-docstrings**: Docstring coverage for critical files
6. **check-license**: Apache 2.0 license headers
7. **compileall**: Python syntax validation

### Strict Type Checking Modules
mypy enforces strict typing on:
- `verl/trainer/config/algorithm`
- `verl/trainer/ppo/core_algos`
- `verl/trainer/ppo/reward`
- `verl/workers/reward_manager`

### Required Docstrings
Public functions/classes in these files must have docstrings:
- `verl/trainer/ppo/ray_trainer.py`
- `verl/trainer/ppo/core_algos.py`
- `verl/trainer/ppo/reward.py`
- `verl/experimental/agent_loop/agent_loop.py`
- `verl/workers/sharding_manager/fsdp_vllm.py`
- `verl/workers/sharding_manager/fsdp_ulysses.py`

## Testing Structure

### Test Categories
- **Standard tests**: Module-based in `tests/<module>/`
- **`special_sanity`**: Quick checks (imports, licenses, docstrings)
- **`special_distributed`**: Multi-GPU unit tests
- **`special_e2e`**: End-to-end training tests
- **`*_on_cpu.py`**: CPU-only tests

### CI/CD Workflows
- **Always runs**: sanity.yml, pre-commit.yml, doc.yml
- **Path-filtered**: GPU/CPU unit tests, e2e tests
- **Backend-specific**: model.yml, vllm.yml, sgl.yml

## Important Files Reference

**Core Orchestration**:
- `verl/trainer/ppo/ray_trainer.py`: Main training loop (2000+ lines)
- `verl/trainer/ppo/core_algos.py`: Algorithm implementations (1800+ lines)

**Worker Implementations**:
- `verl/workers/fsdp_workers.py`: FSDP ActorRolloutRefWorker
- `verl/workers/megatron_workers.py`: Megatron ActorRolloutRefWorker

**Data & Protocol**:
- `verl/protocol.py`: DataProto implementation (1241 lines)
- `verl/base_config.py`: Configuration base class

**Ray Integration**:
- `verl/single_controller/ray/base.py`: RayWorkerGroup, ResourcePool
- `verl/single_controller/base/decorator.py`: Dispatch/collect decorators

## Common Patterns

### Dispatch/Collect for Distributed Workers
```python
@register(dispatch_mode=Dispatch.ONE_TO_ALL)
def init_model(self):
    """Executed on all workers"""

@register(dispatch_mode=Dispatch.GATHER)
def compute_log_prob(self, data: DataProto):
    """Scatter batch, gather results"""
```

### Mode Switching (Hybrid Engine)
```python
async with worker.rollout_mode():
    data = await rollout.generate_sequences(prompts)

async with worker.trainer_mode():
    metrics = update_policy(data)
```

### DataProto Operations
```python
# Chunking for distribution
chunks = data.chunk(n_workers)

# Padding for balanced batches
padded = pad_dataproto_to_divisor(data, divisor=8)

# Adding new fields
data = data.union(new_data)
```

## Development Workflow

1. **Make changes** to code
2. **Run pre-commit** to catch issues: `pre-commit run --all-files`
3. **Regenerate configs** if modified: `bash scripts/generate_trainer_config.sh`
4. **Run relevant tests** locally: `pytest tests/module/`
5. **Commit** - CI runs full suite automatically

## CPO (Constrained Policy Optimization) Specifics

CPO extends GRPO with hint-based advantage adjustment:

**Key Parameters**:
- `cpo_lambda`: Ratio clipping (typical: 1.5-5.0, lower = more conservative)
- `pos_alpha`/`neg_alpha`: Positive/negative example scaling
- `wrap_method`: Advantage wrapper (seq_kl, negonly_mi3, etc.)

**Configuration**:
```yaml
algorithm:
  adv_estimator: cpo
  cpo_lambda: 2.0
  wrap_method: "seq_kl"

actor_rollout_ref:
  actor:
    cpo_lambda: 2.0  # Must match algorithm level
    wrap_method: "seq_kl"
```

**Implementation**:
- `verl/cpo/cpo_advantage_wrapper.py`: 17 wrapper strategies
- `verl/cpo/cpo_utils.py`: Hint injection, difficulty masks
- Activated when `algorithm.adv_estimator='cpo'`
