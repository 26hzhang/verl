# Codebase Structure

## Main Package Structure (`verl/`)
- `verl/__init__.py` - Package initialization with version management
- `verl/base_config.py` - Base configuration classes
- `verl/protocol.py` - Protocol definitions
- `verl/py.typed` - Type information marker

### Core Directories
- `verl/trainer/` - Training logic and algorithms (PPO, GRPO, etc.)
- `verl/workers/` - Worker implementations for distributed training
- `verl/models/` - Model definitions and adapters
- `verl/utils/` - Utility functions and helpers
- `verl/interactions/` - Interaction handling between components
- `verl/tools/` - Additional tools and utilities
- `verl/single_controller/` - Single controller implementations
- `verl/third_party/` - Third-party integrations
- `verl/experimental/` - Experimental features
- `verl/model_merger/` - Model merging utilities
- `verl/version/` - Version information

## Examples (`examples/`)
Organized by trainer/algorithm type:
- `ppo_trainer/` - PPO training examples
- `grpo_trainer/` - GRPO training examples
- `sft/` - Supervised fine-tuning examples
- `generation/` - Generation examples
- `sglang_multiturn/` - Multi-turn conversation examples
- `split_placement/` - Distributed placement examples
- `ray/` - Ray integration examples
- `slurm/` - SLURM cluster examples
- `skypilot/` - SkyPilot cloud examples

## Algorithm Recipes (`recipe/`)
- Contains specific algorithm implementations and recipes
- Each algorithm typically has its own subdirectory

## Tests (`tests/`)
Mirrors the main package structure:
- `tests/trainer/` - Trainer tests
- `tests/models/` - Model tests  
- `tests/workers/` - Worker tests
- `tests/utils/` - Utility tests
- `tests/special_*` - Special test categories:
  - `special_distributed/` - Multi-GPU tests
  - `special_e2e/` - End-to-end tests
  - `special_npu/` - NPU-specific tests
  - `special_sanity/` - Quick sanity checks
  - `special_standalone/` - Standalone environment tests

## Configuration
- `pyproject.toml` - Main project configuration
- `requirements*.txt` - Dependency specifications
- `.pre-commit-config.yaml` - Pre-commit hooks
- `verl/trainer/config/` - Training configuration YAML files

## Scripts (`scripts/`)
- `generate_trainer_config.sh` - Generate configuration files
- `print_cfg.py` - Print configuration
- `diagnose.py` - Diagnostic utilities
- `install_vllm_sglang_mcore.sh` - Installation script

## Documentation
- `docs/` - Documentation source
- `claude_docs/` - Claude-specific documentation  
- `README.md` - Main project documentation
- `CONTRIBUTING.md` - Contribution guidelines