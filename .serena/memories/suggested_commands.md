# Suggested Development Commands

## Setup and Installation
```bash
# Install in development mode
pip install -e .[test,vllm]  # or .[test,sglang]

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

## Code Quality Commands
```bash
# Run pre-commit on staged files
pre-commit run

# Run pre-commit on all files
pre-commit run --all-files

# Run specific pre-commit hooks
pre-commit run --all-files --show-diff-on-failure --color=always ruff
pre-commit run --all-files --show-diff-on-failure --color=always autogen-trainer-cfg
pre-commit run --all-files --show-diff-on-failure --color=always check-docstrings
pre-commit run --all-files --show-diff-on-failure --color=always check-license
```

## Testing Commands
```bash
# Run CPU unit tests
pytest tests/**/test_*_on_cpu.py

# Run GPU unit tests (requires GPU)
pytest tests/ --ignore=tests/special_npu --ignore=tests/special_distributed --ignore=tests/special_e2e

# Run specific test categories
pytest tests/trainer/  # trainer tests
pytest tests/models/   # model tests
pytest tests/workers/  # worker tests
```

## Configuration Management
```bash
# Generate trainer configuration files
scripts/generate_trainer_config.sh

# Print configuration
python3 scripts/print_cfg.py --cfg job
python3 scripts/print_cfg.py --cfg job --config-name=ppo_megatron_trainer.yaml
```

## Documentation
```bash
# Install documentation dependencies
pip install -r requirements-docs.txt

# Build documentation
make clean
make html

# Preview documentation locally
python -m http.server -d _build/html/
```

## Git and Version Control
```bash
# Standard git commands work on Darwin
git status
git add .
git commit -m "message"
git push
```

## System Utilities (Darwin-specific)
```bash
# File operations
ls -la
find . -name "*.py" -type f
grep -r "pattern" .

# Process management
ps aux | grep python
top
```

## Development Workflow
1. Make changes to code
2. Run `pre-commit run` to check formatting and linting
3. Run relevant tests with pytest
4. Generate config files if needed with `scripts/generate_trainer_config.sh`
5. Commit and push changes