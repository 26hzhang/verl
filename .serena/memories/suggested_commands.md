# Suggested Commands for verl Development

## Installation
```bash
# Python-only installation (quick iteration)
pip install -e .[test,vllm]
# or
pip install -e .[test,sglang]

# GPU installation with all dependencies
pip install -e .[test,vllm,gpu]
```

## Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit
pre-commit install

# Run on staged changes
pre-commit run

# Run on all files
pre-commit run --all-files

# Run specific hook
pre-commit run --all-files --show-diff-on-failure --color=always ruff
pre-commit run --all-files --show-diff-on-failure --color=always mypy
pre-commit run --all-files --show-diff-on-failure --color=always autogen-trainer-cfg
```

## Testing
```bash
# Run tests locally (check CI workflows for specific patterns)
pytest tests/

# Specific test categories
pytest tests/test_protocol_v2_on_cpu.py
pytest tests/utils/
pytest tests/trainer/

# With coverage
pytest --cov=verl tests/
```

## Linting and Formatting
```bash
# Format code with ruff
ruff check --fix .
ruff format .

# Type checking with mypy
mypy verl/
```

## Documentation
```bash
# Install doc dependencies
cd docs
pip install -r requirements-docs.txt

# Build HTML docs
make clean
make html

# Preview locally
python -m http.server -d _build/html/
# Open http://localhost:8000
```

## Running Training Examples
```bash
# PPO training
bash examples/ppo_trainer/run_qwen2-7b.sh

# GRPO training
bash examples/grpo_trainer/run_qwen3-8b.sh

# CPO training
bash examples/cpo/run_qwen3_math_grpo_cpo_seqkl_neg.sh

# SFT training
bash examples/sft/gsm8k/run_qwen_05.sh
```

## Utility Commands (macOS/Darwin specific)
```bash
# Standard Unix commands work on Darwin
ls, cd, grep, find, git, etc.

# File operations
cat <file>          # View file
head -n <N> <file>  # First N lines
tail -n <N> <file>  # Last N lines

# Search
grep -r "pattern" .
find . -name "*.py"

# Git operations
git status
git log --oneline
git diff
```

## Configuration Generation
```bash
# Generate trainer config (runs automatically in pre-commit)
bash scripts/generate_trainer_config.sh
```

## Development Workflow
1. Make code changes
2. Run `pre-commit run` to check formatting and linting
3. Run relevant tests: `pytest tests/path/to/test.py`
4. Build docs if needed: `cd docs && make html`
5. Commit changes (pre-commit hooks will run automatically)
