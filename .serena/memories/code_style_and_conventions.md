# Code Style and Conventions

## Linting and Formatting
- **Tool**: Ruff (configured in pyproject.toml)
- **Line Length**: 120 characters
- **Import Sorting**: isort with first-party package "verl"

### Ruff Configuration
- Enabled rules:
  - E (pycodestyle)
  - F (Pyflakes)
  - UP (pyupgrade)
  - B (flake8-bugbear)
  - I (isort)
  - G (logging format)
  
- Ignored rules:
  - F405, F403 (star imports allowed)
  - E731 (lambda assignment allowed)
  - B007 (loop control variable not used)
  - UP032 (f-string format)
  - G004 (f-strings in logging)
  - UP045, UP035 (X | None, deprecated imports)

## Type Checking
- **Tool**: mypy (v1.17.0)
- **General Policy**: `ignore_errors = true` (blanket silence)
- **Strict Modules** (ignore_errors = false):
  - verl.trainer.config.algorithm
  - verl.trainer.ppo.core_algos
  - verl.trainer.ppo.reward
  - verl.workers.reward_manager.*

## Naming Conventions
- Functions: snake_case
- Classes: PascalCase
- Constants: UPPER_SNAKE_CASE
- Private members: _leading_underscore

## File Headers
All source files include Apache 2.0 license header:
```python
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License")
# ...
```

## Documentation
- Docstrings are checked via pre-commit hook
- Docstring coverage enforced by `tests/special_sanity/check_docstrings.py`

## Code Organization
- Use `__all__` to explicitly export public APIs
- Registry pattern for extensible components (e.g., advantage estimators, policy losses)
- Dataclasses for configuration with frozen fields where appropriate
- Type hints encouraged but not strictly enforced everywhere
