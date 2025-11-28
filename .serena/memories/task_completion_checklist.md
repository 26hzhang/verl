# Task Completion Checklist

When completing a coding task in verl, ensure the following:

## 1. Code Quality
- [ ] Code follows ruff formatting rules (120 char line length)
- [ ] Imports are sorted correctly (isort)
- [ ] Type hints added where appropriate (especially for typed modules)
- [ ] Docstrings added for new functions/classes
- [ ] License header included in new files

## 2. Pre-commit Checks
Run all pre-commit hooks before considering task complete:
```bash
pre-commit run --all-files
```

This will automatically check:
- [ ] Ruff linting and formatting
- [ ] mypy type checking (for strict modules)
- [ ] Trainer config generation (autogen-trainer-cfg)
- [ ] Docstring coverage
- [ ] License header presence
- [ ] Python compilation (no syntax errors)

## 3. Testing
- [ ] Add or update tests for new functionality
- [ ] Ensure existing tests pass: `pytest tests/`
- [ ] Consider CI test coverage (see .github/workflows/)
- [ ] For new features, add CI workflow tests if applicable

## 4. Documentation
- [ ] Update docs if adding user-facing features
- [ ] Build docs to verify no errors: `cd docs && make html`
- [ ] Update README.md if adding major features
- [ ] Update CONTRIBUTING.md if changing development workflow

## 5. Configuration
- [ ] Update YAML configs in verl/trainer/config/ if needed
- [ ] Run config generation script: `bash scripts/generate_trainer_config.sh`
- [ ] Ensure Hydra configs are valid

## 6. For New RL Algorithms
- [ ] Add to AdvantageEstimator enum if applicable
- [ ] Register with appropriate registry (ADV_ESTIMATOR_REGISTRY, etc.)
- [ ] Add example training script in examples/
- [ ] Update algorithm config dataclass
- [ ] Add integration tests

## 7. Git Workflow
- [ ] Commit messages are descriptive
- [ ] Changes are focused and logical
- [ ] No unrelated changes included
- [ ] Branch is up to date with main

## 8. Performance Considerations
- [ ] No obvious performance regressions
- [ ] Memory usage is reasonable
- [ ] Distributed training considerations addressed
- [ ] GPU memory efficiency maintained
