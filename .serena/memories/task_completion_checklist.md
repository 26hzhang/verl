# Task Completion Checklist

When completing any development task in verl, follow this checklist:

## Code Quality Checks
1. **Pre-commit hooks**: Run `pre-commit run --all-files` to ensure:
   - Ruff formatting and linting passes
   - MyPy type checking passes (where enabled)
   - License headers are present
   - Docstring coverage is adequate
   - Configuration files are up-to-date

2. **Specific pre-commit commands**:
   ```bash
   pre-commit run --all-files --show-diff-on-failure --color=always ruff
   pre-commit run --all-files --show-diff-on-failure --color=always autogen-trainer-cfg
   pre-commit run --all-files --show-diff-on-failure --color=always check-docstrings
   pre-commit run --all-files --show-diff-on-failure --color=always check-license
   ```

## Configuration Management
3. **Generated configs**: If you modify trainer configs, regenerate with:
   ```bash
   scripts/generate_trainer_config.sh
   ```

## Testing Requirements
4. **Unit tests**: Run appropriate tests based on changes:
   ```bash
   # For CPU-compatible changes
   pytest tests/**/test_*_on_cpu.py
   
   # For GPU-requiring changes (if GPU available)
   pytest tests/ --ignore=tests/special_npu --ignore=tests/special_distributed --ignore=tests/special_e2e
   
   # For specific components
   pytest tests/trainer/  # if trainer changes
   pytest tests/models/   # if model changes
   pytest tests/workers/  # if worker changes
   ```

5. **Add new tests**: For new features, add corresponding tests in the `tests/` directory

## Documentation
6. **Update documentation**: For user-facing changes, update relevant documentation
7. **Docstrings**: Ensure all new functions/classes have proper docstrings

## Final Checks
8. **License compliance**: All new Python files must have Apache-2.0 license headers
9. **Code style**: Follow existing patterns and conventions in the codebase
10. **Dependencies**: Only use libraries already present in the codebase or explicitly add new ones to requirements

## CI/CD Considerations
- Changes should pass all relevant GitHub Actions workflows
- Consider adding new CI tests if introducing significant new functionality
- Minimize test workload while maintaining coverage

## Version Control
- Commit messages should be descriptive
- Consider breaking large changes into smaller, logical commits
- Ensure no sensitive information is committed