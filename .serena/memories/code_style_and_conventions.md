# Code Style and Conventions

## Linting and Formatting
- **Primary tool**: Ruff for linting and formatting
- **Line length**: 120 characters
- **Import sorting**: isort with verl as known-first-party
- **Type checking**: MyPy (with ignore_missing_imports=true, mostly ignored except for specific modules)

## Ruff Configuration
- **Selected rules**: pycodestyle (E), Pyflakes (F), pyupgrade (UP), flake8-bugbear (B), isort (I), logging (G)
- **Ignored rules**: 
  - F405, F403 (star imports)
  - E731 (lambda expression assignment)
  - B007 (loop control variable not used)
  - UP032 (f-string format)
  - G004 (f-string in log statements)
  - UP045, UP035 (type annotations and deprecated imports)

## File Structure Conventions
- Main package: `verl/`
- Examples: `examples/` (organized by trainer type)
- Tests: `tests/` (mirroring package structure)
- Scripts: `scripts/` (utility scripts)
- Recipes: `recipe/` (algorithm implementations)

## Python Conventions
- **Minimum Python version**: 3.10
- **Type hints**: Encouraged but not strictly enforced (MyPy mostly disabled)
- **Docstrings**: Required and checked via pre-commit hooks
- **Import style**: Use isort with verl as known-first-party package

## License Headers
- All Python files must include Apache-2.0 license headers
- Checked via pre-commit hook `check-license`

## Package Data
- Version file: `verl/version/version`
- Config files: `verl/trainer/config/*.yaml` and subdirectories
- All YAML files included in package data