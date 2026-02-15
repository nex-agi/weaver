---
name: testing
description: Run linting and tests for Weaver SDK. Use when running tests, checking coverage, or verifying changes.
---

# Weaver SDK Testing Skill

## How to Use

1. Read agent instructions at `.claude/agents/testing/AGENT.md`
2. Invoke Task tool with `subagent_type="testing"` (specialized agent)
3. Agent will lint and run all tests

## Testing Workflow

```bash
# 1. Check formatting
make check-format

# 2. Run linters
make lint

# 3. Run tests
make test

# 4. Full CI (lint + test)
make ci
```

## Test Commands

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_config.py -v

# Specific test
pytest tests/test_config.py::test_config_defaults -v

# With coverage
make test-cov
```

## Test Patterns

```python
def test_service_client_initialization():
    """Test ServiceClient can be initialized with custom config."""
    client = ServiceClient(base_url="https://test.example.com")
    assert client._config.base_url == "https://test.example.com"
```

## Lint Commands

```bash
# Format check (no changes)
black --check weaver/ tests/
isort --check-only weaver/ tests/

# Format (apply changes)
make format

# Pylint
pylint weaver/

# Mypy
mypy weaver/

# License headers
python tests/lint/check_license_header.py
```

## Output Format

```
## Testing Summary
**Status:** PASS / WARNINGS / FAIL

### Lint Results
[black/isort/pylint/mypy output]

### Test Results
- Total: X | Passed: X | Failed: X | Skipped: X

### Failures
[Failed test details if any]

### Recommendations
[Actions to fix issues]
```

## Related Skills

- **`code-review`** - Code review (runs in parallel)
- **`git-commit`** - Complete commit workflow
