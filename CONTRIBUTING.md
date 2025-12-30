# Contributing to Weaver SDK

Thank you for your interest in contributing to the Weaver SDK!

## Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd weaver
   ```

2. **Install development dependencies**
   ```bash
   make install-dev
   # or
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**
   ```bash
   make pre-commit
   # or
   pre-commit install
   ```

## Development Workflow

### Before Making Changes

1. Create a new branch for your feature/fix
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make sure all tests pass
   ```bash
   make test
   ```

### Making Changes

1. Write your code following the project style
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure license headers are present in new files

### Before Committing

1. **Format your code**
   ```bash
   make format
   ```

2. **Run linters**
   ```bash
   make lint
   ```

3. **Run tests**
   ```bash
   make test
   ```

4. **Check everything passes**
   ```bash
   make ci
   ```

### Committing

Pre-commit hooks will automatically run when you commit. If they fail:

```bash
# Fix issues and try again
git add .
git commit -m "Your commit message"
```

## Code Style

- **Python Version**: Support Python 3.9+
- **Line Length**: 100 characters
- **Formatting**: Black + isort
- **Type Hints**: Use type hints where appropriate
- **Docstrings**: Add docstrings for public APIs

## Testing

### Writing Tests

- Place tests in `tests/` directory
- Name test files with `test_` prefix
- Use descriptive test function names

Example:
```python
def test_service_client_initialization():
    """Test ServiceClient can be initialized with custom config."""
    client = ServiceClient(base_url="https://test.example.com")
    assert client._config.base_url == "https://test.example.com"
```

### Running Tests

```bash
# All tests
make test

# With coverage
make test-cov

# Specific test file
pytest tests/test_config.py -v

# Specific test function
pytest tests/test_config.py::test_config_defaults -v
```

## License Headers

All source files must include the Apache 2.0 license header:

```python
# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

## Pull Request Process

1. **Update documentation** if you're changing APIs
2. **Add tests** for new features
3. **Ensure CI passes** - all tests and linters must pass
4. **Write clear commit messages**
5. **Reference issues** in your PR description

### PR Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] License headers present
- [ ] Code formatted (Black + isort)
- [ ] Linters pass (pylint, mypy)
- [ ] All tests pass
- [ ] Pre-commit hooks pass

## Questions?

If you have questions about contributing, please open an issue or reach out to the maintainers.

## Thank You!

Your contributions help make Weaver better for everyone!
