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

.PHONY: help install install-dev test lint format check clean

help:
	@echo "Available commands:"
	@echo "  make install       - Install package"
	@echo "  make install-dev   - Install package with dev dependencies"
	@echo "  make test          - Run unit tests"
	@echo "  make test-cov      - Run tests with coverage"
	@echo "  make lint          - Run all linters"
	@echo "  make format        - Format code with black and isort"
	@echo "  make check         - Run format check without modifying files"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make pre-commit    - Install pre-commit hooks"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=weaver --cov-report=html --cov-report=term

lint: check-format pylint mypy check-license

check-format:
	black --check weaver/ tests/
	isort --check-only weaver/ tests/

pylint:
	pylint weaver/ --rcfile=.pylintrc || true

mypy:
	mypy weaver/ || true

check-license:
	python tests/lint/check_license_header.py $$(find weaver -name "*.py" -not -path "*/__pycache__/*")

format:
	black weaver/ tests/
	isort weaver/ tests/

check: check-format check-license

clean:
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

pre-commit:
	pre-commit install
	@echo "Pre-commit hooks installed!"

ci: lint test
	@echo "CI checks passed!"
