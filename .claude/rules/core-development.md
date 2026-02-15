# Core Development Rules

## 1. Python Standards

**Python >= 3.10. Line length: 100 characters.**

### Formatting & Linting

- **Black** + **isort** for formatting (`make format`)
- **pylint** for linting (only `weaver/` directory)
- **mypy** for type checking
- Fix linter errors, don't suppress them

### Type Hints

- Type hints on all public API parameters and return types
- Use modern syntax: `list[int]`, `dict[str, Any]`, `X | None`
- Use `@overload` for distinct call signatures

### Docstrings

- Google-style docstrings with Args/Returns/Raises
- Required for all public APIs

### Naming

| Element | Convention | Example |
|---------|-----------|---------|
| Classes | PascalCase | `ServiceClient`, `WeaverConfig` |
| Functions | snake_case | `create_session()` |
| Constants | UPPER_SNAKE_CASE | `DEFAULT_TIMEOUT` |
| Private | Leading underscore | `_http.py`, `_utils.py` |

### Imports

Order: stdlib, third-party, local. Use relative imports within package.

```python
import os
from typing import Any

import httpx
from pydantic import BaseModel

from .config import WeaverConfig
```

## 2. License Headers

**All source files MUST include the Apache 2.0 header:**

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

A pre-commit hook enforces this. Run `python tests/lint/check_license_header.py` to verify.

## 3. Project Patterns

### Architecture

- `weaver/service_client.py` - Main client (sessions, models, training)
- `weaver/training_client.py` - Training operations
- `weaver/sampling_client.py` - Inference/sampling
- `weaver/_http.py` - HTTP client wrapper (httpx)
- `weaver/types/` - Pydantic models and type definitions
- `weaver/cli.py` - Click CLI interface

### Key Libraries

| Library | Usage |
|---------|-------|
| httpx | HTTP client |
| pydantic | Data models and validation |
| torch | Tensor operations |
| transformers | Tokenizers and model loading |
| click | CLI framework |
| rich | Terminal formatting |

### Error Handling

Use `WeaverAPIError` for API errors. Include context in messages:

```python
raise ValueError(f"Invalid shape {shape}: expected {expected}D, got {len(shape)}D")
```

## 4. Security & Co-Author Policy

- No hardcoded secrets. Use `os.getenv()` for API keys.
- Never log sensitive data.
- **NEVER add AI co-author lines** to commits or PRs.

## Quick Checklist

- [ ] License header present in new files
- [ ] Type hints on public APIs
- [ ] `make format` applied
- [ ] `make lint` passes
- [ ] Tests in `tests/` directory
- [ ] No hardcoded secrets
- [ ] No AI co-author lines
