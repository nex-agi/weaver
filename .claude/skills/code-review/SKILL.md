---
name: code-review
description: Review code changes against Weaver SDK project standards before committing. Use when reviewing code or preparing commits.
---

# Weaver SDK Code Review Skill

## How to Use

1. Read agent instructions at `.claude/agents/code-review/AGENT.md`
2. Invoke Task tool with `subagent_type="code-review"` (specialized agent)
3. Agent reviews all changes against project standards

## Review Process

1. **Get changes**: Run `git diff` to see all staged and unstaged changes
2. **Analyze each file** against the review checklist
3. **Report findings**: Provide clear, actionable feedback

## Review Checklist

### 1. License Headers

- [ ] Apache 2.0 header present in all new/modified `.py` files
- [ ] Header matches exact format from `CONTRIBUTING.md`

### 2. Python Code Quality

- [ ] Type hints on all public API parameters and return types
- [ ] Modern syntax: `list[int]`, `X | None` (not `List`, `Optional`)
- [ ] f-strings for formatting (no `.format()` or `%`)
- [ ] Google-style docstrings on public APIs
- [ ] No debug code (`print()`, commented sections)

### 3. Formatting

- [ ] Black-formatted (100 char line length)
- [ ] isort-compliant import ordering
- [ ] Proper import grouping (stdlib, third-party, local)

### 4. Error Handling

- [ ] Custom exceptions used (`WeaverAPIError`, `ValueError`)
- [ ] Error messages include context (what was received vs expected)
- [ ] No bare `except:` clauses

### 5. Commit Content

- [ ] Only relevant changes included
- [ ] No build artifacts (`dist/`, `*.egg-info`)
- [ ] No sensitive information (tokens, API keys)
- [ ] No temporary test files

## Common Issues to Flag

- **Missing license header** in new files
- **Legacy type syntax**: `List[int]` instead of `list[int]`
- **Missing type hints** on public methods
- **Bare print()** for debugging
- **Hardcoded URLs or credentials**

## Output Format

```
## Code Review Summary
**Status:** PASS / WARNINGS / FAIL

### Issues Found
[List issues by category]

### Recommendations
[Specific actions to fix issues]

### Approved Items
[What looks good]
```

## Related Skills

- **`testing`** - Lint and test verification (runs in parallel)
- **`git-commit`** - Complete commit workflow
