---
name: code-review
description: Review code changes against Weaver SDK project standards before committing. Works in any worktree.
---

# Weaver SDK Code Review Skill (Worktree-Aware)

## How to Use

1. Read agent instructions at `.claude/agents/code-review/AGENT.md`
2. Invoke Task tool with `subagent_type="code-review"` (specialized agent)
3. Agent reviews all changes against project standards

## Prerequisites

Verify you're in the correct worktree:
```bash
git rev-parse --show-toplevel
git branch --show-current
```

## Review Checklist

### 1. License Headers

- [ ] Apache 2.0 header present in all new/modified `.py` files
- [ ] Header matches exact format from `.claude/rules/core-development.md`

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
- [ ] Error messages include context
- [ ] No bare `except:` clauses

### 5. Commit Content

- [ ] Only relevant changes included
- [ ] No build artifacts (`dist/`, `*.egg-info`)
- [ ] No sensitive information (tokens, API keys)

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
