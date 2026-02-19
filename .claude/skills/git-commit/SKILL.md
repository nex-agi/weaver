---
name: git-commit
description: Complete git commit workflow in a worktree. Includes review, staging, and message generation.
---

# Git Commit Workflow (Worktree-Aware)

## Prerequisites

Verify you're in a worktree (not main clone):
```bash
# Should show a worktree path, not the main repo
git rev-parse --show-toplevel
git worktree list
```

## Workflow

### Step 1: Analyze Changes

```bash
git diff --name-only
git diff --cached --name-only
```

**Determine testing needs:**

| File Types Changed | Run Code Review | Run Testing |
|--------------------|-----------------|-------------|
| Code (`.py`, tests) | Yes | Yes |
| Docs only (`.md`) | Yes | Skip |
| Config (`.yaml`, `.toml`, `.github/`) | Yes | Skip |
| Mixed (code + docs/config) | Yes | Yes |

### Step 2: Run Review & Tests

Launch code-review and testing as appropriate (**in parallel** when possible).

### Step 3: Address Issues

Fix any problems found by review or testing before proceeding.

### Step 4: Stage Changes

```bash
git add path/to/changed/files
git diff --staged  # Review
```

**Never stage:** Build artifacts (`dist/`, `*.egg-info`), `.env`, `__pycache__/`, coverage files

### Step 5: Commit

Format: `type(scope): description` (72 chars max)

**Types**: feat, fix, refactor, test, docs, style, chore, perf
**Scope**: Module/component (client, types, cli, http, sampling)

```bash
git commit -m "type(scope): description

Detailed explanation if needed.

Fixes #ISSUE_NUMBER"
```

**Good examples:**
```text
feat(client): add async export-sampler support
fix(http): handle connection timeout gracefully
test(types): add model input serialization tests
```

**No AI co-author lines.**

### Step 6: Verify

```bash
git show HEAD --name-only
git log -1
```
