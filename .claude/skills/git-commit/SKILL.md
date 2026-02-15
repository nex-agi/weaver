---
name: git-commit
description: Complete git commit workflow for Weaver SDK including pre-commit review, staging, message generation, and verification. Use when creating commits or preparing changes for commit.
---

# Weaver SDK Git Commit Workflow

## Prerequisites

**Check what changed to determine which agents to run:**

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

**Launch appropriate agents IN PARALLEL:**

- **`code-reviewer`** - ALWAYS run for all changes
- **`testing`** - ONLY run if `.py` files changed

## Workflow

1. Analyze changed files to determine testing needs
2. Launch code-review (always) and testing (if needed) in parallel
3. Wait for agents to complete
4. Address any issues found
5. Stage changes
6. Generate commit message
7. Commit and verify

## Stage Changes

```bash
git add path/to/file1.py path/to/file2.py
git diff --staged  # Review
```

**Never stage**: Build artifacts (`dist/`, `*.egg-info`), `.env`, `__pycache__/`

## Commit Message Format

**Structure**: `type(scope): description` (72 chars max)

**Types**: feat, fix, refactor, test, docs, style, chore, perf
**Scope**: Module/component (client, types, cli, http, sampling)
**Description**: Present tense, action verb, no period

**Good examples:**
```text
feat(client): add async export-sampler support
fix(http): handle connection timeout gracefully
test(types): add model input serialization tests
docs(readme): update installation instructions
```

## Co-Author Policy

**NEVER add AI co-author lines.** Commits reflect human authorship only.

## Post-Commit Verification

```bash
git show HEAD              # View commit
git log -1                 # Check message
git show HEAD --name-only  # Verify files
```

## Checklist

- [ ] Changed files analyzed
- [ ] Code review completed (license headers checked)
- [ ] Tests passed (if code changed)
- [ ] Only relevant files staged
- [ ] No build artifacts or secrets
- [ ] Message format: `type(scope): description` (72 chars, present tense)
- [ ] No AI co-authors
