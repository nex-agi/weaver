---
name: fix-issue
description: Fix a GitHub issue using git worktree for isolation. Fetches issue, creates worktree, plans and implements fix, then creates PR.
---

# Fix Issue Workflow (Worktree-Based)

## Overview

Each issue is worked on in an isolated git worktree. This allows multiple agents to work on the same repo simultaneously without conflicts.

## Workflow

### Step 1: Verify Prerequisites

```bash
gh auth status
```

### Step 2: Fetch Issue

```bash
ISSUE_NUM=<number>
gh issue view $ISSUE_NUM --repo nex-agi/weaver --json number,title,body,state,labels
```

If closed, confirm with user before proceeding.

### Step 3: Update Issue Status

```bash
gh issue comment $ISSUE_NUM --repo nex-agi/weaver --body "🤖 Agent picking up this issue. Creating worktree branch \`agent/issue-${ISSUE_NUM}\`."

# Add in-progress label (create if doesn't exist)
gh issue edit $ISSUE_NUM --repo nex-agi/weaver --add-label "status:in-progress" 2>/dev/null || true
```

### Step 4: Fetch Shared Knowledge

Clone or update the shared knowledge repo for common scripts and docs.

```bash
REPO_ROOT=$(git rev-parse --show-toplevel)
SHARED_KNOWLEDGE="${REPO_ROOT}/../nex-taas-shared-knowledge"
if [ ! -d "$SHARED_KNOWLEDGE" ]; then
  git clone https://github.com/china-qijizhifeng/nex-taas-shared-knowledge.git "$SHARED_KNOWLEDGE"
else
  git -C "$SHARED_KNOWLEDGE" pull --ff-only 2>/dev/null || true
fi
```

### Step 5: Create Worktree

```bash
REPO_ROOT=$(git rev-parse --show-toplevel)
git fetch origin

BRANCH="agent/issue-${ISSUE_NUM}"
WORKTREE_DIR="${REPO_ROOT}/../worktrees/weaver-issue-${ISSUE_NUM}"

# Create worktree with new branch from latest main
git worktree add "$WORKTREE_DIR" -b "$BRANCH" "origin/main"

cd "$WORKTREE_DIR"
```

### Step 6: Plan the Fix

Enter plan mode. Consider:
- Root cause analysis (for bugs)
- Files that need changes
- Implementation strategy
- Testing approach
- Impact on other components

### Step 7: Implement

Work in the worktree directory. Follow project rules in `.claude/rules/`:
- Apache 2.0 license headers on all new `.py` files
- Type hints on public APIs
- Google-style docstrings
- Run `make format` before committing

### Step 8: Test

Run project tests (use `testing` skill):
```bash
make ci  # lint + test
```

### Step 9: Commit and PR

Use `git-commit` skill, then `github-pr` skill.

The PR should reference the issue: `Fixes #ISSUE_NUM`

### Step 10: Handle Blockers

If blocked, update the issue:

```bash
gh issue edit $ISSUE_NUM --repo nex-agi/weaver \
  --remove-label "status:in-progress" --add-label "status:blocked"

gh issue comment $ISSUE_NUM --repo nex-agi/weaver --body "
🚧 Blocked: <describe the problem>

Need guidance on: <specific question>

Progress so far: <what's been done>
"
```

### Step 11: Cleanup (after PR merged)

```bash
cd "$REPO_ROOT"
git worktree remove "$WORKTREE_DIR"
git branch -d "$BRANCH"
```
