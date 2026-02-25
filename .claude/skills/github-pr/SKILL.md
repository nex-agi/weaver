---
name: github-pr
description: Create a GitHub PR from a worktree branch. Use after committing changes.
---

# GitHub PR Workflow (Worktree-Aware)

## Prerequisites

Changes committed via `git-commit` skill.

## Workflow

### Step 1: Verify Current Location

```bash
# Ensure we're in a worktree, not the main clone
git worktree list
pwd
BRANCH_NAME=$(git branch --show-current)
echo "Current branch: $BRANCH_NAME"
```

### Step 2: Check for Existing PR

```bash
gh pr list --head "$BRANCH_NAME" --state open --repo nex-agi/weaver
```

If PR exists, show it with `gh pr view` and exit.

### Step 3: Push Branch

```bash
git push --set-upstream origin "$BRANCH_NAME"
# After rebase: git push --force-with-lease origin "$BRANCH_NAME"
```

### Step 4: Create PR

```bash
gh pr create \
  --repo nex-agi/weaver \
  --base main \
  --head "$BRANCH_NAME" \
  --title "type(scope): description" \
  --body "## Summary
- Key change 1
- Key change 2

## Testing
- [ ] Tests pass (`make test`)
- [ ] Linting clean (`make lint`)
- [ ] License headers present

## Related Issues
Fixes #ISSUE_NUMBER"
```

Auto-extract title/body from commit messages. No AI branding.

### Step 5: Update Issue

```bash
# Extract issue number from branch name if possible
ISSUE_NUM=$(echo "$BRANCH_NAME" | grep -oP 'issue-\K\d+')
if [ -n "$ISSUE_NUM" ]; then
  gh issue edit "$ISSUE_NUM" --repo nex-agi/weaver \
    --remove-label "status:in-progress" --add-label "status:review" 2>/dev/null || true
  gh issue comment "$ISSUE_NUM" --repo nex-agi/weaver \
    --body "✅ PR created: $(gh pr view --json url -q .url). Ready for review."
fi
```
