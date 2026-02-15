---
name: fix-issue
description: Fix a GitHub issue by fetching content, creating a branch, planning the fix, and implementing it. Use when the user asks to fix a specific issue number.
---

# Weaver SDK Issue Fix Workflow

Fetch GitHub issue, create branch, plan, and implement the fix.

## Workflow

1. Check gh CLI authentication
2. Fetch issue content
3. Create issue branch
4. Enter plan mode to design fix
5. Implement the fix
6. Run tests (use `testing` skill)
7. Commit changes (use `git-commit` skill)

## Step 1: Check gh CLI Authentication

```bash
gh auth status
```

**If not authenticated**: Prompt user to run `gh auth login`. Stop here.

## Step 2: Fetch Issue Content

```bash
gh issue view ISSUE_NUMBER --repo nex-agi/weaver
gh issue view ISSUE_NUMBER --repo nex-agi/weaver --json number,title,body,state,labels
```

**If issue is closed**: Ask user if they still want to work on it.

## Step 3: Create Issue Branch

```bash
git checkout main && git pull origin main
BRANCH_NAME="issue-${ISSUE_NUM}-short-description"
git checkout -b "$BRANCH_NAME"
```

## Step 4: Enter Plan Mode

Use `EnterPlanMode` to design the fix. Plan should cover:

- Root cause analysis (for bugs)
- Files that need changes
- Implementation strategy
- Testing approach

## Step 5: Implement the Fix

After plan approval, follow project conventions:

1. Make code changes following `.claude/rules/`
2. Add Apache 2.0 license headers to new files
3. Add/update tests in `tests/`
4. Run `make format` before committing

## Step 6: Run Tests

```text
/testing
```

Fix any failures before committing.

## Step 7: Commit Changes

```text
/git-commit
```

**Commit message format:**
```text
fix(scope): Brief description

Fixes #ISSUE_NUMBER
```

## Step 8: Create PR (Optional)

```text
/github-pr
```

## Checklist

- [ ] gh CLI authenticated
- [ ] Issue content fetched and understood
- [ ] Branch created from latest main
- [ ] Plan created and approved
- [ ] Fix implemented with license headers
- [ ] Tests passing (`make ci`)
- [ ] Changes committed with issue reference
