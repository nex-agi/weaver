---
name: github-pr
description: Create a GitHub pull request after committing and pushing changes. Use when the user asks to create a PR or submit changes for review.
---

# Weaver SDK GitHub Pull Request Workflow

## Prerequisites

Run `/git-commit` skill first to commit all changes.

## Workflow Steps

1. Check for existing PR (exit if found)
2. Push to remote
3. Create PR using gh CLI

## Step 1: Check for Existing PR

```bash
BRANCH_NAME=$(git branch --show-current)
gh pr list --head "$BRANCH_NAME" --state open --repo nex-agi/weaver
```

**If PR exists**: Display with `gh pr view` and exit immediately.

## Step 2: Push

```bash
git push --set-upstream origin BRANCH_NAME
```

## Step 3: Create PR

```bash
gh pr create \
  --repo nex-agi/weaver \
  --title "Brief description of changes" \
  --body "$(cat <<'EOF'
## Summary
- Key change 1
- Key change 2

## Testing
- [ ] All tests pass (`make test`)
- [ ] Linting clean (`make lint`)
- [ ] License headers present
- [ ] Code review completed

## Related Issues
Fixes #ISSUE_NUMBER (if applicable)
EOF
)"
```

**Important**: Do NOT add AI branding footers.

## Common Issues

| Issue | Solution |
|-------|----------|
| PR already exists | `gh pr view` then exit |
| Push rejected | `git push --force-with-lease` |
| gh not authenticated | Tell user to run `gh auth login` |

## Checklist

- [ ] No existing PR for branch
- [ ] Changes committed via git-commit
- [ ] Pushed to remote
- [ ] PR created with clear title/body
