---
name: address-pr-comments
description: Address GitHub PR review comments. Navigate to the correct worktree, make fixes, push updates.
---

# Address PR Comments Workflow (Worktree-Aware)

## Input

PR number or branch name.

## Workflow

### Step 1: Find the Worktree

```bash
# Get PR branch
BRANCH=$(gh pr view <number> --repo nex-agi/weaver --json headRefName -q '.headRefName')

# Find the worktree for this branch
git worktree list | grep "$BRANCH"
cd <worktree-path>
```

### Step 2: Fetch Unresolved Comments

```bash
gh api graphql -f query='
query {
  repository(owner: "nex-agi", name: "weaver") {
    pullRequest(number: <number>) {
      reviewThreads(first: 50) {
        nodes {
          id isResolved
          comments(first: 1) {
            nodes { id databaseId body path line }
          }
        }
      }
    }
  }
}'
```

### Step 3: Classify & Address

| Category | Description | Action |
|----------|-------------|--------|
| **A: Actionable** | Code changes required | Make changes |
| **B: Discussable** | May skip if follows rules | Ask user |
| **C: Informational** | Resolve without changes | Acknowledge |

### Step 4: Commit & Push

```bash
git add -A
git commit -m "chore(pr): address review comments for #<number>"
git push origin "$BRANCH"
```

### Step 5: Reply & Resolve Threads

Reply to each comment via gh API, then resolve the thread.
