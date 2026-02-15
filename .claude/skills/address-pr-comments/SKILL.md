---
name: address-pr-comments
description: Analyze and address GitHub PR review comments intelligently. Use when addressing PR feedback or review comments.
---

# Address PR Comments Workflow

Triage PR review comments, address actionable feedback, and resolve informational comments.

## Input

Accept PR number (`123`, `#123`) or branch name.

## Workflow

1. Match input to PR
2. Fetch unresolved comments
3. Classify comments
4. Get user confirmation (Category B)
5. Address comments with code changes
6. Reply and resolve threads

## Step 1: Match Input to PR

```bash
gh pr view <number> --repo nex-agi/weaver --json number,title,headRefName,state
```

## Step 2: Fetch Unresolved Comments

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

## Step 3: Classify Comments

| Category | Description | Examples |
|----------|-------------|----------|
| **A: Actionable** | Code changes required | Bugs, missing validation |
| **B: Discussable** | May skip if follows rules | Style preferences |
| **C: Informational** | Resolve without changes | Acknowledgments |

## Step 4: Get User Confirmation

Use `AskUserQuestion` for Category B: Address / Skip / Discuss

## Step 5: Address Comments

1. Read files, make changes
2. Ensure license headers preserved
3. Commit using `/git-commit`

## Step 6: Resolve Comments

Reply then resolve thread with GraphQL mutation.

## Checklist

- [ ] PR matched and validated
- [ ] Comments classified
- [ ] Category B reviewed with user
- [ ] Changes made and committed
- [ ] All comments resolved
