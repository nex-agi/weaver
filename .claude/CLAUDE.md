# AI Assistant Rules for Weaver SDK

## Project Rules

All rules are in `.claude/rules/`. Read and follow them when:

- Making code changes
- Reviewing code
- Committing changes
- Writing or running tests

## Worktree Workflow

This project uses **git worktrees** for branch isolation. Each issue or feature gets its own worktree so multiple agents can work on the same repo simultaneously.

**Key principle:** Never work directly in the main clone for feature branches. Use `fix-issue` skill to create worktrees automatically.

```bash
# Check if you're in a worktree
git rev-parse --show-toplevel
git worktree list
```

## Skills and Agents

### Skills (`.claude/skills/`)

Workflow guides for the main assistant:

- **`fix-issue`** - End-to-end GitHub issue fix with worktree isolation and issue status tracking
- **`git-commit`** - Commit workflow with review and testing (worktree-aware)
- **`github-pr`** - Create GitHub PRs with issue status updates
- **`code-review`** - Invokes code review agent
- **`testing`** - Invokes testing agent
- **`address-pr-comments`** - Address PR review feedback (finds correct worktree)

### Agents (`.claude/agents/`)

Specialized subprocesses that run autonomously:

- **`code-reviewer`** - Reviews code changes against project standards
- **`testing`** - Runs linting and test suite

Code review and testing agents can run **in parallel** during commit workflows.

## GitHub Issue Status Labels

Issues are tracked with status labels throughout the workflow:

| Label | Meaning |
|-------|---------|
| `status:in-progress` | Agent is actively working on the issue |
| `status:blocked` | Work is blocked, needs guidance |
| `status:review` | PR created, awaiting review |
