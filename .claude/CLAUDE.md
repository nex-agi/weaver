# AI Assistant Rules for Weaver SDK

Please follow these rules when working on the Weaver SDK project:

## Project Rules

All rules are in `.claude/rules/`. Read and follow them when:

- Making code changes
- Reviewing code
- Committing changes
- Writing or running tests

## Skills and Agents

### Skills (`.claude/skills/`)

Workflow guides for the main assistant:

- **`git-commit`** - Complete commit workflow with review and testing
- **`code-review`** - Invokes code review agent
- **`testing`** - Invokes testing agent
- **`fix-issue`** - End-to-end GitHub issue fix workflow
- **`github-pr`** - Create GitHub pull requests
- **`address-pr-comments`** - Address PR review feedback

### Agents (`.claude/agents/`)

Specialized subprocesses that run autonomously:

- **`code-reviewer`** - Reviews code changes against project standards
- **`testing`** - Runs linting and test suite

Code review and testing agents can run **in parallel** during commit workflows.
