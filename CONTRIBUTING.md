# Contributing Guide

## Workflow

1. Create a branch from main:
   git checkout -b feat/short-description
2. Keep pull requests focused on one feature or fix.
3. Add a clear description of what changed and why.

## Coding conventions

- Python 3.11+
- Keep functions small and with explicit names.
- Avoid changing unrelated files in the same pull request.
- Prefer deterministic behavior for emergency workflow logic.

## Testing

- Run smoke checks for both Streamlit apps before opening PR.
- Run script-level checks for ML pipeline changes.
- If no automated test exists, include manual test steps in the PR.

## Commit style (recommended)

Use conventional prefixes:
- feat: new capability
- fix: bug fix
- docs: documentation only
- chore: maintenance work

Example:
feat: add operator/conductor docker compose setup
