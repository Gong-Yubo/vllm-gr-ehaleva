# Pre-commit Setup Guide

This project uses [pre-commit](https://pre-commit.com/) to ensure code quality and consistency.

## Quick Start

   1. **Install pre-commit**:

      ```bash
      pip install pre-commit
      ```

   2. **Install git hooks** (one-time setup):

      ```bash
      pre-commit install
      ```

   3. **Run manually** (optional):

      ```bash
      pre-commit run --all-files
      ```

That's it! Pre-commit will now run automatically on every `git commit`.

## What Pre-commit Does

The following checks run automatically:

- **YAML/JSON/TOML validation**: Ensures config files are valid
- **Trailing whitespace**: Removes trailing whitespace
- **End of file fixer**: Ensures files end with a newline
- **Mixed line endings**: Normalizes line endings to LF
- **Debug statements**: Prevents accidental `pdb`/`ipdb` commits
- **Ruff**: Python linting and formatting
- **Typos**: Spell checking
- **Actionlint**: GitHub Actions workflow validation
- **Pickle check**: Prevents insecure pickle/cloudpickle imports
- **Sign-off**: Adds DCO sign-off to commits

## Manual Usage

### Run on all files

```bash
pre-commit run --all-files
```

### Run on staged files only

```bash
pre-commit run
```

### Run a specific hook

```bash
pre-commit run ruff-check --all-files
pre-commit run check-pickle-imports --all-files
```

### Update hooks

When `.pre-commit-config.yaml` changes, update hooks:

```bash
pre-commit autoupdate
```

## Bypassing Hooks

### Skip all hooks

```bash
git commit --no-verify
```

### Skip a specific hook

```bash
SKIP=check-pickle-imports git commit
SKIP=ruff-check git commit
```

## Troubleshooting

### Hooks not running

- Make sure you ran `pre-commit install`
- Check that `.git/hooks/pre-commit` exists

### Hook fails but you want to commit anyway

- Fix the issues manually, or
- Use `git commit --no-verify` (not recommended)

### Update hook versions

```bash
pre-commit autoupdate
```

### Clear cache and reinstall

```bash
pre-commit clean
pre-commit install
```

## CI Integration

Pre-commit also runs in CI (GitHub Actions). The same checks that run locally will run in CI, ensuring consistency.
