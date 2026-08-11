# Development

The project supports Python 3.10 and newer. Create and activate a virtual
environment, then install the runtime and development dependencies together:

```bash
python3 -m venv .venv
source .venv/bin/activate
make install-dev
```

Run the complete local quality gate with:

```bash
make check
```

The individual commands are `make lint`, `make typecheck`, and `make test`.
Existing type-checking migration warnings remain visible but do not fail the
gate; new error-level diagnostics do.

Ruff is also the Python formatter. Format the repository and sort imports with:

```bash
make format
```

Check formatting without changing files with `make format-check`. Formatting is
not yet part of the repository-wide CI gate because the existing code has not
been migrated to Ruff formatting.

To run Ruff automatically on changed Python files before each commit:

```bash
pre-commit install
```
