#!/usr/bin/env bash
# Publish the DoOR Python Toolkit to PyPI.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"

cd "$ROOT"

if [[ ! -f "pyproject.toml" ]]; then
  echo "Error: run this script from the repository root." >&2
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: Python binary '$PYTHON_BIN' not found." >&2
  exit 1
fi

if ! "$PYTHON_BIN" -m build --version >/dev/null 2>&1; then
  echo "Error: python-build is required. Install it with 'pip install build'." >&2
  exit 1
fi

if ! "$PYTHON_BIN" -m twine --version >/dev/null 2>&1; then
  echo "Error: twine is required. Install it with 'pip install twine'." >&2
  exit 1
fi

if [[ -z "${PYPI_TOKEN:-}" && -z "${TWINE_PASSWORD:-}" ]]; then
  cat <<'EOF' >&2
Error: missing credentials.
Set PYPI_TOKEN (recommended) or TWINE_PASSWORD/TWINE_USERNAME before running this script.
For API tokens, export:
  export PYPI_TOKEN=pypi-xxxxxxxxxxxx
EOF
  exit 1
fi

# Ensure version consistency between package metadata and source.
VERSION_CHECK="$("$PYTHON_BIN" <<'PY'
from pathlib import Path
import re
import sys

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - fallback for 3.8-3.10
    try:
        import tomli as tomllib  # type: ignore[assignment]
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency missing
        raise SystemExit(
            "Missing dependency: install the 'tomli' package for Python < 3.11."
        ) from exc

root = Path(".")

with (root / "pyproject.toml").open("rb") as fh:
    pyproject_version = tomllib.load(fh)["project"]["version"]

with (root / "src" / "door_toolkit" / "__init__.py").open("r", encoding="utf-8") as fh:
    match = re.search(r'__version__\s*=\s*"([^"]+)"', fh.read())
    init_version = match.group(1) if match else None

if pyproject_version != init_version:
    print(
        f"Version mismatch: pyproject.toml={pyproject_version}, "
        f"src/door_toolkit/__init__.py={init_version}",
        file=sys.stderr,
    )
    sys.exit(1)

print(pyproject_version)
PY
)" || {
  echo "Version mismatch detected. Aborting." >&2
  exit 1
}

echo "Preparing release for version: $VERSION_CHECK"

echo "Cleaning previous build artifacts..."
rm -rf build dist *.egg-info

echo "Building source distribution and wheel..."
"$PYTHON_BIN" -m build

echo "Validating distribution files with twine..."
"$PYTHON_BIN" -m twine check dist/*

if [[ -n "${PYPI_TOKEN:-}" ]]; then
  export TWINE_USERNAME="__token__"
  export TWINE_PASSWORD="${PYPI_TOKEN}"
fi

echo "Uploading packages to PyPI..."
"$PYTHON_BIN" -m twine upload dist/*

echo "Publish complete. Remember to push tags and update release notes if needed."
