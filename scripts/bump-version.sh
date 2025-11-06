#!/usr/bin/env bash
# Bump the package version (major, minor, or patch) and create a release commit/tag.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
cd "$ROOT"

usage() {
  echo "Usage: $(basename "$0") <major|minor|patch> [--no-tag]" >&2
  exit 1
}

if [[ $# -lt 1 ]]; then
  usage
fi

BUMP_TYPE="$1"
shift

case "$BUMP_TYPE" in
  major|minor|patch) ;;
  *) usage ;;
esac

CREATE_TAG=true
if [[ "${1:-}" == "--no-tag" ]]; then
  CREATE_TAG=false
  shift
fi

if [[ $# -gt 0 ]]; then
  usage
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Error: working tree is not clean. Commit or stash changes before bumping the version." >&2
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: Python binary '$PYTHON_BIN' not found." >&2
  exit 1
fi

NEW_VERSION="$("$PYTHON_BIN" <<'PY'
from pathlib import Path
import re
import sys

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - fallback for 3.8-3.10
    try:
        import tomli as tomllib  # type: ignore[assignment]
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency missing
        raise SystemExit("Missing dependency: install the 'tomli' package for Python < 3.11.") from exc

bump_type = sys.argv[1]
root = Path(".")
pyproject_path = root / "pyproject.toml"
setup_path = root / "setup.py"
init_path = root / "src" / "door_toolkit" / "__init__.py"

with pyproject_path.open("rb") as fh:
    project_data = tomllib.load(fh)

current_version = project_data["project"]["version"]

try:
    major, minor, patch = map(int, current_version.split("."))
except ValueError as exc:  # pragma: no cover - defensive
    raise SystemExit(f"Invalid semantic version: {current_version}") from exc

if bump_type == "major":
    major += 1
    minor = 0
    patch = 0
elif bump_type == "minor":
    minor += 1
    patch = 0
else:  # patch
    patch += 1

new_version = f"{major}.{minor}.{patch}"

replacements = [
    (pyproject_path, r'version\s*=\s*"[0-9]+\.[0-9]+\.[0-9]+"', f'version = "{new_version}"'),
    (setup_path, r'version="\d+\.\d+\.\d+"', f'version="{new_version}"'),
    (init_path, r'__version__\s*=\s*"[^"]+"', f'__version__ = "{new_version}"'),
]

for path, pattern, replacement in replacements:
    original = path.read_text(encoding="utf-8")
    updated, count = re.subn(pattern, replacement, original, count=1)
    if count != 1:
        raise SystemExit(f"Failed to update version in {path}")
    path.write_text(updated, encoding="utf-8")

print(new_version)
PY
"$BUMP_TYPE")"

echo "Version bumped to $NEW_VERSION"

git add pyproject.toml setup.py src/door_toolkit/__init__.py
git commit -m "chore: bump version to $NEW_VERSION"

if $CREATE_TAG; then
  git tag -a "v$NEW_VERSION" -m "Release v$NEW_VERSION"
  echo "Created annotated tag v$NEW_VERSION"
else
  echo "Skipping tag creation (--no-tag supplied)"
fi

echo "Done. Push with: git push && git push --tags"
