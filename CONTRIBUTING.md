# Contributing to the DoOR Python Toolkit

Thanks for your interest in contributing! We welcome bug reports, feature ideas, documentation improvements, and code contributions that make this toolkit more useful for the olfactory neuroscience community.

---

## Code of conduct

We expect everyone to foster a respectful, inclusive environment. Be kind, assume good intent, and help newcomers succeed. If you experience or witness unacceptable behaviour, please email [c.b.hanan@wustl.edu](mailto:c.b.hanan@wustl.edu).

---

## Reporting bugs

1. Search existing issues to avoid duplicates.
2. Open a new issue using the **Bug report** template.
3. Include:
   - steps to reproduce,
   - expected vs. actual behaviour,
   - environment details (OS, Python version, package version),
   - relevant logs or stack traces.

Critical regressions should be flagged clearly in the issue title.

---

## Suggesting features

Use the **Feature request** template to share ideas. Tell us:
- what problem you are solving,
- how you envision the solution working,
- any alternative approaches considered.

Community discussion helps refine scope before code changes begin.

---

## Development setup

```bash
git clone https://github.com/colehanan1/door-python-toolkit.git
cd door-python-toolkit
python -m venv .venv
source .venv/bin/activate
make install-dev
```

Helpful Makefile targets:
- `make extract INPUT=... OUTPUT=...` – run the data extractor
- `make validate CACHE=door_cache` – check a cache directory
- `make lint` / `make format` – static analysis and formatting checks
- `make test` – run the full test suite
- `make clean` – remove build and coverage artifacts

When working from the repository without installing the package, the helper scripts in `scripts/` ensure `PYTHONPATH` is configured correctly:
- `./scripts/door-extract`
- `./scripts/validate-cache`

---

## Coding standards

- **Python style:** Black (line length 100) and Flake8 (configured in CI). Run `make format` and `make lint`.
- **Typing:** Add or preserve type hints where practical. `make type-check` runs MyPy with project settings.
- **Imports:** Use absolute imports from `door_toolkit`. Keep standard library, third-party, and local imports grouped.
- **Documentation:** Update docstrings and README sections impacted by your changes. Add examples if they help clarify behaviour.

---

## Testing requirements

- Add or update unit tests in `tests/` to cover new behaviour.
- Run `make test` before submitting a pull request.
- If PyTorch functionality is affected, ensure the optional `test_torch_integration` path is considered (install extras `pip install .[torch]` locally if needed).
- For extractor workflow changes, include fixtures or mocks so tests do not rely on large datasets.

---

## Pull request process

1. Create a working branch (`git switch -c feature/my-improvement`).
2. Make focused commits with descriptive messages.
3. Update documentation, changelog, and version number when appropriate.
4. Ensure CI passes (formatting, linting, typing, tests).
5. Submit a PR and complete the checklist:
   - Linked issue (if applicable)
   - Explanation of changes and testing
   - Screenshots or logs for UI/CLI updates

PR reviews focus on correctness, clarity, and maintainability. Be ready to iterate based on feedback.

---

## Changelog updates

When your change affects users, add an entry under the **Unreleased** section of `CHANGELOG.md` with a short description (e.g. “Fixed cache validation for missing metadata” or “Added support for Python 3.12”).

---

## Release workflow

Maintainers use the scripts in `scripts/` (`bump-version.sh`, `publish-pypi.sh`) to prepare and publish releases. If you are helping with a release:
- Merge all relevant PRs.
- Update version numbers and changelog entries.
- Tag the release and push to GitHub.

---

## Recognition

We credit contributors in the release notes and encourage you to add yourself to the acknowledgments section if your contributions are substantial. Thank you for helping improve the DoOR Python Toolkit!
