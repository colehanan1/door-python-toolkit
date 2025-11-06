# Changelog

All notable changes to this project will be documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- _Placeholder_ – add new entries here.

### Changed
- _Placeholder_ – record behaviour changes here.

### Fixed
- _Placeholder_ – document bug fixes here.

## [0.2.0] - 2025-11-06

### Added
- Multi-odor CLI workflow via `door-extract --odors` for side-by-side receptor comparisons, including automatic spread ranking.
- Receptor group shortcuts (`--receptor or|ir|gr|neuron`) with tail summaries that highlight the lowest responding odorants alongside the top hits.
- CSV export support (`--save`) for receptor and odor comparison tables, writing dash-separated headers for easy downstream processing.
- Coverage output now reports both the strongest and weakest receptors to speed up exploratory analysis.

### Changed
- README instructions updated with multi-odor and receptor-tail examples, plus clarified debugging guidance.

### Fixed
- Normalised cache indices to use `InChIKey`, preventing lookup errors when encoding odors from extracted datasets.
- Coerced response matrices to numeric dtype to keep coverage statistics and ranking functions stable.

## [0.1.0] - 2025-11-06

### Added
- Initial public release of the DoOR Python Toolkit.
- `DoORExtractor` for converting DoOR R package assets into Python-friendly formats.
- `DoOREncoder` for encoding odorant names into projection neuron activation patterns.
- Utilities for listing odorants, loading response matrices, exporting subsets, and validating caches.
- Command-line interface (`door-extract`) for extraction, validation, and cache inspection.
- Optional PyTorch integration and accompanying unit tests.
- Continuous integration workflows, documentation scaffolding, and example notebooks.

### Changed
- Not applicable (initial release).

### Fixed
- Not applicable (initial release).

## Future versions

Upcoming releases will continue to expand the toolkit (e.g., receptor selection strategies, similarity search improvements, and data import helpers). Contributions are welcome—see `CONTRIBUTING.md`.

---

Release links:

- [Unreleased](https://github.com/colehanan1/door-python-toolkit/compare/v0.2.0...HEAD)
- [0.2.0](https://github.com/colehanan1/door-python-toolkit/compare/v0.1.0...v0.2.0)
- [0.1.0](https://github.com/colehanan1/door-python-toolkit/releases/tag/v0.1.0)
