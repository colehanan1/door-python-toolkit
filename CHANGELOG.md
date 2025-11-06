# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Advanced receptor selection strategies
- Chemical similarity predictions for missing responses
- Integration with FlyWire connectome data
- Web API for DoOR queries
- Pre-trained embeddings

## [0.1.0] - 2025-11-06

### Added
- Initial release
- `DoORExtractor` for extracting R data to Python formats
- `DoOREncoder` for encoding odorant names to neural activations
- Utility functions for searching, filtering, and analyzing data
- PyTorch integration support
- Command-line interface (`door-extract`)
- Comprehensive test suite
- Examples and documentation
- GitHub Actions CI/CD

### Features
- Extract 693 odorants × 78 receptors from DoOR v2.0
- Pure Python implementation (no R required)
- Parquet-based caching for fast loading
- Name-based odorant lookup with case-insensitive search
- Batch encoding support
- Receptor coverage statistics
- Chemical metadata access
- Similar odorant search by response pattern

### Documentation
- README with quick start guide
- API reference in docstrings
- Usage examples (basic and PyTorch)
- Contributing guidelines

[Unreleased]: https://github.com/yourusername/door-python-toolkit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/door-python-toolkit/releases/tag/v0.1.0
