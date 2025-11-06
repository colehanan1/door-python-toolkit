# Contributing to DoOR Python Toolkit

Thank you for considering contributing to the DoOR Python Toolkit! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful, inclusive, and constructive. We're all here to advance Drosophila neuroscience research.

## How to Contribute

### Reporting Bugs

1. Check [existing issues](https://github.com/colehanan1/door-python-toolkit/issues) first
2. Create a new issue with:
   - Clear description of the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version and OS
   - Error messages/stack traces

### Suggesting Features

1. Open an issue with `[Feature Request]` prefix
2. Describe the use case and benefits
3. Provide example code if possible

### Contributing Code

#### Setup Development Environment

```bash
# Fork and clone
git clone https://github.com/colehanan1/door-python-toolkit.git
cd door-python-toolkit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev]

# Run tests
pytest tests/
```

#### Development Workflow

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes**
   - Write clear, documented code
   - Follow PEP 8 style guidelines
   - Add type hints
   - Include docstrings (Google/NumPy style)

3. **Test your changes**
   ```bash
   # Run tests
   pytest tests/ -v
   
   # Check coverage
   pytest --cov=door_toolkit tests/
   
   # Format code
   black door_toolkit/
   
   # Lint
   flake8 door_toolkit/
   
   # Type check
   mypy door_toolkit/
   ```

4. **Commit changes**
   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   ```
   
   Use conventional commit messages:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `test:` Test additions/changes
   - `refactor:` Code refactoring
   - `style:` Formatting changes
   - `chore:` Maintenance tasks

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   
   Then open a Pull Request on GitHub with:
   - Clear title and description
   - Reference related issues
   - Include tests for new features
   - Update documentation as needed

#### Code Style

- **Formatting**: Use `black` for automatic formatting
- **Line length**: 88 characters (black default)
- **Imports**: Sort with `isort`
- **Docstrings**: Google or NumPy style
- **Type hints**: Use where appropriate

Example:
```python
from typing import List, Optional
import numpy as np

def encode_odorants(
    odor_names: List[str],
    fill_missing: float = 0.0
) -> np.ndarray:
    """
    Encode odorant names to activation vectors.
    
    Args:
        odor_names: List of odorant names
        fill_missing: Value for missing responses (default: 0.0)
        
    Returns:
        NumPy array of shape (n_odors, n_receptors)
        
    Raises:
        KeyError: If odorant not found in database
        
    Example:
        >>> activations = encode_odorants(["acetic acid", "ethanol"])
        >>> print(activations.shape)
        (2, 78)
    """
    # Implementation
    pass
```

#### Testing Guidelines

- Write tests for all new features
- Aim for >80% code coverage
- Use descriptive test names
- Include edge cases
- Use pytest fixtures for setup

Example:
```python
import pytest
from door_toolkit import DoOREncoder

class TestDoOREncoder:
    @pytest.fixture
    def encoder(self):
        return DoOREncoder("door_cache")
    
    def test_encode_single_odorant(self, encoder):
        """Test encoding a single odorant."""
        result = encoder.encode("acetic acid")
        assert result.shape == (78,)
        assert result.dtype == np.float32
```

#### Documentation

- Update README.md for user-facing changes
- Add docstrings to all public functions/classes
- Include examples in docstrings
- Update API reference if needed

### Pull Request Process

1. **Before submitting:**
   - All tests pass
   - Code is formatted (`black`)
   - No linting errors (`flake8`)
   - Documentation is updated
   - CHANGELOG.md is updated

2. **PR checklist:**
   - [ ] Tests added/updated
   - [ ] Documentation updated
   - [ ] CHANGELOG.md updated
   - [ ] Code formatted with black
   - [ ] No flake8 errors
   - [ ] Type hints added
   - [ ] All CI checks pass

3. **Review process:**
   - Maintainers will review your PR
   - Address feedback promptly
   - Keep PR focused and small
   - Be patient and respectful

## Release Process

(For maintainers)

1. Update version in `setup.py` and `__init__.py`
2. Update CHANGELOG.md
3. Create git tag: `git tag v0.1.0`
4. Push tag: `git push origin v0.1.0`
5. Build and publish: `python -m build && twine upload dist/*`

## Questions?

- Open an issue for questions
- Check [documentation](https://door-python-toolkit.readthedocs.io)
- Email: c.b.hanan@wustl.edu

## Recognition

Contributors will be acknowledged in:
- README.md contributors section
- CHANGELOG.md for their contributions
- GitHub contributors page

Thank you for contributing to Drosophila neuroscience research! 🧬🐝
