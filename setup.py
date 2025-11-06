"""Setup configuration for door-python-toolkit."""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="door-python-toolkit",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Python toolkit for working with the DoOR (Database of Odorant Responses) database",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/door-python-toolkit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="drosophila olfaction neuroscience door odorant receptor",
    python_requires=">=3.8",
    install_requires=[
        "pyreadr>=0.4.7",
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "pyarrow>=12.0.0",
    ],
    extras_require={
        "torch": ["torch>=2.0.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "door-extract=door_toolkit.cli:extract_main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/door-python-toolkit/issues",
        "Source": "https://github.com/yourusername/door-python-toolkit",
        "Documentation": "https://door-python-toolkit.readthedocs.io",
    },
)
