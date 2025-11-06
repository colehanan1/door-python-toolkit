"""Packaging configuration for the DoOR Python Toolkit."""

from pathlib import Path

from setuptools import find_packages, setup

PROJECT_ROOT = Path(__file__).parent
README = PROJECT_ROOT / "README.md"

setup(
    name="door-python-toolkit",
    version="0.2.0",
    description="Pure Python tooling for working with the DoOR (Database of Odorant Responses) dataset.",
    long_description=README.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Cole Hanan",
    author_email="c.b.hanan@wustl.edu",
    license="MIT",
    url="https://github.com/colehanan1/door-python-toolkit",
    project_urls={
        "Documentation": "https://door-python-toolkit.readthedocs.io",
        "Source": "https://github.com/colehanan1/door-python-toolkit",
        "Issue Tracker": "https://github.com/colehanan1/door-python-toolkit/issues",
        "Changelog": "https://github.com/colehanan1/door-python-toolkit/blob/main/CHANGELOG.md",
        "Lab": "https://ramanlab.wustl.edu/",
    },
    keywords=[
        "drosophila",
        "olfaction",
        "neuroscience",
        "bioinformatics",
        "odorant",
        "receptor",
        "door",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "pyarrow>=12.0.0",
    ],
    extras_require={
        "extract": ["pyreadr>=0.4.7"],
        "torch": ["torch>=2.0.0"],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
        "dev": [
            'tomli>=2.0.1; python_version<"3.11"',
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "twine>=4.0.0",
            "build>=1.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    entry_points={
        "console_scripts": [
            "door-extract=door_toolkit.cli:extract_main",
        ],
    },
)
