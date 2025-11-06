PYTHON ?= python
PIP ?= $(PYTHON) -m pip
SRC_DIR := src

INPUT ?= data/raw/DoOR.data-2.0.0/data
OUTPUT ?= door_cache

.PHONY: help install install-dev extract validate lint format type-check test clean

help:
	@echo "Usage: make <target>"
	@echo
	@echo "Targets:"
	@echo "  install        Install runtime dependencies from requirements.txt"
	@echo "  install-dev    Install runtime + developer dependencies (pip install -e .[dev])"
	@echo "  extract        Run DoOR extraction (override INPUT=... OUTPUT=...)"
	@echo "  validate       Validate an existing cache directory (override CACHE=...)"
	@echo "  lint           Run flake8 against src/"
	@echo "  format         Check formatting with black"
	@echo "  type-check     Run mypy against src/"
	@echo "  test           Execute pytest suite"
	@echo "  clean          Remove build artifacts and coverage caches"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install -e .[dev]

extract:
	PYTHONPATH=$(SRC_DIR) $(PYTHON) -m door_toolkit.cli --input $(INPUT) --output $(OUTPUT)

CACHE ?= $(OUTPUT)

validate:
	./scripts/validate-cache $(CACHE)

lint:
	flake8 src/door_toolkit --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 src/door_toolkit --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	black --check src/door_toolkit

type-check:
	mypy src/door_toolkit --ignore-missing-imports

test:
	PYTHONPATH=$(SRC_DIR) pytest tests -v

clean:
	rm -rf dist build *.egg-info .pytest_cache .mypy_cache coverage.xml htmlcov
