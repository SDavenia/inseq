#* Variables
SHELL := /usr/bin/env bash
PYTHON := python3

#* Docker variables
IMAGE := inseq
VERSION := latest

.PHONY: help
help:
	@echo "Commands:"
	@echo "poetry-download    : downloads and installs the poetry package manager"
	@echo "poetry-remove      : removes the poetry package manager"
	@echo "install            : installs required dependencies"
	@echo "install-dev        : installs the dev dependencies for the project"
	@echo "check-style        : run checks on all files without fixing them."
	@echo "fix-style          : run checks on files and potentially modifies them."
	@echo "check-safety       : run safety checks on all tests."
	@echo "lint               : run linting on all files (check-style + check-safety)"
	@echo "test               : run all tests."
	@echo "codecov            : check coverage of all the code."
	@echo "docs               : serve generated documentation locally."
	@echo "docker-build       : builds docker image for the package."
	@echo "docker-remove      : removes built docker image."
	@echo "clean              : cleans all unecessary files."

#* Poetry
.PHONY: poetry-download
poetry-download:
	curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | $(PYTHON) -

.PHONY: poetry-remove
poetry-remove:
	curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | $(PYTHON) - --uninstall

#* Installation
install:
	poetry update

.PHONY: install-dev
install-dev:
	poetry lock -n && poetry export --without-hashes > requirements.txt
	poetry install -n
	-poetry run mypy --install-types --non-interactive ./
	poetry run pre-commit install
	pre-commit autoupdate

#* Linting
.PHONY: check-style
check-style:
	poetry run isort --diff --check-only --settings-path pyproject.toml ./
	poetry run black --diff --check --config pyproject.toml ./
#   poetry run darglint --verbosity 2 inseq tests
	poetry run flake8 --config setup.cfg ./
#	poetry run mypy --config-file pyproject.toml ./

.PHONY: fix-style
fix-style:
	poetry run pyupgrade --exit-zero-even-if-changed --py38-plus **/*.py
	poetry run isort --settings-path pyproject.toml ./
	poetry run black --config pyproject.toml ./

.PHONY: check-safety
check-safety:
	poetry check
	poetry run safety check --full-report
	poetry run bandit -ll --recursive inseq tests

.PHONY: lint
lint: check-style check-safety

#* Linting
.PHONY: test
test:
	poetry run pytest -c pyproject.toml

.PHONY: codecov
codecov:
	poetry run pytest --cov inseq --cov-report html

#* Docs
.PHONY: docs
docs:
	cd docs && make html SPHINXOPTS="-W -j 4"
	cd docs/_build/html && python3 -m http.server 8080

#* Docker
# Example: make docker VERSION=latest
# Example: make docker IMAGE=some_name VERSION=0.1.0
.PHONY: docker-build
docker-build:
	@echo Building docker $(IMAGE):$(VERSION) ...
	docker build \
		-t $(IMAGE):$(VERSION) . \
		-f ./docker/Dockerfile --no-cache

# Example: make clean_docker VERSION=latest
# Example: make clean_docker IMAGE=some_name VERSION=0.1.0
.PHONY: docker-remove
docker-remove:
	@echo Removing docker $(IMAGE):$(VERSION) ...
	docker rmi -f $(IMAGE):$(VERSION)

#* Cleaning
.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

.PHONY: build-remove
build-remove:
	rm -rf build/

.PHONY: clean
clean: pycache-remove build-remove docker-remove
