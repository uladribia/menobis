.PHONY: all clean develop lint format test test-python test-rust docs

PROJECT_PKG ?= menobis
UV_EXCLUDE_NEWER ?= $(shell date -u -d '7 days ago' '+%Y-%m-%dT%H:%M:%SZ')
UV := UV_EXCLUDE_NEWER=$(UV_EXCLUDE_NEWER) uv

all: lint test

clean:
	rm -rf .pytest_cache .ruff_cache .hypothesis htmlcov target
	rm -f .coverage

develop:
	$(UV) run maturin develop

lint:
	$(UV) run --frozen ruff format --check
	$(UV) run --frozen ruff check
	$(UV) run --frozen ty check
	cargo fmt --all -- --check
	cargo clippy --workspace --all-targets -- -D warnings

format:
	$(UV) run --frozen ruff format
	$(UV) run --frozen ruff check --fix
	cargo fmt --all

test: test-python test-rust

test-python:
	$(UV) run --frozen pytest

test-rust:
	cargo test --workspace

docs:
	$(UV) run --frozen mkdocs build --strict
