.PHONY: install install-dev fix test ci

install: ## Install prod dependencies
	uv sync

install-dev: ## Install dev dependencies + wire git hooks
	uv sync --extra dev
	git config core.hooksPath .githooks

fix: ## Auto-fix everything: lint fixes + format
	uv run ruff check --fix .
	uv run ruff format .

test: ## Run the test suite
	uv run pytest

ci: ## What CI runs — lint + format-check + tests (no mutation)
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest
