help:
	@echo "Available commands:"
	@echo "  install - Install dependencies and pre-commit hooks"
	@echo "  clean   - Clean up virtual environment and lockfile"

# install dependencies and pre-commit hooks
install:
	uv sync --all-groups
	uv run pre-commit install

# clean up virtual environment and lockfile
clean:
	rm -rf .venv
	rm -rf uv.lock