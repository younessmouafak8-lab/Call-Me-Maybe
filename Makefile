


install:
	@pip install uv
	@uv sync

run:
	@uv run  python3 -m src

debug:
	@uv run  python3 -pdb -m src

clean:
	@echo "Cleaning temporary files..."
	rm -rf __pycache__ .mypy_cache .pytest_cache
	rm -rf */__pycache__
	rm -f *.pyc

lint:
	@flake8 . --exclude=.venv,llm_sdk
	@mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --exclude llm_sdk