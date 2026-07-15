INPUT = "data/input/function_calling_tests.json"
FUNC_DEF = "data/input/functions_definition.json"
OUTPUT = "data/output/function_calls.json"
MODEL = "Qwen/Qwen2.5-Coder-0.5B"

install:
	@pip install uv
	@uv sync

run:
	@uv run  python3 -m src --input $(INPUT) --functions_definition $(FUNC_DEF) --output $(OUTPUT)

debug:
	@uv run  python3 -pdb -m src

clean:
	@echo "Cleaning temporary files..."
	rm -rf __pycache__ .mypy_cache .pytest_cache
	rm -rf */__pycache__
	rm -f *.pyc

lint:
	@flake8 src 
	@mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --exclude llm_sdk