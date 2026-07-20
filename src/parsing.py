"""Input parsing and validation utilities.

This module handles CLI argument parsing and validation of the two input
JSON files (prompts and function definitions) required by the
function-calling pipeline, ensuring both are well-formed before generation
begins.
"""

import json
import argparse
import os


def check_keys(data: list) -> dict:
    """Build a dict from JSON key/value pairs while rejecting duplicate keys.

    Intended for use as the `object_pairs_hook` of `json.load`, so that a
    JSON object with a repeated key raises an error instead of silently
    keeping only the last occurrence.

    Args:
        data: List of (key, value) pairs as produced by the JSON decoder
            for a single object.

    Returns:
        A dict built from `data`.

    Raises:
        ValueError: If the same key appears more than once in `data`.
    """
    lst = set()
    for key, _ in data:
        if key in lst:
            raise ValueError("duplicated keys are not tolerated")
        lst.add(key)

    return dict(data)


def get_prompts(file: str) -> list[str]:
    """Load and validate the list of natural-language prompts.

    Reads the given JSON file, expecting a non-empty list of objects each
    containing exactly one key, "prompt", mapped to a non-empty string.

    Args:
        file: Path to the JSON file containing the prompts.

    Returns:
        The list of prompt strings, in file order.

    Raises:
        ValueError: If the file is not valid JSON, is not a non-empty
            list, contains items that aren't dictionaries, contains extra
            keys, is missing the "prompt" key, or has a "prompt" value that
            is not a non-empty string.
    """
    with open(file, "r") as f:
        try:
            data = json.load(f, object_pairs_hook=check_keys)
        except json.decoder.JSONDecodeError:
            raise ValueError("invalid json file")
    if not isinstance(data, list) or not data:
        raise ValueError("expected a non-empty list")
    prompts = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError("expected a dictionary")
        if len(item) > 1:
            raise ValueError("invalid key:value pair in dictionary "
                             f"number {i + 1}")
        if "prompt" not in item:
            raise ValueError(f"missing prompt key in dictionary "
                             f"number {i + 1}")
        prompt = item["prompt"]
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"expected a non-empty string in dictionary "
                             f"number {i + 1}")
        prompts.append(item["prompt"])
    return prompts


def get_functions(file: str) -> list[dict]:
    """Load and validate the available function definitions.

    Reads the given JSON file, expecting a non-empty list of function
    definitions. Each definition must declare exactly "name", "description",
    "parameters", and "returns", with well-formed names, a description of
    reasonable length, and parameter/return types restricted to
    "number", "string", "boolean", or "integer".

    Args:
        file: Path to the JSON file containing the function definitions.

    Returns:
        The validated list of function definition dictionaries, as loaded
        from the file.

    Raises:
        ValueError: If the file is not valid JSON, or any function
            definition is malformed: not a dictionary, missing/extra keys,
            invalid name or description, invalid "parameters"/"returns"
            structure, or an unsupported parameter/return type.
    """
    valid_types = {"number", "string", "boolean", "integer"}
    names = set()
    with open(file, "r") as f:
        try:
            data = json.load(f, object_pairs_hook=check_keys)
        except json.decoder.JSONDecodeError:
            raise ValueError("invalid json file")
    if not isinstance(data, list) or not data:
        raise ValueError("expected a list")
    for func in data:
        if not isinstance(func, dict):
            raise ValueError("expected a dictionary")
        if 'name' not in func or 'description' not in func or \
                'parameters' not in func or "returns" not in func:
            raise ValueError("missing key")
        if len(func) > 4:
            raise ValueError("invalid key:value pair")
        if not isinstance(func["name"], str) or \
                not isinstance(func["description"], str):
            raise ValueError("expected a str for function"
                             " name and description")
        if not func["name"].isidentifier() or not func["name"].isascii():
            raise ValueError("invalid function name")
        if func["name"] in names:
            raise ValueError("duplicate function name")
        names.add(func["name"])
        if len(func["description"]) < 5:
            raise ValueError("invalid function description")

        if not isinstance(func['parameters'], dict) or \
                not isinstance(func['returns'], dict):
            raise ValueError("expected a dictionary")
        if "type" not in func['returns']:
            raise ValueError("return field missing type")
        if len(func['returns']) > 1:
            raise ValueError("multiple key:value pair in return value")
        for param, value in func['returns'].items():
            if not len(param) or ' ' in param:
                raise ValueError(f"invalid return value '{param}'")
            if not isinstance(value, str):
                raise ValueError(f"type '{param}' must be a string")
            if value not in valid_types:
                raise ValueError(f"parameter '{param}' has unsupported type "
                                 f"'{value}'")
        for param, value in func['parameters'].items():
            if not len(param) or ' ' in param:
                raise ValueError(f"invalid parameter '{param}'")
            if not isinstance(value, dict):
                raise ValueError(f"parameter '{param}' must be a dictionary")
            if 'type' not in value:
                raise ValueError(f"parameter '{param}' missing 'type' field")
            if len(value) > 1:
                raise ValueError("invalid key:value pair in dictionary "
                                 f"{param}")
            if value['type'] not in valid_types:
                raise ValueError(f"parameter '{param}' has unsupported type "
                                 f"'{value['type']}'")
    return data


def parsing() -> tuple | list:
    """Parse CLI arguments and load/validate all input files.

    Reads the `--input`, `--functions_definition`, `--output`, and `--model`
    command-line arguments (each with sensible defaults), validates that
    the input/output file paths end in ".json", loads and validates the
    prompts and function definitions, and ensures the output directory
    exists.

    Returns:
        On success, a tuple (prompts, functions, output_file, model) where
        `prompts` is the list of prompt strings, `functions` is the list of
        validated function definitions, `output_file` is the path to write
        results to, and `model` is the model name/path to use.
        On any failure (missing file, permission error, invalid JSON/schema,
        or any other exception), prints an error message and returns an
        empty list instead of raising.
    """
    try:
        parse = argparse.ArgumentParser()
        parse.add_argument("--input",
                           default="data/input/function_calling_tests.json")
        parse.add_argument("--functions_definition",
                           default="data/input/functions_definition.json")
        parse.add_argument("--output",
                           default="data/output/function_calling_results.json")
        parse.add_argument("--model", default="Qwen/Qwen3-0.6B")
        args = parse.parse_args()
        prompts = get_prompts(args.input)
        functions = get_functions(args.functions_definition)
        output_file = args.output
        model = args.model
        for file in [args.input, args.functions_definition, args.output]:
            if not file.endswith(".json"):
                raise ValueError("file name must end with '.json'")
        directory = os.path.dirname(output_file)
        if directory:
            os.makedirs(directory, exist_ok=True)
        return (prompts, functions, output_file, model)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return []
    except PermissionError:
        print("Error: permission denied")
        return []
    except ValueError as e:
        print(f"Error: {e}")
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []
