import json
import argparse
import os


def get_prompts(file: str) -> list[str]:
    with open(file, "r") as f:
        try:
            data = json.load(f)
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
    valid_types = {"number", "string", "boolean", "integer"}
    with open(file, "r") as f:
        try:
            data = json.load(f)
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
        if ' ' in func["name"] or not len(func["name"]) or \
                '"' in func["name"] or ',' in func["name"]:
            raise ValueError("invalid function name")
        if not isinstance(func['parameters'], dict):
            raise ValueError("expected a dictionary")
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
    try:
        parse = argparse.ArgumentParser()
        parse.add_argument("--input",
                           default="data/input/function_calling_tests.json")
        parse.add_argument("--functions_definition",
                           default="data/input/functions_definition.json")
        parse.add_argument("--output",
                           default="data/output/function_calling_results.txt")
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
