import json
from pydantic import BaseModel, ValidationError, Field
import argparse


class Prompt(BaseModel):
    prompt: str = Field(min_length=1)


def get_prompts(file: str) -> list[Prompt]:
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
            raise ValueError(f"invalid key:value pair in dictionary number {i + 1}")

        prompts.append(Prompt(prompt=item.get('prompt')).prompt)
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
                'parameters' not in func:
            raise ValueError("missing key")
        if ' ' in func["name"] or \
                not len(func["name"]) or '"' in func["name"] or ',' in func["name"]:
            raise ValueError("invalid function name")
        if not isinstance(func['parameters'], dict):
            raise ValueError("expected a dictionary")
        for param, value in func['parameters'].items():
            if not len(param):
                raise ValueError(f"invalid parameter '{param}'")
            if not isinstance(value, dict):
                raise ValueError(f"parameter '{param}' must be a dictionary")
            if 'type' not in value:
                raise ValueError(f"parameter '{param}' missing 'type' field")
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
                           default="data/output/function_calling_results.json")
        parse.add_argument("--model", default="Qwen/Qwen3-0.6B")
        args = parse.parse_args()
        prompts = get_prompts(args.input)
        functions = get_functions(args.functions_definition)
        output_file = args.output
        model = args.model
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
    except ValidationError as e:
        print("Error:", end="")
        for err in e.errors():
            print(err["msg"])
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []
