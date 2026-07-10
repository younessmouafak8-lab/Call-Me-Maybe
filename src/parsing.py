import json
from pydantic import BaseModel, ValidationError, Field
import argparse


class Prompt(BaseModel):
    prompt: str = Field(min_length=1)


def get_prompts(file: str) -> list[Prompt]:
    with open(file, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("expected a list")
    prompts = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError("expected a dictionary")
        if len(item) > 1:
            raise ValueError(f"invalid key:value pair in dictionary number {i + 1}")

        prompts.append(Prompt(prompt=item.get('prompt')).prompt)
    return prompts


def get_functions(file: str) -> list[dict]:
    with open(file, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("expected a list")
    for func in data:
        if not isinstance(func, dict):
            raise ValueError("expected a dictionary")
        if 'name' not in func or 'description' not in func or \
                'parameters' not in func:
            raise ValueError("missing key")
        if not isinstance(func['parameters'], dict):
            raise ValueError("expected a dictionary")
        for param, value in func['parameters'].items():
            ...
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
        args = parse.parse_args()
        prompts = get_prompts(args.input)
        functions = get_functions(args.functions_definition)
        output_file = args.output
        return (prompts, functions, output_file)
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
