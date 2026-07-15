from .parsing import parsing
import json
import numpy as np
from time import time
from typing import Union


def prompt_builder(prompt: str, functions: list) -> str:
    return f"""
    You are a function-calling assistant.
    Your task is to analyze the user's request and return a single JSON object\
          describing the function call.

    Available functions:
    {functions}

    Output schema:
    {{"prompt": "<original user input>", "name": "<function name>",\
          "parameters": {{ ... }}}}
    Example:
    Input:
    What is the sum of 2 and 3?
    Output:
    {{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", \
        "parameters": {{"a": 2.0, "b": 3.0}}}}

    User Input:
    {prompt}
    """


def complete_parameters(parameters: dict, name: str) -> list:
    params = parameters[name]
    prms = []
    for param in params.keys():
        param_type = params[param]["type"]
        if param_type == "string":
            string = f'"{param}": "'
        else:
            string = f'"{param}":'
        prms.append((param, string, param_type))

    return prms


def valide_ids(vocabulary: dict) -> tuple:

    number_ids = set()
    for id, token in vocabulary.items():
        if token and all(c in "0123456789,." for c in token):
            number_ids.add(int(id))

    integer_ids = set()
    for id, token in vocabulary.items():
        if token and all(c in "0123456789,}" for c in token):
            integer_ids.add(int(id))

    boolean_id = set()
    for id, token in vocabulary.items():
        if token and all(c in "TrueFalse" for c in token):
            boolean_id.add(int(id))

    return (number_ids, integer_ids, boolean_id)


def check_this(logits: list, ids: list) -> None:
    for i in range(len(logits)):
        if i not in ids:
            logits[i] = -np.inf


def convert_value(value: str, param_type: str,
                  token: str) -> (float | int | bool | str):
    result: Union[float | int | bool | str]
    if value and param_type == "number":
        result = float(value)
    elif value and param_type == "integer":
        result = int(value)
    elif value and param_type == "boolean":
        if value == "False":
            result = bool(0)
        else:
            result = bool(1)
    else:
        value += token
        result = value.split('"')[0]
    return result


def validate_name(ids: list, index: int, gen_ids: list) -> list:
    ids = [lst for lst in ids if lst[:index] == gen_ids]
    return [i[index] for i in ids]


def main() -> None:
    p = parsing()
    if not p:
        return
    prompts, functions, output_file, model_name = p
    func_def = [f"{func['name']}: {func['parameters']}" for func in functions]
    params = {func["name"]: func["parameters"] for func in functions}
    from llm_sdk import Small_LLM_Model as model  # type: ignore[attr-defined]
    m = model(model_name)
    vocabulary_path = m.get_path_to_vocab_file()
    with open(vocabulary_path, "r") as f:
        vocabulary = json.load(f)
    vocabulary = {value: key for key, value in vocabulary.items()}
    start = time()
    output = []
    number_ids, integer_ids, boolean_ids = valide_ids(vocabulary)
    static_part = ' "parameters": {'
    static_ids = m.encode(static_part).tolist()[0]
    name_ids = [m.encode(func["name"] + '",').tolist()[0]
                for func in functions]
    for prompt in prompts:
        string = f'{{"prompt": "{prompt}", "name": "'
        ids = m.encode(prompt_builder(prompt, func_def) + (string)).tolist()[0]
        name_generated = False
        param_generated = False
        param_saved = False
        dic = {"prompt": prompt}
        parameters_dic = {}
        param_type = ""
        param_value: str = ""
        name = ""
        value = ""
        prm: list[tuple] = []
        tokens_generated = 0
        i = 0
        gen_ids: list[int] = []
        while 1:
            logits = m.get_logits_from_input_ids(ids)
            copy = logits.copy()
            if not name_generated:
                n_ids = validate_name(name_ids, i, gen_ids)
                check_this(copy, n_ids)
                i += 1
            elif (name_generated and param_generated and
                    param_type == "number" and not param_saved):
                if prm:
                    check_this(copy, number_ids)
                if '.' in param_value and param_value.endswith("0") and \
                        not prm:
                    check_this(copy, m.encode('}}').tolist()[0])

            elif (name_generated and param_generated and
                    param_type == "integer" and not param_saved):
                check_this(copy, integer_ids)
            elif (name_generated and param_generated and
                    param_type == "boolean" and not param_saved):
                if not ("True" in param_value or "False" in param_value):
                    check_this(copy, boolean_ids)
                elif "," not in param_value:
                    check_this(copy, m.encode(',').tolist()[0])
            next_token_id = int(np.argmax(copy))
            ids.append(next_token_id)
            value = m.decode(next_token_id)
            string += value
            if not name_generated:
                if '",' in value:
                    name_generated = True
                    dic.update({"name": name})
                else:
                    name += value
                    gen_ids.append(next_token_id)
            if name_generated and "parameters" not in string:
                ids += static_ids
                string += static_part
            if name_generated and "parameters" in string:
                tokens_generated += 1
                if not param_generated:
                    prm = complete_parameters(params, name)
                    if not prm:
                        param_saved = True
                    param_generated = True
                    param_value = ""
                    if prm:
                        param_name, temp, param_type = prm.pop(0)
                        ids += m.encode(temp).tolist()[0]
                        string += temp
                elif not param_saved and (',' in value or '}' in value):
                    tokens_generated = 0
                    result = convert_value(param_value, param_type, value)
                    parameters_dic.update({param_name: result})
                    param_value = ""
                    if prm:
                        param_name, temp, param_type = prm.pop(0)
                        ids += m.encode(temp).tolist()[0]
                        string += temp
                    else:
                        param_saved = True
                elif param_generated and not param_saved:
                    param_value += value
                if param_saved:
                    dic.update({"parameters": parameters_dic})
                    if string.endswith('"}'):
                        ids += m.encode("}").tolist()[0]
                        string += "}"
                    if not string.strip().endswith('}}'):
                        ids += m.encode("}}").tolist()[0]
                        string += "}}"
                if tokens_generated > len(prompt) + 10:
                    tokens_generated = 0
                    if not len(prm):
                        result = convert_value(param_value,
                                               param_type, value)
                        parameters_dic.update({param_name: result})
                        dic.update({"parameters": parameters_dic})
                        ids += m.encode("}}").tolist()[0]
                        string += "}}"
                        param_saved = True
                    else:
                        ids += m.encode(",").tolist()[0]
                        string += ","

            # print(value)
            print(string)
            if param_saved:
                break
        print(dic)
        print("###############################################")
        output.append(dic)
    end = time()
    print(end - start)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=4)


try:
    main()
except Exception as e:
    print(f"Error: {e}")
