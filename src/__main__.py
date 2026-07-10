from .parsing import parsing
from llm_sdk.llm_sdk import Small_LLM_Model as model
import json
import numpy as np
from time import time


def prompt_builder(prompt, functions):
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
    JSON:
    """


def get_name(token: str):
    if not hasattr(get_name, "name"):
        get_name.name = ""
    if '",' not in token:
        get_name.name += token
        return None
    result = get_name.name
    get_name.name = ""
    return result


def complete_parameters(parameters: dict, name):
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


def valide_ids(functions, vocabulary):
    allowed_chars = set()
    for func in functions:
        for c in func['name']:
            allowed_chars.add(c)
    allowed_chars.add('"')
    allowed_chars.add(',')
    name_ids = set()
    for id, token in vocabulary.items():
        if token and all(c in allowed_chars for c in token):
            name_ids.add(int(id))

    number_ids = set()
    for id, token in vocabulary.items():
        if token and all(c in "0123456789.," for c in token):
            number_ids.add(int(id))

    integer_ids = set()
    for id, token in vocabulary.items():
        if token and all(c in "0123456789," for c in token):
            integer_ids.add(int(id))

    boolean_id = []
    for id, token in vocabulary.items():
        if token and all(c in "TrueFalse" for c in token):
            boolean_id.append(int(id))

    return (name_ids, number_ids, integer_ids)


def check_this(logits, ids):
    for i in range(len(logits)):
        if i not in ids:
            logits[i] = -np.inf


def main():
    p = parsing()
    if not p:
        return
    prompts, functions, output_file = p
    func_def = [f"{func['name']}: {func['parameters']}" for func in functions]
    params = {func["name"]: func["parameters"] for func in functions}
    m = model()
    vocabulary_path = m.get_path_to_vocab_file()
    with open(vocabulary_path, "r") as f:
        vocabulary = json.load(f)
    vocabulary = {value: key for key, value in vocabulary.items()}
    start = time()
    output = []
    name_ids, number_ids, integer_ids = valide_ids(functions, vocabulary)
    static_part = ' "parameters": {'
    static_ids = m.encode(static_part).tolist()[0]
    for prompt in prompts:
        string = f'{{"prompt": "{prompt}", "name": "'
        ids = m.encode(prompt_builder(prompt, func_def) + (string)).tolist()[0]
        name_generated = False
        param_generated = False
        param_saved = False
        dic = {"prompt": prompt}
        parameters_dic = {}
        param_type = ""
        param_value = ""
        while 1:
            logits = m.get_logits_from_input_ids(ids)
            copy = logits.copy()
            if not name_generated:
                check_this(copy, name_ids)
            elif (name_generated and param_generated and
                    param_type == "number" and not param_saved):
                check_this(copy, number_ids)
            elif (name_generated and param_generated and
                    param_type == "integer" and not param_saved):
                check_this(copy, integer_ids)
            next_token_id = int(np.argmax(copy))
            ids.append(next_token_id)
            value = m.decode(next_token_id)
            string += value
            if not name_generated:
                name = get_name(value)
                if '",' in value:
                    name_generated = True
                    dic.update({"name": name})
            if name_generated and "parameters" not in string:
                ids += static_ids
                string += static_part
            if name_generated and "parameters" in string:
                if not param_generated:
                    prm = complete_parameters(params, name)
                    param_generated = True
                    param_value = ""
                    param_name, temp, param_type = prm.pop(0)
                    ids += m.encode(temp).tolist()[0]
                    string += temp
                elif not param_saved and (',' in value or '}' in value):
                    if param_value and param_type == "number":
                        param_value = float(param_value)
                    elif param_value and param_type == "integer":
                        param_value = int(param_value)
                    else:
                        param_value += value
                        param_value = param_value.split('"')[0]
                    parameters_dic.update({param_name: param_value})
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
                    string = string.rstrip(',')
                    dic.update({"parameters": parameters_dic})
                    if string.endswith('"}'):
                        ids += m.encode("}").tolist()[0]
                        string += "}"
                    elif param_type != "string":
                        ids += m.encode("}}").tolist()[0]
                        string += "}}"

            print(value)
            print(string)
            if param_saved:
                break
            try:
                if string.endswith("\n"):
                    break
                json.loads(string)
                break
            except json.JSONDecodeError:
                pass
        print(dic)
        print("###############################################")
        output.append(dic)
    end = time()
    print(end - start)
    with open(output_file, "w") as f:
        json.dump(output, f)


try:
    main()
except Exception as e:
    print(f"Error: {e}")
