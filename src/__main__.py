from .parsing import parsing
from llm_sdk.llm_sdk import Small_LLM_Model as model
import json
import numpy as np
from time import time


def prompt_builder(prompt, functions):
    return f"""
    You are a function-calling assistant.
    Your task is to analyze the user's request and return a single JSON object describing the function call.

    Available functions:
    {functions}

    Output schema:
    {{"prompt": "<original user input>", "name": "<function name>", "parameters": {{ ... }}}}
    Example:
    Input:
    What is the sum of 2 and 3?
    Output:
    {{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {{"a": 2.0, "b": 3.0}}}}

    User Input:
    {prompt}
    JSON:
    """


def get_name(token: str, string: str):
    name = ""
    if '",' not in token:
        return name
    else:
        name = string.split('"')[7]
    return name


def complete_parameters(parameters: dict, name):
    params = parameters[name]
    for param in params.keys():
        yield f'"{param}": '


def main():
    p = parsing()
    if not p:
        return
    prompts, functions, output_file = p
    func_def = [f"{func['name']}: {func['parameters']}" for func in functions]
    params = {func["name"]: func["parameters"] for func in functions}
    m = model()
    # vocabulary_path = m.get_path_to_vocab_file()
    # with open(vocabulary_path, "r") as f:
    #     vocabulary = json.load(f)
    # vocabulary = {value: key for key, value in vocabulary.items()}
    output = []
    # start  =time
    for prompt in prompts:
        string = f'{{"prompt": "{prompt}", "name": "'
        # print(string,  end="", flush=True)
        ids = m.encode(prompt_builder(prompt, func_def) + (string)).tolist()[0]
        # generated_ids = []
        name_generated = False
        param_generated = False
        while 1:
            logits = m.get_logits_from_input_ids(ids)
            next_token_id = int(np.argmax(logits))
            ids.append(next_token_id)
            # generated_ids.append(next_token_id)
            value = m.decode(next_token_id)
            string += value
            if not name_generated:
                name = get_name(value, string)
            if name:
                name_generated = True
            if name_generated and "parameters" not in string:
                parameters = ' "parameters": {'
                ids += m.encode(parameters).tolist()[0]
                string += parameters
            if name_generated and "parameters" in string:
                if not param_generated:
                    prm = complete_parameters(params, name)
                    param_generated = True
                try
                temp = next(prm)
                ids += m.encode(temp).tolist()[0]
                string += temp
            print(string)
            try:
                if string.endswith("\n"):
                    break
                json.loads(string)
                break
            except json.JSONDecodeError:
                pass
        output.append(json.loads(string))
        print(name)
        print("###############################################")
    with open("data/output.json", "w") as f:
        json.dump(output, f)


main()
