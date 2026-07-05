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
    for param in params.keys():
        yield f' "{param}": '


# def add_parameter(string, model):
#     try:
#         temp = next(string)
#         ids += model.encode(temp).tolist()[0]
#         string += temp
#     except StopIteration:
#         pass

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
    # start  =time
    output = []
    for prompt in prompts:
        string = f'{{"prompt": "{prompt}", "name": "'
        # print(string,  end="", flush=True)
        ids = m.encode(prompt_builder(prompt, func_def) + (string)).tolist()[0]
        # generated_ids = []
        name_generated = False
        param_generated = False
        param_generated = False
        dic = {"prompt": prompt}
        while 1:
            logits = m.get_logits_from_input_ids(ids)
            next_token_id = int(np.argmax(logits))
            ids.append(next_token_id)
            # generated_ids.append(next_token_id)
            value = m.decode(next_token_id)
            string += value
            if not name_generated:
                name = get_name(value)
                if '",' in value:
                    name_generated = True
                    dic.update({"name": name})
            if name_generated and "parameters" not in string:
                parameters = ' "parameters": {'
                ids += m.encode(parameters).tolist()[0]
                string += parameters
            if name_generated and "parameters" in string:
                if not param_generated:
                    prm = complete_parameters(params, name)
                    param_generated = True
                    try:
                        temp = next(prm)
                        ids += m.encode(temp).tolist()[0]
                        string += temp
                    except StopIteration:
                        pass
                elif ',' in value:
                    try:
                        temp = next(prm)
                        ids += m.encode(temp).tolist()[0]
                        string += temp
                    except StopIteration:
                        pass
            print(string)
            try:
                if string.endswith("\n"):
                    break
                json.loads(string)
                break
            except json.JSONDecodeError:
                pass
        print("###############################################")
        output.append(dic)
    with open("data/output.json", "w") as f:
        json.dump(output, f)


main()
