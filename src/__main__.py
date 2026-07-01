from .parsing import parsing
from llm_sdk.llm_sdk import Small_LLM_Model as model
import json
import numpy as np


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


def main():
    p = parsing()
    if not p:
        return
    prompts, functions, output_file = p
    func_def = [f"{func['name']}: {func['description']}" for func in functions]
    # prompt_builder(func_def)
    m = model()
    # vocabulary_path = m.get_path_to_vocab_file()
    # with open(vocabulary_path, "r") as f:
    #     vocabulary = json.load(f)
    # vocabulary = {value: key for key, value in vocabulary.items()}
    for prompt in prompts:
        tensor = m.encode(prompt_builder(prompt, func_def))
        ids = tensor.tolist()[0]
        generated_ids = []
        while 1:
            logits = m.get_logits_from_input_ids(ids)
            next_token_id = int(np.argmax(logits))
            ids.append(next_token_id)
            generated_ids.append(next_token_id)
            result = m.decode(generated_ids)

            try:
                obj = json.loads(result)
                break
            except json.JSONDecodeError:
                pass
        print(result, end="")



main()
