"""Constrained-decoding function-calling pipeline.

This module drives a small LLM token-by-token to translate natural language
prompts into structured function calls (name + typed parameters), using
constrained decoding to guarantee valid, schema-compliant JSON output.
"""

from .parsing import parsing
from llm_sdk import Small_LLM_Model as model  # type: ignore[attr-defined]
import json
import numpy as np
from time import time
from typing import Union, Any


def prompt_builder(prompt: str, functions: list) -> str:
    """Build the natural-language prompt sent to the LLM.

    Wraps the user's request together with the list of available function
    signatures and a one-shot example, so the model has the context needed
    to pick a function and format its parameters.

    Args:
        prompt: The original natural-language user input.
        functions: List of formatted function descriptions (name,
            parameters, description) to present to the model.

    Returns:
        The full prompt string to be tokenized and fed to the model.
    """
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
    What is the sum of 14 and 15?
    Output:
    {{"prompt": "What is the sum of 14 and 15?", "name": "fn_add_numbers", \
        "parameters": {{"a": 14.0, "b": 15.0}}}}

    User Input:
    {prompt}
    """


def complete_parameters(parameters: dict, name: str) -> list:
    """Build the ordered list of parameters that must be generated
    for a function.

    For the chosen function name, look up its parameter definitions and
    prepare, for each parameter, the literal text that should be injected
    into the generation stream just before its value (e.g. an opening
    quote for strings, nothing extra for numbers/booleans).

    Args:
        parameters: Mapping of function name to its parameter definitions,
            as declared in functions_definition.json.
        name: Name of the function whose parameters should be prepared.


    Returns:
        A list of (param_name, injected_string, param_type) tuples, in the
        order the parameters should be generated.
    """
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


def check_this(logits: list, ids: set | list) -> None:
    """Mask out every logit whose token ID is not in the allowed set.

    Mutates `logits` in place, setting every index not present in `ids` to
    negative infinity so it can never be selected by argmax. This is the
    core masking step of constrained decoding.

    Args:
        logits: Raw logits produced by the model for the next token.
        ids: Collection of token IDs that are allowed at this generation
            step.

    Returns:
        None. The `logits` list is modified in place.
    """
    for i in range(len(logits)):
        if i not in ids:
            logits[i] = -np.inf


def convert_value(value: str, param_type: str,
                  token: str, prm: list) -> (float | int | bool | str):
    """Convert an accumulated raw parameter value to its final typed value.

    Depending on `param_type`, casts the accumulated string to a float,
    int, or bool. For strings, trims the trailing JSON punctuation
    (`",` or `"}` / `"}}`) that was generated as part of the decoding
    stream so only the actual string content remains.

    Args:
        value: The raw text accumulated so far for this parameter.
        param_type: Declared type of the parameter ("number", "integer",
            "boolean", or "string").
        token: The last generated token, appended to `value` before
            trimming (used only for the string case).
        prm: Remaining parameters still to be generated; used to decide
            whether this string value is followed by another parameter.

    Returns:
        The parameter value converted to its proper Python type
        (float, int, bool, or str).
    """
    try:
        result: Union[float | int | bool | str] = ""
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
            if prm or value.endswith('",'):
                result = value.split('",')[0]
            elif value.strip().endswith('"}') or value.strip().endswith('"}}'):
                result = value.split('"}')[0]
            else:
                result = value
        return result
    except ValueError:
        print("try entering a valid value next time :)")
        if param_type == "number" or param_type == "integer":
            result = 0
    return result


def validate_name(ids: list, index: int, gen_ids: list) -> list:
    """Filter candidate function-name token sequences by
    the tokens generated so far.

    Keeps only the candidate token-ID sequences (one per known function
    name) whose prefix matches what has already been generated, then
    returns the next expected token ID for each surviving candidate. This
    lets constrained decoding restrict the next token to only those that
    can continue a valid function name.

    Args:
        ids: List of candidate token-ID sequences, one per function name.
        index: Position in the sequence currently being generated.
        gen_ids: Token IDs generated so far for the function name.

    Returns:
        The list of token IDs at position `index` for every candidate
        sequence still consistent with `gen_ids`.
    """
    ids = [lst for lst in ids if lst[:index] == gen_ids]
    return [i[index] for i in ids]


def my_encode(text: str, m: model) -> Any:
    ids = m.encode(text)
    ids = np.array(ids[0], dtype=int).tolist()
    return ids


def main() -> None:
    """Run the full function-calling pipeline.

    Parses CLI arguments and input files, loads the small LLM and its
    vocabulary, then for each prompt performs constrained, token-by-token
    generation to produce a JSON object with the selected function name and
    correctly typed parameters. Results are written to the configured
    output file.

    Returns:
        None. Side effects: writes the JSON results to `output_file` and
        prints progress/debugging information to stdout.
    """
    p = parsing()
    if not p:
        return
    prompts, functions, output_file, model_name = p

    func_def = [f"{func['name']}: {func['parameters']}, "
                f"{func['description']}" for func in functions]

    params = {func["name"]: func["parameters"] for func in functions}

    m = model(model_name)

    output = []

    number_ids = set()
    for n in '0123456789.+-,':
        number_ids.add(my_encode(n, m)[0])

    integer_ids = set()
    for integer in "0123456789,}+-":
        integer_ids.add(my_encode(integer, m)[0])

    boolean_ids = set()
    for b in ["True", "False"]:
        boolean_ids.add(my_encode(b, m)[0])

    name_ids = [my_encode(func["name"] + '",', m)
                for func in functions]

    static_part = ' "parameters": {'
    static_ids = my_encode(static_part, m)
    start = time()
    for prompt in prompts:
        string = f'{{"prompt": {prompt}, "name": "'
        ids = my_encode(prompt_builder(prompt, func_def) + (string), m)
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
            copy = logits

            if not name_generated:
                n_ids = validate_name(name_ids, i, gen_ids)
                check_this(copy, n_ids)
                i += 1

            elif (name_generated and param_generated and
                    param_type == "number" and not param_saved):
                if prm:
                    check_this(copy, number_ids)
                else:
                    nbr_ids = [my_encode(c, m)[0]
                               for c in '0123456789.+-}']
                    check_this(copy, nbr_ids)

            elif (name_generated and param_generated and
                    param_type == "integer" and not param_saved):
                check_this(copy, integer_ids)

            elif (name_generated and param_generated and
                    param_type == "boolean" and not param_saved):
                if not ("True" in param_value or "False" in param_value):
                    check_this(copy, boolean_ids)
                elif "," not in param_value and prm:
                    check_this(copy, my_encode(',', m))
                elif not prm:
                    check_this(copy, my_encode('}', m))

            next_token_id = int(np.argmax(copy))
            value = m.decode(next_token_id)
            ids.append(next_token_id)
            string += value

            if not name_generated:
                if '",' in value:
                    tmp = value.split('",')
                    if tmp[0]:
                        name += tmp[0]
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
                        ids += my_encode(temp, m)
                        string += temp
                elif not param_saved and (',' in value or '}' in value):
                    tokens_generated = 0
                    result = convert_value(param_value, param_type, value, prm)
                    parameters_dic.update({param_name: result})
                    param_value = ""
                    if prm:
                        param_name, temp, param_type = prm.pop(0)
                        ids += my_encode(temp, m)
                        string += temp
                    else:
                        param_saved = True
                elif param_generated and not param_saved:
                    param_value += value

                if param_saved:
                    dic.update({"parameters": parameters_dic})
                    if string.endswith('"}'):
                        ids += my_encode("}", m)
                        string += "}"
                    elif not string.strip().endswith('}}') and \
                            "}" not in string:
                        ids += my_encode("}}", m)
                        string += "}}"
                    elif "}}" not in string:
                        ids += my_encode("}", m)
                        string += "}"

                if tokens_generated > len(prompt) + 10:
                    tokens_generated = 0
                    if not len(prm):
                        result = convert_value(param_value,
                                               param_type, value, prm)
                        parameters_dic.update({param_name: result})
                        dic.update({"parameters": parameters_dic})
                        ids += m.my_encode("}}", m)
                        string += "}}"
                        param_saved = True
                    else:
                        ids += my_encode(",", m)
                        string += ","

            print(value)
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
