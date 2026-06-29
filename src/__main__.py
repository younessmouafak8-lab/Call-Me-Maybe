from .parsing import parsing
from llm_sdk.llm_sdk import Small_LLM_Model as model
import json

def prompt_builder(prompt, functions):
    return f"""Extract the function name and parameters
    as a valid JSON object. \
      Functions: \
          {functions} \
      Example: \
          Input text: What is the sum of 2 and 3? \
          JSON output: \"name\": \"fn_add_numbers\", \"parameters\": {{"a": 2.0, "b": 3.0}}.\
      Rules: \
          1. If a parameter type is a Number cast it to a float. \
          2. Output ONLY the raw JSON. \
      Input Text: {prompt} \
          JSON output: \
  """



def main():
    p = parsing()
    if not p:
        return
    prompts, functions, output_file = p
    func_def = [f"{func['name']}: {func['description']}" for func in functions]
    # prompt_builder(func_def)
    m = model()
    vocabulary_path = m.get_path_to_vocab_file()
    with open(vocabulary_path, "r") as f:
        vocabulary = json.load(f)
    tensor = m.encode(prompt_builder(prompts[0], func_def))
    ids = tensor.tolist()[0]
    logits = m.get_logits_from_input_ids(ids)
    print(logits)


main()
