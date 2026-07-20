*This project has been created as part of the 42 curriculum by ymouafak.*

# call me maybe — Function Calling with Constrained Decoding

## Description

This project turns natural-language requests into structured, machine-executable function calls, using a small (0.6B parameter) language model.

Given a prompt like *"What is the sum of 2 and 3?"*, the goal is not to answer the question in plain text, but to output a JSON object naming the function to call and its correctly typed arguments:

```json
{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

Small models are unreliable at producing valid JSON on their own (prompting alone might succeed only ~30% of the time). To reach near-perfect reliability, this project implements **constrained decoding**: at every generation step, the model's logits are masked so that only tokens consistent with the JSON schema (and the expected parameter types) can be selected. The result is output that is always valid JSON and always matches the function/parameter schema, regardless of how well the model "wants" to behave.

Given a set of function definitions (name, parameters, types, description) and a list of prompts, the program:
1. Picks the correct function for each prompt.
2. Extracts and correctly types every required argument.
3. Writes the results to a single output JSON file.

## Instructions

### Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for dependency management
- The `llm_sdk` package, copied into the same directory as `src` (provided separately, not included in this repository)

### Installation

```bash
uv sync
```

This installs the project dependencies (`numpy`, `pydantic`) into a managed virtual environment.

### Running

```bash
uv run python -m src [--functions_definition <function_definition_file>] [--input <input_file>] [--output <output_file>]
```

By default:
- `--input` defaults to `data/input/function_calling_tests.json`
- `--functions_definition` defaults to `data/input/functions_definition.json`
- `--output` defaults to `data/output/function_calling_results.json`
- `--model` defaults to `Qwen/Qwen3-0.6B`

Example:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Makefile

```bash
make install   # install dependencies
make run       # run the program with default paths
make debug     # run the program under pdb
make lint      # flake8 + mypy
make clean     # remove __pycache__, .mypy_cache, etc.
```

## Algorithm Explanation

The core idea is to intervene on the model's output distribution **before** a token is chosen, instead of hoping the model naturally produces valid JSON.

1. **Explicit mask construction from single-character/literal tokens.** At startup, instead of scanning the whole vocabulary for tokens whose *surface text* happens to be built only from allowed characters, each mask is built directly from the tokenizer's encoding of the individual characters (or literals) that are actually legal at that position: `number_ids` from encoding each of `0123456789.+-,` on its own, `integer_ids` from encoding each of `0123456789,}+-` on its own, and `boolean_ids` from encoding the two full literals `"True"` and `"False"` as whole tokens. Building the masks this way ties them to the tokenizer's own token IDs for those exact symbols, rather than to a same-charset heuristic that could also match unrelated words.

2. **Skeleton injection.** Instead of letting the model generate JSON punctuation freely, the fixed parts of the schema (`{"prompt": "...", "name": "`, ` "parameters": {`, the parameter key and opening quote, the closing `}}`) are injected directly into the token stream. The model only ever has to generate the *variable* parts: the function name and the parameter values.

3. **Function-name masking.** Before the function name is fully generated, the decoder only allows tokens that are a valid continuation of at least one real function name (pre-encoded from the function definitions). At each step, candidate name-token sequences are filtered down to those matching the tokens generated so far, and only the next token of the surviving candidates is allowed. This guarantees the model can never hallucinate a function name that doesn't exist.

4. **Type-constrained parameter masking, aware of remaining parameters.** Once a parameter's type is known (from `functions_definition.json`), the logits are masked using the pre-built sets, with the allowed set adapted to whether more parameters still need to be generated afterwards:
   - `number` → while another parameter still follows, only the `number_ids` set (digits, `.`, `+`, `-`, `,`) is allowed; on the *last* parameter, `,` is dropped in favor of `}`, since a comma would no longer make sense right before the object closes.
   - `integer` → only the `integer_ids` set (digits, `,`, `}`, `+`, `-`) is allowed.
   - `boolean` → only the two whole-literal tokens (`True`/`False`) are allowed until one is chosen; afterwards, the decoder allows `,` only if another parameter still follows, or `}` only if this was the last one — so the terminator token itself is now selected based on position rather than always defaulting to a comma.
   - `string` → left unconstrained (any token), since arbitrary text should be preserved as-is; the JSON-closing punctuation is injected rather than generated.

5. **Value extraction and typing.** As tokens are generated for a parameter, they are accumulated into a raw string. Once a delimiter (`,` or `}`) is produced, the raw value is converted to its final Python type (`float`, `int`, `bool`, or a cleaned-up `str`) and stored.

6. **Termination and safety net.** After all parameters for a function are collected, the remaining closing braces are injected to complete the JSON object. A token-count safety net also forces closure if a single parameter generates abnormally long output, preventing an infinite loop on a misbehaving generation.

The result: every generated token is either injected (guaranteed valid) or masked to a schema-valid subset built from the tokenizer's own encoding of the allowed symbols, so the final string is always parseable JSON that matches the declared schema exactly.

## Design Decisions

- **Skeleton injection over pure generation**: encoding fixed JSON punctuation directly into the token stream (rather than asking the model to generate it) removes an entire class of formatting errors (missing quotes, wrong brace count, etc.).
- **Pre-computed type sets**: classifying the vocabulary into number/integer/boolean token sets once at startup, rather than per-token, keeps the per-step masking cheap (`valide_ids` runs once per file, not once per token).
- **Separation of parsing and generation**: input validation and CLI handling live in `parsing.py`, fully decoupled from the decoding logic in the main module. Malformed input files are rejected early with clear error messages, before any model call is made.
- **Strict duplicate-key detection**: a custom `object_pairs_hook` (`check_keys`) rejects JSON files with duplicated keys instead of silently keeping the last one, catching a common source of malformed input files.
- **Graceful degradation**: `parsing()` never raises; every expected failure mode (missing file, permission error, invalid JSON, schema violation) is caught, reported, and turned into an empty result so the caller can exit cleanly instead of crashing.

## Performance Analysis

- **Accuracy**: Function-name and parameter-type masking make it structurally impossible to select an unknown function or a value of the wrong type, so schema compliance is effectively 100% by construction rather than by chance. Correct *argument extraction* (i.e., choosing the right numeric value or string) still depends on the underlying model's language understanding.
- **Reliability**: The output is always syntactically valid JSON, since punctuation is injected rather than generated — there is no failure mode where the file fails to parse.
- **Speed**: Generation is done one token at a time with a single forward pass per token (`get_logits_from_input_ids`), so total runtime scales linearly with the number of prompts and the average number of tokens per parameter. A safety-net token cap (`len(prompt) + 10`) prevents any single prompt from stalling generation indefinitely.

## Challenges Faced

- **Masking without breaking valid tokens**: character-level classification of the vocabulary (e.g., a token containing only digits and commas) had to be broad enough to allow multi-character numeric tokens (e.g., `"12"`, `"34,"`) while still excluding anything that could break the schema.
- **Knowing when a parameter value is "done"**: since token boundaries don't align with JSON punctuation, the raw parameter value has to be accumulated across multiple tokens and only converted to its final type once a delimiter token (`,` or `}`) appears.
- **Avoiding infinite generation loops**: a misbehaving model could in theory keep generating value tokens forever for a single parameter; a token-count safety net forces the current parameter to close after a bounded number of tokens.
- **Function-name ambiguity during generation**: since several function names can share a common prefix, the decoder needs to keep the *set* of remaining candidate names consistent with each new token, rather than committing to one name early.

## Testing Strategy

- **Input validation testing**: `parsing.py` was exercised against malformed files — missing keys, duplicated keys, wrong types, unsupported parameter types, non-list/non-dict structures, empty prompts — to confirm every failure mode raises a clear, descriptive `ValueError` and that `parsing()` degrades gracefully instead of crashing.
- **Schema compliance testing**: output JSON files were validated with `json.load` to confirm they always parse, and checked against `functions_definition.json` to confirm parameter keys and types match exactly.
- **Edge cases**: prompts with large numbers, special characters in strings, boolean-valued parameters, functions with multiple parameters, and functions with no parameters were used to confirm the parameter-generation loop terminates correctly in every case.
- **Manual inspection**: the `print(value)` / `print(string)` trace in the generation loop was used during development to inspect the token-by-token construction of the output and catch masking issues early.

## Example Usage

Given `data/input/functions_definition.json`:

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
    "returns": {"type": "number"}
  },
  {
    "name": "fn_greet",
    "description": "Generate a greeting message for a person by name.",
    "parameters": {"name": {"type": "string"}},
    "returns": {"type": "string"}
  }
]
```

And `data/input/function_calling_tests.json`:

```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Greet shrek"}
]
```

Running:

```bash
uv run python -m src
```

Produces `data/output/function_calling_results.json`:

```json
[
  {"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}},
  {"prompt": "Greet shrek", "name": "fn_greet", "parameters": {"name": "shrek"}}
]
```

## Resources
- Function Calling?: https://finetunedb.com/blog/what-is-function-calling-simply-explained/
- LLM Visualization: https://bbycroft.net/llm
- LLM Generation Road: https://blog.langformers.com/how-llms-work/


### AI Usage

AI assistance (Claude) was used for:
- Adding PEP 257–style docstrings (summary, `Args`, `Returns`, `Raises`) to every function in `main.py` and `parsing.py`, after the core logic had already been implemented and tested.
- Drafting and structuring this README
- Explaining some core concepts, like the steps that goes on in the LLM after given the text input.