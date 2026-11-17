# Config-only evolution: the agent's code is frozen

In this run the agent's source code is fixed. It is the file `${HARNESS_FILE}` (read-only), and every candidate runs exactly that code. What you create is one file, `${CONFIG_FILE}`: a JSON document giving a value for each of the frozen agent's settings. Nothing but `${CONFIG_FILE}` is read from your output; an `agent.py` you write is ignored.

## The config

Two sections, each with a fixed set of keys — the ones in the config you start from. A key can be neither added nor removed.

- `"models"` — one entry per model slot of the frozen agent. A slot is `{"model": <handle>, "effort": <effort>}`. `model` is one of the handle names in the table under "LLM calls" below. `effort` is `null` for that handle's default reasoning, or one of the handle's available overrides from the same table; it applies to every call the frozen agent makes through that slot. A slot whose value is a list holds any number of such entries, including none.
- `"constants"` — one entry per numeric or boolean constant. Each keeps its type: an integer stays an integer, a boolean stays a boolean, and a non-integer number may take any finite value.

Each value replaces the right-hand side of the assignment of the same name at the top level of `${HARNESS_FILE}`. Everything else in that file — prompts and other strings, values written inside functions, and all code — is the same for every candidate.

A config that is not valid (malformed JSON, a missing or unknown key, a value of the wrong type, a handle that is not in the table, an effort the handle does not offer) scores 0 on every query without the agent being run. The reason is in that problem's `error.md`.

## Checking a config

- `python ${HARNESS_TOOL} validate ${CONFIG_FILE}` reports whether a config is valid, and if not, why.
- `python ${HARNESS_TOOL} render ${CONFIG_FILE}` prints the agent source the config produces.

## Reading the rest of this document

The sections below describe the task, the tools, the models, the scoring and the diagnostics. They apply unchanged to the frozen agent; where they describe how agent code is written, they describe the environment the frozen code runs in.
