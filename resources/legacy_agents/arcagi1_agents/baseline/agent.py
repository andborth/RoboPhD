import json, re

def solve(train_inputs, train_outputs, test_inputs, llm):
    # --- Demonstration: print() output is captured as agent_stdout in diagnostics ---
    print(f"Training examples: {len(train_inputs)}, Test inputs: {len(test_inputs)}")

    training_examples = "\n".join(f"Input: {i}\nOutput: {o}" for i, o in zip(train_inputs, train_outputs))
    problem_inputs = "\n".join(f"Input {i}: {x}" for i, x in enumerate(train_inputs + test_inputs))

    prompt = f"Solve an ARC AGI puzzle. Training examples:\n{training_examples}\n\nPredict output for EACH input as JSON [[...]]:\n{problem_inputs}"
    response = llm(prompt)

    grids = [json.loads(g) for g in re.findall(r"\[\[.*?\]\]", response.replace("\n", ""))]
    n_train = len(train_inputs)

    # --- Demonstration: print() output is captured as agent_stdout in diagnostics ---
    print(f"Grids parsed: {len(grids)} (expected {n_train + len(test_inputs)})")

    return {
        "train": grids[:n_train],
        "test": [[g] for g in grids[n_train:]]
    }
