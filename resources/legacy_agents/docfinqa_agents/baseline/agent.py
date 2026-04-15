import re


def answer(document, question, llm, embed):
    """Answer a financial question over a full SEC filing."""
    # Chunk the document
    chunks, size, overlap = [], 2000, 400
    words = document.split()
    for i in range(0, len(words), size - overlap):
        chunks.append(" ".join(words[i:i+size]))

    # Retrieve the most relevant chunk
    q_emb = embed(question)
    best, best_score = chunks[0], -1
    for c in chunks:
        c_emb = embed(c)
        score = sum(a * b for a, b in zip(q_emb, c_emb))
        if score > best_score:
            best, best_score = c, score

    # Generate a Python program that computes the answer
    program = llm(
        f"Context:\n{best}\n\n"
        f"Question: {question}\n\n"
        f"Write a short Python program using named variables.\n"
        f"The last line must assign the result to `answer`.\n"
        f"Program:"
    )
    return program.strip()
