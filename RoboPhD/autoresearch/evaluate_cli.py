#!/usr/bin/env python3
"""
Evaluator CLI — copied into workspace as _evaluate.py.

The autoresearch agent calls this to evaluate candidates against the
runner's HTTP evaluation server.

Usage:
    python _evaluate.py train <id> [<id> ...]   # score on training examples
    python _evaluate.py val                       # mean score on held-out validation set
    python _evaluate.py budget                    # remaining budget (no cost)
"""

import json
import sys
import urllib.request


def _server_url():
    with open("_server_port") as f:
        port = int(f.read().strip())
    return f"http://localhost:{port}"


def _post(path, data=None):
    url = f"{_server_url()}{path}"
    body = json.dumps(data or {}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _get(path):
    url = f"{_server_url()}{path}"
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())


def main():
    if len(sys.argv) < 2:
        print("Usage: python _evaluate.py {train <id>...|val|budget}", file=sys.stderr)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "train":
        if len(sys.argv) < 3:
            print("Usage: python _evaluate.py train <id> [<id> ...]", file=sys.stderr)
            sys.exit(1)
        ids = sys.argv[2:]
        result = _post("/eval/train", {"example_ids": ids})
    elif cmd == "val":
        result = _post("/eval/val")
    elif cmd == "budget":
        result = _get("/budget")
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
