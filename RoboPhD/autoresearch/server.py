"""
HTTP evaluation server for autoresearch.

Runs in a thread inside run_autoresearch.py. The agent's _evaluate.py CLI
POSTs to this server to score candidates.
"""

import json
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Callable, Dict, List

logger = logging.getLogger(__name__)


class EvalServer:
    """Threaded HTTP server wrapping a task evaluator with budget tracking."""

    def __init__(
        self,
        evaluator: Callable,
        candidate_fn: Callable[[], dict],
        train_examples: List[dict],
        val_examples: List[dict],
        evaluation_budget: int,
        workspace: Path,
    ):
        self.evaluator = evaluator
        self.candidate_fn = candidate_fn  # callable returning current candidate dict
        self.train_examples = {ex["id"]: ex for ex in train_examples}
        self.val_examples = val_examples
        self.budget_remaining = evaluation_budget
        self.budget_total = evaluation_budget
        self.workspace = workspace
        self._lock = threading.Lock()

        handler = _make_handler(self)
        self._httpd = HTTPServer(("localhost", 0), handler)
        self.port = self._httpd.server_address[1]

        # Write port file
        (workspace / "_server_port").write_text(str(self.port))
        self._write_budget()

    def _write_budget(self):
        budget = {
            "budget_remaining": self.budget_remaining,
            "budget_total": self.budget_total,
            "budget_used": self.budget_total - self.budget_remaining,
        }
        (self.workspace / "_budget.json").write_text(json.dumps(budget, indent=2))

    def _consume_budget(self, n: int) -> bool:
        """Consume n budget. Returns True if budget was available (or in-flight completion)."""
        with self._lock:
            if self.budget_remaining <= 0:
                return False
            self.budget_remaining -= n
            self._write_budget()
            return True

    def start(self):
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        logger.info(f"Eval server started on port {self.port}")

    def stop(self):
        self._httpd.shutdown()
        logger.info("Eval server stopped")


def _make_handler(server: EvalServer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # suppress request logs

        def _send_json(self, data: dict, status: int = 200):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self) -> dict:
            length = int(self.headers.get("Content-Length", 0))
            if length == 0:
                return {}
            return json.loads(self.rfile.read(length))

        def do_GET(self):
            if self.path == "/budget":
                self._send_json({"budget_remaining": server.budget_remaining})
            else:
                self._send_json({"error": "not found"}, 404)

        def do_POST(self):
            if self.path == "/eval/train":
                self._handle_train()
            elif self.path == "/eval/val":
                self._handle_val()
            else:
                self._send_json({"error": "not found"}, 404)

        def _handle_train(self):
            body = self._read_body()
            example_ids = body.get("example_ids", [])
            if not example_ids:
                self._send_json({"error": "example_ids required"}, 400)
                return

            cost = len(example_ids)
            if not server._consume_budget(cost):
                self._send_json({
                    "error": "budget_exhausted",
                    "budget_remaining": server.budget_remaining,
                }, 429)
                return

            candidate = server.candidate_fn()
            scores = {}
            diagnostics = {}
            budget_exhausted = server.budget_remaining < 0

            for eid in example_ids:
                if eid not in server.train_examples:
                    scores[eid] = 0.0
                    diagnostics[eid] = {"error": f"unknown example id: {eid}"}
                    continue
                example = server.train_examples[eid]
                try:
                    score, diag = server.evaluator(candidate, example)
                    scores[eid] = score
                    diagnostics[eid] = diag
                except Exception as e:
                    logger.warning(f"Evaluator error on {eid}: {e}")
                    scores[eid] = 0.0
                    diagnostics[eid] = {"error": str(e)}

            resp = {
                "scores": scores,
                "diagnostics": diagnostics,
                "budget_remaining": server.budget_remaining,
            }
            if budget_exhausted:
                resp["budget_exhausted"] = True
            self._send_json(resp)

        def _handle_val(self):
            cost = len(server.val_examples)
            if not server._consume_budget(cost):
                self._send_json({
                    "error": "budget_exhausted",
                    "budget_remaining": server.budget_remaining,
                }, 429)
                return

            candidate = server.candidate_fn()
            total_score = 0.0
            count = 0
            budget_exhausted = server.budget_remaining < 0

            for example in server.val_examples:
                try:
                    score, _ = server.evaluator(candidate, example)
                    total_score += score
                except Exception as e:
                    logger.warning(f"Evaluator error on val example: {e}")
                count += 1

            mean_score = total_score / count if count else 0.0
            resp = {
                "mean_score": mean_score,
                "budget_remaining": server.budget_remaining,
            }
            if budget_exhausted:
                resp["budget_exhausted"] = True
            self._send_json(resp)

    return Handler
