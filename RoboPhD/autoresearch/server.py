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

from RoboPhD.eval_utils import run_parallel_eval

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
        max_workers: int | None = None,
        eval_timeout: int = 300,
    ):
        self.evaluator = evaluator
        self.candidate_fn = candidate_fn  # callable returning current candidate dict
        self.train_examples = {ex["id"]: ex for ex in train_examples}
        self.val_examples = val_examples
        self.budget_remaining = evaluation_budget
        self.budget_total = evaluation_budget
        self.workspace = workspace
        self.max_workers = max_workers
        self.eval_timeout = eval_timeout
        self._lock = threading.Lock()
        self._last_logged_pct = 100  # track 5% increments (start at 100%)

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

    def _consume_budget(self, n: int) -> tuple[bool, bool]:
        """Consume n budget.

        Returns (allowed, overdrawn):
            allowed: True if budget was > 0 before this call (request proceeds).
            overdrawn: True if budget is now <= 0 after this call (no future requests).
        """
        with self._lock:
            if self.budget_remaining <= 0:
                return False, True
            self.budget_remaining -= n
            overdrawn = self.budget_remaining <= 0
            self._write_budget()
            self._maybe_log_budget()
            return True, overdrawn

    def _maybe_log_budget(self):
        """Log when budget crosses a 5% threshold. Must be called under _lock."""
        pct = (self.budget_remaining / self.budget_total * 100) if self.budget_total > 0 else 0
        # Find the highest 5% mark at or below current pct
        current_mark = int(pct // 5) * 5
        if current_mark < self._last_logged_pct:
            used = self.budget_total - self.budget_remaining
            logger.info(
                f"Budget: {self.budget_remaining}/{self.budget_total} remaining "
                f"({used} used, {pct:.0f}%)"
            )
            self._last_logged_pct = current_mark

    def _run_eval(self, candidate: dict, examples: list[dict]) -> dict:
        """Evaluate examples using shared parallel infrastructure."""
        return run_parallel_eval(
            self.evaluator, candidate, examples,
            max_workers=self.max_workers,
            eval_timeout=self.eval_timeout,
        )

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
            allowed, overdrawn = server._consume_budget(cost)
            if not allowed:
                self._send_json({
                    "error": "budget_exhausted",
                    "budget_remaining": server.budget_remaining,
                }, 429)
                return

            candidate = server.candidate_fn()

            # Resolve example IDs, catching unknown ones
            unknown_scores = {}
            unknown_diags = {}
            valid_examples = []
            for eid in example_ids:
                if eid not in server.train_examples:
                    unknown_scores[eid] = 0.0
                    unknown_diags[eid] = {"error": f"unknown example id: {eid}"}
                else:
                    valid_examples.append(server.train_examples[eid])

            if valid_examples:
                result = server._run_eval(candidate, valid_examples)
            else:
                result = {"scores": [], "diagnostics": []}

            # Merge results back keyed by example ID
            scores = dict(unknown_scores)
            diagnostics = dict(unknown_diags)
            for i, ex in enumerate(valid_examples):
                scores[ex["id"]] = result["scores"][i]
                diagnostics[ex["id"]] = result["diagnostics"][i]

            mean = sum(scores.values()) / len(scores) if scores else 0.0
            logger.info(
                f"Train eval: {len(example_ids)} examples, "
                f"mean={mean:.3f}, budget={server.budget_remaining}"
            )

            resp = {
                "scores": scores,
                "diagnostics": diagnostics,
                "budget_remaining": server.budget_remaining,
            }
            if overdrawn:
                resp["budget_exhausted"] = True
            self._send_json(resp)

        def _handle_val(self):
            cost = len(server.val_examples)
            allowed, overdrawn = server._consume_budget(cost)
            if not allowed:
                self._send_json({
                    "error": "budget_exhausted",
                    "budget_remaining": server.budget_remaining,
                }, 429)
                return

            candidate = server.candidate_fn()
            result = server._run_eval(candidate, server.val_examples)

            mean_score = result["test_results"]["mean_test_score"]
            count = result["test_results"]["total_test_problems"]
            logger.info(
                f"Val eval: {count} examples, "
                f"mean={mean_score:.3f}, budget={server.budget_remaining}"
            )

            resp = {
                "mean_score": mean_score,
                "budget_remaining": server.budget_remaining,
            }
            if overdrawn:
                resp["budget_exhausted"] = True
            self._send_json(resp)

    return Handler
