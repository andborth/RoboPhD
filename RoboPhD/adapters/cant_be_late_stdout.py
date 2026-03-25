"""
Can't Be Late adapter with stdout capture from agent subprocess.

Subclasses CantBeLateEvaluator to surface subprocess stdout as a
diagnostic. This lets the evolution AI design its own diagnostics
by adding print() statements to the agent's _step() method.

Results are non-comparable with the base cant_be_late task.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from RoboPhD.adapters.cant_be_late import (
    CantBeLateEvaluator,
    CANT_BE_LATE_FILE_MAPPING,
    OBJECTIVE,
)
from RoboPhD.adapters.cant_be_late_constants_unmodified import OPTIMIZATION_BACKGROUND


BACKGROUND = OPTIMIZATION_BACKGROUND + """

Diagnostics: Any print() output from the agent's _step() method is \
captured and included in evaluation diagnostics as agent_stdout. Use \
print() to log any information you think would be helpful for you to \
see in improving the agent in later rounds of testing and refinement."""


class CantBeLateStdoutEvaluator(CantBeLateEvaluator):
    """CantBeLateEvaluator with stdout capture from agent subprocess.

    Reads proc_stdout from the parent's _last_details after super().__call__()
    and adds it as agent_stdout in diagnostics.
    """

    def __call__(
        self,
        candidate: Dict[str, str],
        example: Dict[str, Any],
        *,
        problem_dir: Optional[Path] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        self._last_details_local.details = {}  # Reset before parent call
        score, diagnostics = super().__call__(candidate, example, problem_dir=problem_dir)

        details = getattr(self._last_details_local, "details", {})
        proc_stdout = details.get("proc_stdout", "")
        # Filter out simulator cost summary line (format: "mean: <float>; std: <float>; ...")
        agent_lines = [
            line for line in proc_stdout.splitlines()
            if not (line.startswith("mean: ") and "; std: " in line)
        ]
        agent_stdout = "\n".join(agent_lines).strip()
        if agent_stdout:
            diagnostics["agent_stdout"] = agent_stdout

        return score, diagnostics
