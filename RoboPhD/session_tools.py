"""Materialize task-shipped session helper scripts into a run directory."""
import shutil
from pathlib import Path
from typing import List, Optional


def materialize_session_tools(
    run_root: Path, paths: Optional[List[str]]
) -> None:
    """Copy helper scripts into ``<run_root>/session_tools/``.

    The directory sits inside the evolution sessions' read scope but
    outside their write root, so sessions can run the scripts but not
    modify them. Copies overwrite on every startup so resumed runs pick
    up repo-side fixes; a missing source file is a hard error rather
    than a silently thinner toolset. No-op when ``paths`` is empty.
    """
    if not paths:
        return
    dest_dir = Path(run_root) / "session_tools"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        src = Path(p)
        if not src.is_file():
            raise FileNotFoundError(
                f"session_tools entry does not exist or is not a file: {p}"
            )
        shutil.copy2(src, dest_dir / src.name)
