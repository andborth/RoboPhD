"""Tests for transcript_summarizer's session-flow labels: run-root
normalization ($RUN), head+tail truncation, Edit signatures, Read spans."""

import json

from RoboPhD.utilities.transcript_summarizer import (
    BASH_LABEL_MAX,
    _derive_run_root,
    _normalize_command,
    _shorten_path,
    _tool_one_liner,
    _truncate_head_tail,
    summarize_transcript,
)

RUN = "/x/robophd_runs/robophd/task_20260721_215631"


def test_derive_run_root_from_evolution_output_layout(tmp_path):
    out = tmp_path / "run" / "evolution_output" / "iteration_013" / "session_summary.md"
    assert _derive_run_root(out) == str(tmp_path / "run")


def test_derive_run_root_meta_and_no_marker(tmp_path):
    meta = tmp_path / "run" / "meta_evolution_output" / "iteration_004" / "s.md"
    assert _derive_run_root(meta) == str(tmp_path / "run")
    assert _derive_run_root(tmp_path / "elsewhere" / "s.md") is None


def test_shorten_path():
    assert _shorten_path(f"{RUN}/agents/iter9/agent.py", RUN) == "$RUN/agents/iter9/agent.py"
    assert _shorten_path(RUN, RUN) == "$RUN"
    assert _shorten_path("/other/place.py", RUN) == "/other/place.py"
    assert _shorten_path("/other/place.py", None) == "/other/place.py"


def test_normalize_command_strips_noise():
    cmd = f"R={RUN}; /opt/anaconda3/envs/demo/bin/python {RUN}/x.py | head"
    out = _normalize_command(cmd, RUN)
    assert out == "R=$RUN; python $RUN/x.py | head"


def test_truncate_head_tail_keeps_both_ends():
    text = "A" * 300 + "TAIL_MARKER"
    out = _truncate_head_tail(text, 200, 40)
    assert len(out) == 200
    assert out.startswith("A")
    assert out.endswith("TAIL_MARKER")
    assert " … " in out
    assert _truncate_head_tail("short", 200, 40) == "short"


def test_bash_label_spends_budget_on_payload_not_prefix():
    """The regression that motivated this: a run-root prefix ate the whole
    budget and the informative tail was cut."""
    cmd = f"cd {RUN}/evolution_output && ls && cat iteration_012/reasoning.md | head -50"
    label = _tool_one_liner("Bash", {"command": cmd}, run_root=RUN)
    assert "cat iteration_012/reasoning.md | head -50" in label
    assert RUN not in label


def test_bash_description_still_preferred():
    label = _tool_one_liner("Bash", {"command": "x" * 500, "description": "run tests"},
                            run_root=RUN)
    assert label == "→ Bash: run tests"


def test_edit_signature_first_lines():
    label = _tool_one_liner(
        "Edit",
        {"file_path": f"{RUN}/evolution_output/iteration_013/agent.py",
         "old_string": "META_MAX_AUTHORS = 3\nMORE", "new_string": "META_MAX_AUTHORS = 6\nMORE"},
        run_root=RUN,
    )
    assert label == ('→ Edit $RUN/evolution_output/iteration_013/agent.py: '
                     '"META_MAX_AUTHORS = 3" → "META_MAX_AUTHORS = 6"')


def test_edit_signature_uses_first_differing_line():
    """Shared leading context lines would make every signature in a block
    of edits identical — the differing pair is the signal."""
    label = _tool_one_liner(
        "Edit",
        {"file_path": "f.py",
         "old_string": "SHARED = 1  # context\nknob = 3",
         "new_string": "SHARED = 1  # context\nknob = 6"},
        run_root=None,
    )
    assert label == '→ Edit f.py: "knob = 3" → "knob = 6"'


def test_edit_signature_pure_insertion():
    label = _tool_one_liner(
        "Edit",
        {"file_path": "f.py",
         "old_string": "anchor line",
         "new_string": "anchor line\nadded line"},
        run_root=None,
    )
    assert label == '→ Edit f.py: "anchor line" → "added line"'


def test_edit_snippets_truncated():
    label = _tool_one_liner("Edit", {"file_path": "f.py", "old_string": "x" * 100,
                                     "new_string": "y"}, run_root=None)
    assert "x" * 39 + "…" in label
    assert "x" * 41 not in label


def test_read_span_annotations():
    assert _tool_one_liner("Read", {"file_path": "f.py", "offset": 100, "limit": 50}) \
        == "→ Read f.py (lines 100–149)"
    assert _tool_one_liner("Read", {"file_path": "f.py", "offset": 7}) \
        == "→ Read f.py (from line 7)"
    assert _tool_one_liner("Read", {"file_path": "f.py"}) == "→ Read f.py"


def test_end_to_end_summary_uses_run_legend(tmp_path):
    run = tmp_path / "myrun"
    out_dir = run / "evolution_output" / "iteration_003"
    out_dir.mkdir(parents=True)
    agent_path = str(out_dir / "agent.py")

    def asst(*blocks):
        return {"type": "assistant", "timestamp": "2026-07-22T10:00:00.000Z",
                "message": {"model": "claude-opus-4-8",
                            "usage": {"input_tokens": 1, "output_tokens": 2},
                            "content": list(blocks)}}

    transcript = tmp_path / "t.jsonl"
    transcript.write_text("\n".join(json.dumps(m) for m in [
        asst({"type": "text", "text": "Starting."},
             {"type": "tool_use", "name": "Bash",
              "input": {"command": f"cd {run} && ls"}}),
        asst({"type": "tool_use", "name": "Edit",
              "input": {"file_path": agent_path, "old_string": "a = 1",
                        "new_string": "a = 2"}}),
    ]))

    summary = summarize_transcript(transcript, out_dir / "session_summary.md").read_text()
    assert f"- **$RUN**: {run}" in summary
    assert "cd $RUN && ls" in summary
    assert '→ Edit $RUN/evolution_output/iteration_003/agent.py: "a = 1" → "a = 2"' in summary
    assert str(run) + "/evolution_output/iteration_003/agent.py\n" not in summary.replace(" (edited)", "")


def test_no_marker_layout_leaves_paths_untouched(tmp_path):
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(json.dumps({
        "type": "assistant", "timestamp": "2026-07-22T10:00:00.000Z",
        "message": {"model": "m", "usage": {},
                    "content": [{"type": "tool_use", "name": "Bash",
                                 "input": {"command": "ls /abs/path"}}]}}))
    summary = summarize_transcript(transcript, tmp_path / "s.md").read_text()
    assert "$RUN" not in summary
    assert "ls /abs/path" in summary
