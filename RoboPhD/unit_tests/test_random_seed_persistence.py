"""Tests for run-seed persistence and resume continuity.

A run's identity is the seed it was LAUNCHED with. Until 2026-08-26 that value was
never written down: `_save_checkpoint` stored `self.random_seed`, which the
per-iteration reseed had already overwritten with `(original + iteration * 10000)`.
Two things followed -- a finished run could not report its own seed, and `--resume`
read the derived value back as the original, silently re-basing the RNG so a resumed
run drew from a stream no uninterrupted run would produce.

The contract these tests pin: `original_seed` is persisted, resume derives its seed
from it, and the iteration a resume starts on draws exactly what it would have drawn
had the run never stopped.
"""

import ast
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from RoboPhD.researcher import (  # noqa: E402
    ParallelAgentResearcher,
    SEED_ITERATION_STRIDE,
    _iteration_seed,
    _recover_original_seed,
)

resolve = ParallelAgentResearcher._resolve_original_seed


# --- derivation ------------------------------------------------------------

def test_iteration_seed_is_stride_spaced_and_wraps():
    assert _iteration_seed(218170, 0) == 218170
    assert _iteration_seed(218170, 19) == 218170 + 19 * SEED_ITERATION_STRIDE == 408170
    assert _iteration_seed(2**32 - 1, 1) == (2**32 - 1 + SEED_ITERATION_STRIDE) % (2**32)


@pytest.mark.parametrize("original,iteration", [(0, 0), (8170, 21), (218170, 19), (2**32 - 5, 3)])
def test_recovery_inverts_derivation(original, iteration):
    assert _recover_original_seed(_iteration_seed(original, iteration), iteration) == original


# --- resolution on resume --------------------------------------------------

def test_original_seed_is_used_when_present():
    ckpt = {"original_seed": 218170, "random_seed": _iteration_seed(218170, 19)}
    assert resolve(ckpt, 19) == 218170


def test_legacy_checkpoint_recovers_and_warns(caplog):
    """The real case: asta_ds1000_20260826_091106 stored 408170 after 19 iterations,
    having been launched with --random-seed 218170."""
    with caplog.at_level("WARNING"):
        assert resolve({"random_seed": 408170}, 19) == 218170
    assert "218170" in caplog.text and "408170" in caplog.text


def test_disagreeing_seed_fields_warn_but_prefer_original(caplog):
    """A run resumed under the pre-fix behavior carries a random_seed that no longer
    matches its original_seed. original_seed stays the authority."""
    with caplog.at_level("WARNING"):
        assert resolve({"original_seed": 2866, "random_seed": 312866}, 21) == 2866
    assert "disagree" in caplog.text


def test_consistent_fields_do_not_warn(caplog):
    with caplog.at_level("WARNING"):
        resolve({"original_seed": 8170, "random_seed": _iteration_seed(8170, 21)}, 21)
    assert caplog.text == ""


def test_missing_both_fields_raises():
    with pytest.raises(ValueError, match="original_seed and random_seed"):
        resolve({"last_completed_iteration": 4}, 4)


# --- the requirement: a resume continues the uninterrupted stream -----------

POOL = [f"problem_{i}" for i in range(100)]


def _batch_at(original_seed: int, iteration: int, n: int = 20):
    """Reproduce what an iteration draws: reseed, then sample the context pool.
    Mirrors researcher.py's reseed-then-`random.sample(self.contexts, ...)` order."""
    random.seed(_iteration_seed(original_seed, iteration))
    return sorted(random.sample(POOL, n))


def test_resume_draws_what_the_uninterrupted_run_would_have_drawn():
    original = 218170
    uninterrupted = {i: _batch_at(original, i) for i in range(1, 22)}

    # Interrupt after 10, resume: original_seed comes from the checkpoint, and the
    # seed for iteration 11 is re-derived rather than continued from the stored value.
    ckpt = {"original_seed": original, "random_seed": _iteration_seed(original, 10)}
    resumed_original = resolve(ckpt, 10)
    for i in range(11, 22):
        assert _batch_at(resumed_original, i) == uninterrupted[i], f"iteration {i} diverged"


def test_pre_fix_resume_would_have_diverged():
    """Guards the regression: reading random_seed back as the original re-bases the
    stream. If this ever stops differing, the derivation has been flattened."""
    original = 218170
    shifted = _iteration_seed(original, 10)          # what the old code adopted
    assert _batch_at(shifted, 11) != _batch_at(original, 11)


def test_from_iteration_rewind_replays_that_iterations_draw():
    """--from-iteration N re-runs N, so it must reproduce N's original batch."""
    original = 8170
    first_pass = _batch_at(original, 7)
    ckpt = {"original_seed": original, "random_seed": _iteration_seed(original, 12)}
    assert _batch_at(resolve(ckpt, 12), 7) == first_pass


# --- persistence and wiring (structural, not source-substring) --------------
#
# These check the SHAPE of the code, not its text. The substring form -- matching
# "'original_seed': self.original_seed," and friends -- fails on a dict reordering,
# a quote-style change or a line wrap, none of which alter behavior. This repo has
# rewritten that pattern twice for the same reason (cad0f71's tar guard matched an
# ast.unparse substring and broke when the gate got stricter; d535df6d's seed_agents
# guard walks the IfExp instead).

_TREE = ast.parse((Path(__file__).parent.parent / "researcher.py").read_text())


_RESEARCHER = next(n for n in _TREE.body
                   if isinstance(n, ast.ClassDef) and n.name == "ParallelAgentResearcher")


def _func(name: str) -> ast.FunctionDef:
    """A method of ParallelAgentResearcher by name.

    Scoped to the class: researcher.py defines several (MemoryMonitor,
    StuckProcessReaper, ParallelAgentEvolver), so a module-wide walk for
    "__init__" finds the wrong one.
    """
    return next(n for n in ast.walk(_RESEARCHER)
                if isinstance(n, ast.FunctionDef) and n.name == name)


def _assignments_to(func: ast.FunctionDef, attr: str) -> list:
    return [n for n in ast.walk(func) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Attribute) and t.attr == attr for t in n.targets)]


def _reads_checkpoint_directly(node: ast.AST) -> bool:
    """True if the expression pulls a value straight out of resume_checkpoint."""
    for n in ast.walk(node):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "get"
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "resume_checkpoint"):
            return True
        if (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                and n.value.id == "resume_checkpoint"):
            return True
    return False


def test_checkpoint_persists_the_launch_seed_and_keeps_the_legacy_field():
    """Without original_seed a finished run cannot report the seed needed to
    reproduce or seed-pair it. random_seed stays because the resume validation and
    external readers expect the key."""
    save = _func("_save_checkpoint")
    ckpt = next(d for d in ast.walk(save) if isinstance(d, ast.Dict)
                and any(isinstance(k, ast.Constant) and k.value == "last_completed_iteration"
                        for k in d.keys))
    pairs = {k.value: v for k, v in zip(ckpt.keys, ckpt.values)
             if isinstance(k, ast.Constant)}
    assert "original_seed" in pairs, "the launch seed is not persisted"
    assert "random_seed" in pairs, "the legacy derived field was dropped"
    bound = pairs["original_seed"]
    assert isinstance(bound, ast.Attribute) and bound.attr == "original_seed", (
        "original_seed must be bound to self.original_seed, not to a derived value"
    )


def test_resume_resolves_the_launch_seed_instead_of_reading_the_derived_field():
    """The defect: self.original_seed = resume_checkpoint.get('random_seed'), which
    re-bases the RNG on every resume."""
    init = _func("__init__")
    assigns = _assignments_to(init, "original_seed")
    assert assigns, "__init__ never sets original_seed"
    assert any(isinstance(a.value, ast.Call)
               and getattr(a.value.func, "attr", "") == "_resolve_original_seed"
               for a in assigns), "the resume path must go through the resolver"
    for a in assigns:
        assert not _reads_checkpoint_directly(a.value), (
            "original_seed must never be read straight out of the checkpoint -- that "
            "field holds the derived per-iteration value"
        )


def test_the_stride_is_never_re_inlined():
    """Anti-drift guard, same shape as asta_ds1000's
    test_validators_are_not_reimplemented_locally. Three sites share this arithmetic
    and an inverse that drifts from its forward function fails silently rather than
    loudly, so the literal lives in exactly one place."""
    holders = {n.targets[0].id for n in _TREE.body
               if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
               and isinstance(n.value, ast.Constant) and n.value.value == SEED_ITERATION_STRIDE}
    assert "SEED_ITERATION_STRIDE" in holders

    for fn in ("__init__", "run", "_save_checkpoint"):
        node = _func(fn)   # raises if the method is renamed -- deliberate, not skipped
        for n in ast.walk(node):
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
                operands = [n.left, n.right]
                assert not any(isinstance(o, ast.Constant) and o.value == SEED_ITERATION_STRIDE
                               for o in operands), (
                    f"{fn} re-inlines the stride; call _iteration_seed instead"
                )


def test_derivation_and_recovery_stay_mutually_inverse():
    """The forward function and its inverse are only correct relative to each other."""
    for original in (0, 1, 8170, 218170, 2**32 - 1):
        for iteration in (0, 1, 19, 21, 999):
            derived = _iteration_seed(original, iteration)
            assert 0 <= derived < 2**32
            assert _recover_original_seed(derived, iteration) == original
