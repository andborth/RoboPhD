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


# --- persistence -----------------------------------------------------------

def test_checkpoint_writes_original_seed_alongside_the_derived_value():
    src = (Path(__file__).parent.parent / "researcher.py").read_text()
    assert "'original_seed': self.original_seed," in src, (
        "the launch seed must be persisted; without it a finished run cannot report "
        "the seed needed to reproduce or seed-pair it"
    )
    assert "'random_seed': self.random_seed," in src, (
        "keep the legacy field: external readers and older tooling expect the key"
    )


def test_resume_derives_from_original_not_from_the_stored_value():
    src = (Path(__file__).parent.parent / "researcher.py").read_text()
    assert "self.random_seed = _iteration_seed(self.original_seed, current_iteration)" in src
    assert "self.original_seed = self._resolve_original_seed(" in src
    assert "self.original_seed = resume_checkpoint.get(\"random_seed\")" not in src, (
        "the defect being fixed: reading the derived value back as the original"
    )
