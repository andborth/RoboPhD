#!/usr/bin/env bash
#
# Full test sweep: one pytest process per suite.
#
# The suites cannot share a process. Each example ships a flat
# evaluator.py / main.py / model_registry.py and imports them by bare
# name after putting its own directory on sys.path, so in a shared
# process the first suite to import wins `sys.modules` and later suites
# silently test the wrong example's code. Their unit_tests/ directories
# also collide as a package named `unit_tests`, and forcing collection
# past that deadlocks on a multiprocessing-based test.
#
# `pytest.ini` therefore scopes a bare `pytest` to RoboPhD/unit_tests.
# This script is what runs everything. Extra args are forwarded to each
# pytest invocation, e.g. `scripts/run_tests.sh -x -q`.
set -uo pipefail
cd "$(dirname "$0")/.."

suites=("RoboPhD/unit_tests")
for dir in examples/*/unit_tests; do
    [ -d "$dir" ] && suites+=("$dir")
done

failed=()
for suite in "${suites[@]}"; do
    printf '\n\033[1m=== %s ===\033[0m\n' "$suite"
    python -m pytest "$suite" "$@" || failed+=("$suite")
done

printf '\n========================================\n'
if [ ${#failed[@]} -eq 0 ]; then
    printf 'All %d suites passed.\n' "${#suites[@]}"
else
    printf 'FAILED %d/%d suites:\n' "${#failed[@]}" "${#suites[@]}"
    printf '  %s\n' "${failed[@]}"
    exit 1
fi
