# =============================================================================
# Vendored Can't Be Late utilities
#
# Source: https://github.com/gepa-ai/gepa/tree/main/examples/adrs/can_be_late/utils
# Fetched: 2026-03-11
#
# This package contains dataset loading and simulation logic for the
# "Can't Be Late" cloud scheduling problem (NSDI'24).
#
# simulation.py has one modification: removed `from gepa.optimize_anything
# import SideInfo` (SideInfo is just `dict`). All other files are exact copies.
# =============================================================================

from .dataset import load_trace_dataset
from .simulation import (
    FAILED_SCORE,
    get_program_path,
    syntax_is_valid,
    syntax_failure_info,
    run_simulation,
    simulation_failure_info,
    simulation_success_info,
)
