"""
External evaluator domain for RoboPhD.

Wraps evaluator functions (from GEPA or other frameworks) as a RoboPhD domain,
allowing RoboPhD's evolution loop to optimize candidates using any evaluator
that follows the (candidate, example) -> (score, diagnostics) interface.
"""
