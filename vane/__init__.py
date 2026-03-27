"""
VANE — Velocity, Acceleration, and Nonlinearity Estimation.

A framework for detecting LLM reasoning failures via geometric
analysis of hidden-state trajectories across transformer layers.
"""

from .metrics import (
    compute_metrics,
    build_features,
    build_features_full,
    get_ablation_features,
    METRIC_GROUPS,
    ALL_GROUPS,
    ALL_WINDOWS,
    ABLATION_CONFIGS,
)

__version__ = "1.0.0"
__all__ = [
    "compute_metrics",
    "build_features",
    "build_features_full",
    "get_ablation_features",
    "METRIC_GROUPS",
    "ALL_GROUPS",
    "ALL_WINDOWS",
    "ABLATION_CONFIGS",
]
