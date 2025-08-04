"""
--------
stats.py
--------

A module for statistical functions.
"""

import sys
import inspect

import numpy as np

from .risk import (compute_var, compute_cvar, compute_evar,
                   compute_max_drawdown, compute_downside_risk,
                   compute_average_drawdown)
from .reward import (compute_final_pnl, compute_final_pnl_percentage,
                     compute_stability_of_timeseries)
from .risk_adjusted_reward import (compute_tail_ratio, compute_omega_ratio,
                                   compute_calmar_ratio, compute_sharpe_ratio,
                                   compute_sortino_ratio,
                                   compute_sterling_ratio,
                                   compute_var_sharpe_ratio,
                                   compute_cvar_sharpe_ratio)

__all__ = [
    "compute_cvar",
    "compute_var",
    "compute_evar",
    "compute_var_sharpe_ratio",
    "compute_cvar_sharpe_ratio",
    "compute_final_pnl",
    "compute_sharpe_ratio",
    "compute_sortino_ratio",
    "compute_max_drawdown",
    "compute_average_drawdown",
    "compute_tail_ratio",
    "compute_omega_ratio",
    "compute_calmar_ratio",
    "compute_sterling_ratio",
    "compute_downside_risk",
    "compute_stability_of_timeseries",
    "compute_final_pnl_percentage",
]


def compile_numba_functions(size: int = 10) -> dict:
    """Compile the numba functions"""

    results = dict()
    rng = np.random.default_rng()
    values = rng.standard_normal(size)
    values = values[values != 0]  # remove zeros to avoid division by zero
    for name, f in inspect.getmembers(sys.modules[__name__]):
        if callable(f) and name.startswith("compute_"):
            results[name] = f(values)
    return results
