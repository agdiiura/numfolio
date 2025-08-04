"""
---------
reward.py
---------

A module for statistical functions.
"""

import numba
import numpy as np
import statsmodels.api as sm

from ._stats import _compute_pnl


@numba.njit
def compute_average_returns(returns: np.ndarray, r: float = 0) -> float:
    """
    Compute the average returns.

    Args:
        returns: a vector-like object of returns
        r: risk-free level (default is 0)

    Returns:
        the average return value

    Examples:

        >>> compute_average_returns(np.array([0.01, 0.02, -0.01]))
        0.006666666666666667

        >>> compute_average_returns(np.array([-0.1, 0.05, 0.05]))
        -0.03333333333333333

    """
    return np.nanmean(returns) - r


@numba.njit
def compute_final_pnl(returns: np.ndarray) -> float:
    """
    Compute the final PnL value

    Args:
        returns: a vector-like object of returns

    Returns:
        final PnL (last cumulative sum value)

    Examples:

        >>> compute_final_pnl(np.array([0.01, 0.02, -0.01]))
        0.02

        >>> compute_final_pnl(np.array([-0.1, 0.05, 0.05]))
        0.0

    """
    pnl = _compute_pnl(returns)
    return pnl[-1] - pnl[0]


@numba.njit
def compute_final_pnl_percentage(returns: np.ndarray, baseline: float = 1) -> float:
    """
    Compute final PnL as percentage (multiplied by 100).

    Args:
        returns: a vector-like object of returns
        baseline: default value for portfolio

    Returns:
        final PnL percentage

    Examples:

        >>> compute_final_pnl_percentage(np.array([0.01, 0.02, -0.01]))
        2.0

        >>> compute_final_pnl_percentage(np.array([-0.1, 0.05, 0.05]))
        0.0

    """
    return 100.0 * compute_final_pnl(returns) / baseline


def compute_stability_of_timeseries(returns: np.ndarray) -> float:
    """
    Compute the stability of a time series by regressing
    cumulative returns against time.

    Args:
        returns: a vector-like object of returns

    Returns:
        stability coefficient (R-squared of regression)

    Examples:

        >>> compute_stability_of_timeseries(np.array([0.01, 0.02, 0.03]))
        1.0

        >>> compute_stability_of_timeseries(np.array([0.0, 0.0, 0.0]))
        0.0

    """
    pnl = _compute_pnl(returns)

    lags = np.arange(pnl.size)

    model = sm.OLS(pnl, lags)
    res = model.fit()

    return res.rsquared


if __name__ == "__main__":
    pass
