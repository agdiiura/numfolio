"""
-------
risk.py
-------

A module for statistical functions.
"""

import numba
import numpy as np

from scipy.optimize import minimize_scalar

from ._stats import _compute_pnl, _compute_loss, annualized_factor


@numba.vectorize(
    [
        "int32(int32,int32)",
        "int64(int64,int64)",
        "float32(float32,float32)",
        "float64(float64,float64)",
    ]
)
@numba.njit
def _numba_max(x, y):
    """
    Vectorized numba version of np.maximum.accumulate
    See: https://stackoverflow.com/questions/56551989
    """
    return x if x > y else y


def compute_downside_risk(returns: np.ndarray, r: float = 0.0) -> float:
    """
    Compute the annualized Downside Risk measure

    Args:
        returns: a vector-like object of returns
        r:  risk-free level

    Returns:
        the semideviance value

    Examples:

        >>> compute_downside_risk(np.array([0.01, -0.02, 0.03]), r=0.0)
        2.82842712474619

        >>> compute_downside_risk(np.array([0.1, 0.2, 0.3]), r=0.0)
        0.0

    References:

        Nawrocki, David N.
        "A brief history of downside risk measures."
        The Journal of Investing 8.3 (1999): 9-25

    """
    downside_deviations = returns[returns < r]
    std = np.nanstd(downside_deviations)
    if np.isfinite(std):
        return annualized_factor * std

    return np.nan


# @numba.njit
def compute_max_drawdown(returns: np.ndarray) -> float:
    """
    Compute the Maximum Drawdown
    https://stackoverflow.com/questions/22607324

    Args:
        returns: a vector-like object of returns

    Returns:
        the max-drawdown value

    Examples:

        >>> compute_max_drawdown(np.array([0.0, -0.1, 0.2, -0.1, 0.3]))
        0.30000000000000004

        >>> compute_max_drawdown(np.array([0.1, 0.2, 0.3]))
        0.0

    """
    pnl = _compute_pnl(returns)

    # end of the period
    i = np.argmax(_numba_max.accumulate(pnl) - pnl)
    if pnl[:i].size > 0:
        # j = np.argmax(pnl[:i]) start of period
        return np.max(pnl[:i]) - pnl[i]
    else:
        return np.nan


def compute_average_drawdown(returns: np.ndarray) -> float:
    """
    Compute the Average Drawdown

    Args:
        returns: a vector-like object of returns

    Returns:
        the average drawdown value

    Examples:

        >>> compute_average_drawdown(np.array([0.0, -0.1, 0.2, -0.1, 0.3]))
        0.1

        >>> compute_average_drawdown(np.array([0.1, 0.2, 0.3]))
        0.0

    """
    pnl = _compute_pnl(returns)
    return np.nanmean(_numba_max.accumulate(pnl) - pnl)


@numba.njit
def compute_var(returns: np.ndarray, alpha: float | np.ndarray = 0.05) -> float:
    """
    Compute Value-at-Risk (VaR) using the quantile method

    Args:
        returns: a vector-like object of returns
        alpha: quantile level

    Returns:
        value-at-risk at quantile alpha

    Examples:

        >>> compute_var(np.array([0.01, -0.02, 0.03]))
        0.02

        >>> compute_var(np.array([-0.1, -0.05, 0.0]), alpha=0.1)
        0.05

    References:

        Artzner, Philippe, et al.
        "Coherent measures of risk."
        Mathematical finance 9.3 (1999): 203-228.

    """

    loss = _compute_loss(returns)
    return np.nanquantile(a=loss, q=1.0 - alpha)


@numba.njit
def compute_cvar(
    returns: np.ndarray, alpha: float = 0.05, n_step: int = 100, low_alpha: float = 0.001
) -> float:
    """
    Compute Conditional Value-at-Risk (CVaR) by numerical approximation.

    Args:
        returns: a vector-like object of returns
        alpha: quantile level
        n_step: number of step in the numerical approximation
        low_alpha: low level of alpha used in integration

    Returns:
        conditional value-at-risk

    Examples:

        >>> compute_cvar(np.array([0.01, -0.02, 0.03]))
        0.019970000000000004

        >>> compute_cvar(np.array([-0.1, -0.05, 0.0]), alpha=0.1)
        0.0499

    References:

        Artzner, Philippe, et al.
        "Coherent measures of risk."
        Mathematical finance 9.3 (1999): 203-228.

        Rockafellar, R. Tyrrell, and Stanislav Uryasev.
        "Optimization of conditional value-at-risk."
        Journal of risk 2 (2000): 21-42.

        Norton, Matthew, Valentyn Khokhlov, and Stan Uryasev.
        "Calculating CVaR and bPOE for common probability
        distributions with application to portfolio optimization
        and density estimation."
        Annals of Operations Research 299.1 (2021): 1281-1315.

    """

    alphas = np.linspace(low_alpha, alpha, n_step)
    return np.nanmean(compute_var(returns=returns, alpha=alphas))


@numba.njit
def _compute_evar(z: float, returns: np.ndarray, alpha: float = 0.05) -> float:
    """Compute the EVaR as a function of the z parameter"""
    if z <= 0 or np.isinf(z):
        return np.inf
    m = np.nanmean(np.exp(-returns * z))
    return (np.log(m) - np.log(alpha)) / z


def compute_evar(returns: np.ndarray, alpha: float = 0.5) -> float:
    """
    Compute Entropic Value at Risk (EVaR)

    Args:
        returns: a vector-like object of returns
        alpha: quantile level

    Returns:
        the Entropic Value at Risk value

    Examples:

        >>> compute_evar(np.array([0.01, -0.02, 0.03]))
        0.013531...

        >>> compute_evar(np.array([-0.1, -0.05, 0.0]), alpha=0.1)
        0.067...

    References:

        Ahmadi-Javid, Amir.
        "Entropic value-at-risk: A new coherent risk measure."
        Journal of Optimization Theory and Applications 155.3 (2012): 1105-1123.

    """

    res = minimize_scalar(_compute_evar, args=(returns, alpha), method="Brent")

    if res.success:
        return res.fun
    else:
        return np.nan


if __name__ == "__main__":
    pass
