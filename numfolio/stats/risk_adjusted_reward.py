"""
-----------------------
risk_adjusted_reward.py
-----------------------

A module for statistical functions.
"""

import numba
import numpy as np

from .risk import (compute_var, compute_cvar, compute_max_drawdown,
                   compute_average_drawdown)
from ._stats import annualized_factor
from .reward import compute_average_returns


@numba.njit
def compute_sharpe_ratio(returns: np.ndarray, r: float = 0.0) -> float:
    """
    Compute the annualized Sharpe ratio of the returns.

    Args:
        returns: a vector-like object of returns
        r: risk-free level

    Returns:
        annualized Sharpe-ratio value

    Examples:
        >>> compute_sharpe_ratio(np.array([0.01, 0.02, 0.03]), r=0.0)
        12.598349018279691

        >>> compute_sharpe_ratio(np.array([0.0, 0.0, 0.0]), r=0.0)
        nan

    References:

        Sharpe, William F.
        "The sharpe ratio."
        Journal of portfolio management 21.1 (1994): 49-58.

    """
    std = np.nanstd(returns)
    if np.isfinite(std):
        return annualized_factor * compute_average_returns(returns=returns, r=r) / std

    return np.nan


@numba.njit
def compute_sortino_ratio(returns: np.ndarray, r: float = 0.0) -> float:
    """
    Compute the annualized Sortino-ratio, penalizing downside volatility only

    Args:
        returns: a vector-like object of returns
        r:  risk-free level

    Returns:
        the Sortino-ratio value

    Examples:

        >>> compute_sortino_ratio(np.array([0.01, 0.02, -0.01]), r=0.0)
        12.598349018279691

        >>> compute_sortino_ratio(np.array([-0.01, -0.02, -0.03]), r=0.0)
        nan

    References:

        Sortino, Frank A., and Lee N. Price.
        "Performance measurement in a downside risk framework."
        The Journal of Investing 3.3 (1994): 59-64.

    """
    downside_deviations = returns[returns < r]
    std = np.nanstd(downside_deviations)
    if np.isfinite(std):
        return annualized_factor * compute_average_returns(returns=returns, r=r) / std

    return np.nan


# @numba.njit('float64(float64[:], float64)', cache=True) (error with MDD)
def compute_calmar_ratio(returns: np.ndarray, r: float = 0.0) -> float:
    """
    Compute the Calmar ratio: annualized return
    divided by the maximum drawdown.

    Args:
        returns: a vector-like object of returns
        r:  risk-free level

    Returns:
        the Calmar-ratio value

    Examples:

        >>> compute_calmar_ratio(np.array([0.01, 0.02, 0.03]))
        5.7735

        >>> compute_calmar_ratio(np.array([-0.1, 0.05, 0.05]))
        nan

    References:

        Magdon-Ismail, Malik, and Amir F. Atiya.
        "Maximum drawdown."
        Risk Magazine 17.10 (2004): 99-102.

        Petroni, Filippo, and Giulia Rotundo.
        "Effectiveness of measures of performance during speculative bubbles."
        Physica A: Statistical Mechanics and its
        Applications 387.15 (2008): 3942-3948.

    """

    mdd = compute_max_drawdown(returns)
    if mdd != 0:
        return annualized_factor * compute_average_returns(returns=returns, r=r) / mdd

    return np.nan


# @numba.njit('float64(float64[:], float64)', cache=True) (error with MDD)
def compute_sterling_ratio(returns: np.ndarray, r: float = 0.0) -> float:
    """
    Compute the Sterling ratio: annualized return
    divided by the average drawdown.

    Args:
        returns: a vector-like object of returns
        r:  risk-free level

    Returns:
        the Sterling-ratio value

    Examples:

        >>> compute_sterling_ratio(np.array([0.01, 0.02, -0.03]))
        0.0

        >>> compute_sterling_ratio(np.array([0.01, 0.02, -0.03]))
        nan

    References:

        Bacon, Carl R.
        Practical portfolio performance measurement and attribution.
        John Wiley & Sons, 2023.

        van Heerden, Chris.
        "Establishing the risk denominator in a Sharpe ratio framework
        for share selection from a momentum investment strategy approach."
        South African Journal of Economic and Management Sciences 23.1 (2020): 1-19.

    """

    add = compute_average_drawdown(returns)
    if add != 0:
        return annualized_factor * compute_average_returns(returns=returns, r=r) / add

    return np.nan


@numba.njit
def compute_var_sharpe_ratio(
    returns: np.ndarray, r: float = 0.0, alpha: float | np.ndarray = 0.05
) -> float:
    """
    Compute Risk-Adjusted Return on Capital (RAROC), defined as the ratio of the expected return
    over the CVaR

    Args:
        returns: a vector-like object of returns
        r:  risk-free level
        alpha: quantile level

    Returns:
        the VaR Sharpe Ratio value

    Examples:

        >>> compute_var_sharpe_ratio(np.array([0.01, 0.02, -0.01]), alpha=0.05)
        0.668

        >>> compute_var_sharpe_ratio(np.array([-0.1, -0.05, 0.0]), alpha=0.1)
        0.0

    References:

        Stoughton, Neal M., and Josef Zechner.
        "Optimal capital allocation using RAROC and EVA."
        Journal of Financial Intermediation 16.3 (2007): 312-342.

        Prokopczuk, Marcel, et al.
        "Quantifying risk in the electricity business: A RAROC-based approach."
        Energy Economics 29.5 (2007): 1033-1049.

    """

    var = compute_var(returns, alpha=alpha)
    if var != 0:
        return annualized_factor * compute_average_returns(returns=returns, r=r) / var

    return np.nan


@numba.njit
def compute_cvar_sharpe_ratio(
    returns: np.ndarray, r: float = 0.0, alpha: float | np.ndarray = 0.05
) -> float:
    """
    Compute Risk-Adjusted Return on Capital (RAROC), defined as the ratio of the expected return
    over the CVaR

    Args:
        returns: a vector-like object of returns
        r:  risk-free level
        alpha: quantile level

    Returns:
        the CVaR Sharpe Ratio value

    Examples:

        >>> compute_cvar_sharpe_ratio(np.array([0.01, 0.02, -0.01]), alpha=0.05)
        0.668

        >>> compute_cvar_sharpe_ratio(np.array([-0.1, -0.05, 0.0]), alpha=0.1)
        0.0

    References:

        Dowd, Kevin.
        "A value risk approach to risk-return analysis."
        Journal of Portfolio Management 25.4 (1999): 60.

        Esfahanipour, Akbar, and Somayeh Mousavi.
        "A genetic programming model to generate risk-adjusted technical
        trading rules in stock markets."
        Expert Systems with Applications 38.7 (2011): 8438-8445.

    """

    cvar = compute_cvar(returns, alpha=alpha)
    if cvar != 0:
        return annualized_factor * compute_average_returns(returns=returns, r=r) / cvar

    return np.nan


@numba.njit
def compute_tail_ratio(returns: np.ndarray) -> float:
    """
    Compute the Tail Ratio: ratio of the absolute value of
    the 95th percentile gains to the absolute value of
    the 5th percentile losses.

    Args:
        returns: a vector-like object of returns

    Returns:
        the tail-ratio value

    Examples:

        >>> compute_tail_ratio(np.array([0.01, -0.02, 0.03]))
        1.5

        >>> compute_tail_ratio(np.array([0.1, -0.05, 0.05]))
        2.0

    References:

        Konno, Hiroshi, Katsuhiro Tanaka, and Rei Yamamoto.
        "Construction of a portfolio with shorter downside tail
        and longer upside tail."
        Computational Optimization and Applications 48.2 (2011): 199-212.

    """

    den = np.abs(np.nanquantile(returns, 0.05))
    if den != 0:
        return np.abs(np.nanquantile(returns, 0.95)) / den
    else:
        return np.nan


@numba.njit
def compute_omega_ratio(returns: np.ndarray, r: float = 0.0) -> float:
    """
    Compute the annualized Omega ratio, which is the ratio of gains over
    losses relative to a threshold r.

    Args:
        returns: a vector-like object of returns
        r:  risk-free level

    Returns:
        the Omega-ratio value

    Examples:

        >>> compute_omega_ratio(np.array([0.01, 0.02, -0.01]), r=0.0)
        2.0

        >>> compute_omega_ratio(np.array([-0.01, -0.02, -0.03]), r=0.0)
        0.0

    References:

        Kapsos, M., Zymler, S., Christofides, N., & Rustem, B
        "Optimizing the Omega ratio using linear programming."
        Journal of Computational Finance 17.4 (2014): 49-57.

    """

    returns_less_thresh = returns - r

    num = np.sum(returns_less_thresh[returns_less_thresh > 0.0])
    den = -1.0 * np.sum(returns_less_thresh[returns_less_thresh < 0.0])

    if den > 0.0:
        return annualized_factor * num / den
    else:
        return np.nan


if __name__ == "__main__":
    pass
