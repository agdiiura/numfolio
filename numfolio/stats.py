"""
--------
stats.py
--------

A module for statistical functions.
"""

import sys
import inspect

import numba
import numpy as np
import statsmodels.api as sm

from scipy.optimize import minimize_scalar

__all__ = [
    "compute_cvar",
    "compute_var",
    "compute_evar",
    "compute_raroc",
    "compute_final_pnl",
    "compute_sharpe_ratio",
    "compute_sortino_ratio",
    "compute_max_drawdown",
    "compute_tail_ratio",
    "compute_omega_ratio",
    "compute_calmar_ratio",
    "compute_downside_risk",
    "compute_stability_of_timeseries",
    "compute_final_pnl_percentage",
]

annualized_factor = np.sqrt(252.0)


@numba.njit("float64[:](float64[:])", cache=True)
def _compute_pnl(returns: np.ndarray) -> np.ndarray:
    """
    Compute cumulative PNL from input returns.

    Args:
        returns: array-like of returns

    Returns:
        cumulative PNL array

    Examples:
        >>> _compute_pnl(np.array([0.01, -0.02, 0.03]))
        array([0.01, -0.01, 0.02])

        >>> _compute_pnl(np.array([0.1, 0.1, 0.1]))
        array([0.1, 0.2, 0.3])

    """
    pnl = returns[np.isfinite(returns)].cumsum()
    return pnl[np.isfinite(pnl)]


@numba.njit("float64[:](float64[:])", cache=True)
def _compute_loss(returns: np.ndarray) -> np.ndarray:
    """
    Compute losses (negatives) from input returns.

    Args:
        returns: array-like of returns

    Returns:
        losses array (negatives of returns)

    Examples:
        >>> _compute_loss(np.array([0.01, -0.02, 0.03]))
        array([-0.01,  0.02, -0.03])

        >>> _compute_loss(np.array([-0.05, 0.1, 0.0]))
        array([0.05, -0.1 , -0. ])

    """
    return -returns[np.isfinite(returns)]


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
        return annualized_factor * (np.nanmean(returns) - r) / std

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
        return annualized_factor * (np.nanmean(returns) - r) / std

    return np.nan


@numba.njit
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
        return annualized_factor * (np.nanmean(returns) - r) / mdd

    return np.nan


@numba.njit
def compute_raroc(
    returns: np.ndarray, r: float = 0.0, alpha: float | np.ndarray = 0.05
) -> float:
    """
    Compute Risk-Adjusted Return on Capital (RAROC), defined as the ratio of the expected return
    over the VaR

    Args:
        returns: a vector-like object of returns
        r:  risk-free level
        alpha: quantile level

    Returns:
        the RAROC value

    Examples:

        >>> compute_raroc(np.array([0.01, 0.02, -0.01]), alpha=0.05)
        0.668

        >>> compute_raroc(np.array([-0.1, -0.05, 0.0]), alpha=0.1)
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
        return annualized_factor * (np.nanmean(returns) - r) / var

    return np.nan


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


if __name__ == "__main__":
    pass
