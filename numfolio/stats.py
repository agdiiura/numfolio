"""
--------
stats.py
--------

A module for statistical functions.

The usual signature for this module is the following

>>> def compute_something(returns: np.ndarray, **kwargs) -> float:
>>>     # do something
>>>     # return something
"""

import sys
import inspect

import numba
import numpy as np
import statsmodels.api as sm

from scipy.optimize import minimize_scalar

__all__ = [
    "compute_cvar",
    "compute_cvar_mid",
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
    """Compute PNL from input returns"""
    pnl = returns[np.isfinite(returns)].cumsum()
    return pnl[np.isfinite(pnl)]


@numba.njit("float64[:](float64[:])", cache=True)
def _compute_loss(returns: np.ndarray) -> np.ndarray:
    """Compute Loss from input returns"""
    return -returns[np.isfinite(returns)]


@numba.njit
def compute_sharpe_ratio(returns: np.ndarray, r: float = 0.0) -> float:
    """
    Compute the Sharpe-ratio

    • Sharpe, William F.
        "The sharpe ratio."
        Journal of portfolio management 21.1 (1994): 49-58.

    Args:
        returns: a vector-like object of returns
        r: risk-free level

    Returns:
        the Sharpe-ratio value

    """
    std = np.nanstd(returns)
    if np.isfinite(std):
        return annualized_factor * (np.nanmean(returns) - r) / std

    return np.nan


@numba.njit
def compute_sortino_ratio(returns: np.ndarray, r: float = 0.0) -> float:
    """
    Compute the Sortino-ratio

    • Sortino, Frank A., and Lee N. Price.
        "Performance measurement in a downside risk framework."
        The Journal of Investing 3.3 (1994): 59-64.

    Args:
        returns: a vector-like object of returns
        r:  risk-free level

    Returns:
        the Sortino-ratio value

    """
    downside_deviations = returns[returns < r]
    std = np.nanstd(downside_deviations)
    if np.isfinite(std):
        return annualized_factor * (np.nanmean(returns) - r) / std

    return np.nan


@numba.njit
def compute_downside_risk(returns: np.ndarray, r: float = 0.0) -> float:
    """
    Compute the Downside Risk measure

    • Nawrocki, David N.
        "A brief history of downside risk measures."
        The Journal of Investing 8.3 (1999): 9-25

    Args:
        returns: a vector-like object of returns
        r:  risk-free level

    Returns:
        the semideviance value

    """
    downside_deviations = returns[returns < r]
    std = downside_deviations.std()
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
    Compute VaR using numba through the quantile method

    • Artzner, Philippe, et al.
        "Coherent measures of risk."
        Mathematical finance 9.3 (1999): 203-228.

    Args:
        returns: a vector-like object of returns
        alpha: quantile level

    Returns:
        value-at-risk

    """

    loss = _compute_loss(returns)
    return np.nanquantile(a=loss, q=1.0 - alpha)


@numba.njit
def compute_cvar(
    returns: np.ndarray, alpha: float = 0.05, n_step: int = 100, low_alpha: float = 0.001
) -> float:
    """
    Compute CVaR by approximating the integral with the discrete
    version of it. The method uses numba.

    • Artzner, Philippe, et al.
        "Coherent measures of risk."
        Mathematical finance 9.3 (1999): 203-228.

    • Rockafellar, R. Tyrrell, and Stanislav Uryasev.
        "Optimization of conditional value-at-risk."
        Journal of risk 2 (2000): 21-42.

    • Norton, Matthew, Valentyn Khokhlov, and Stan Uryasev.
        "Calculating CVaR and bPOE for common probability
        distributions with application to portfolio optimization
        and density estimation."
        Annals of Operations Research 299.1 (2021): 1281-1315.

    Args:
        returns: a vector-like object of returns
        alpha: quantile level
        n_step: number of step in the numerical approximation
        low_alpha: low level of alpha

    Returns:
        conditional value-at-risk

    """

    alphas = np.linspace(low_alpha, alpha, n_step)
    return np.nanmean(compute_var(returns=returns, alpha=alphas))


@numba.njit
def compute_cvar_mid(returns: np.ndarray, alpha: float = 0.05) -> float:
    """
    Compute CVaR by approximating the integral with
    the rectangle rule. The method uses numba.

    Args:
        returns: a vector-like object of returns
        alpha: quantile level

    Returns:
        conditional value-at-risk

    """

    return compute_var(returns=returns, alpha=0.5 * alpha)


@numba.njit
def _compute_evar(z: float, returns: np.ndarray, alpha: float = 0.05) -> float:
    """Compute the EVaR as a function of the z parameter"""
    if z <= 0 or np.isinf(z):
        return np.inf
    m = np.nanmean(np.exp(-returns / z))
    return z * (np.log(m) - np.log(alpha))


def compute_evar(returns: np.ndarray, alpha: float = 0.5) -> float:
    """
    Compute the EVaR

    • Ahmadi-Javid, Amir.
        "Entropic value-at-risk: A new coherent risk measure."
        Journal of Optimization Theory and Applications 155.3 (2012): 1105-1123.

    Args:
        returns: a vector-like object of returns
        alpha: quantile level

    Returns:
        the Entropic Value at Risk value

    """

    res = minimize_scalar(_compute_evar, args=(returns, alpha), method="Brent")

    if res.success:
        return res.fun
    else:
        return np.nan


@numba.njit
def compute_tail_ratio(returns: np.ndarray) -> float:
    """
    Compute the tail ratio, defined as the ration between the 95° percentile
    over the 5° percentile of the distribution

    • Konno, Hiroshi, Katsuhiro Tanaka, and Rei Yamamoto.
        "Construction of a portfolio with shorter downside tail
        and longer upside tail."
        Computational Optimization and Applications 48.2 (2011): 199-212.

    Args:
        returns: a vector-like object of returns

    Returns:
        the tail-ratio value

    """

    den = np.abs(np.nanquantile(returns, 0.05))
    if den != 0:
        return np.abs(np.nanquantile(returns, 0.95)) / den
    else:
        return np.nan


@numba.njit
def compute_omega_ratio(returns: np.ndarray, r: float = 0.0) -> float:
    """
    Compute the omega-ratio

    • Kapsos, M., Zymler, S., Christofides, N., & Rustem, B
        "Optimizing the Omega ratio using linear programming."
        Journal of Computational Finance 17.4 (2014): 49-57.

    Args:
        returns: a vector-like object of returns
        r:  risk-free level

    Returns:
        the Omega-ratio value

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
    Compute the calmar-ratio, defined as the ratio of the expected return
    over the max-drawdown

    • Magdon-Ismail, Malik, and Amir F. Atiya.
        "Maximum drawdown."
        Risk Magazine 17.10 (2004): 99-102.

    • Petroni, Filippo, and Giulia Rotundo.
        "Effectiveness of measures of performance during speculative bubbles."
        Physica A: Statistical Mechanics and its
        Applications 387.15 (2008): 3942-3948.

    Args:
        returns: a vector-like object of returns
        r:  risk-free level

    Returns:
        the Calmar-ratio value

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
    Compute the RAROC, defined as the ratio of the expected return
    over the VaR

    • Stoughton, Neal M., and Josef Zechner.
        "Optimal capital allocation using RAROC and EVA."
        Journal of Financial Intermediation 16.3 (2007): 312-342.

    • Prokopczuk, Marcel, et al.
        "Quantifying risk in the electricity business: A RAROC-based approach."
        Energy Economics 29.5 (2007): 1033-1049.

    Args:
        returns: a vector-like object of returns
        r:  risk-free level
        alpha: quantile level

    Returns:
        the RAROC value

    """

    var = compute_var(returns, alpha=alpha)
    if var != 0:
        return annualized_factor * (np.nanmean(returns) - r) / var

    return np.nan


@numba.njit
def compute_final_pnl(returns: np.ndarray) -> float:
    """
    Compute the final P&L

    Args:
        returns: a vector-like object of returns

    Returns:
        final value of the pnl

    """
    pnl = _compute_pnl(returns)
    return pnl[-1] - pnl[0]


@numba.njit
def compute_final_pnl_percentage(returns: np.ndarray, baseline: float = 1) -> float:
    """
    Compute the final P&L as a percentage

    Args:
        returns: a vector-like object of returns
        baseline: default value for portfolio

    Returns:
        final value of the pnl as a percentage

    """
    return 100.0 * compute_final_pnl(returns) / baseline


def compute_stability_of_timeseries(returns: np.ndarray) -> float:
    """
    Compute the stability of the timeseries.
    Computes an ordinary least squares linear fit, and returns R-squared.

    Args:
        returns: a vector-like object of returns

    Returns:
        the R^2 of the regression

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
    for name, f in inspect.getmembers(sys.modules[__name__]):
        if callable(f) and name.startswith("compute_"):
            results[name] = f(values)
    return results


if __name__ == "__main__":
    pass
