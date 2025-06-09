"""
-------
core.py
-------

A module for core functionalities.
"""

import inspect
import warnings

from typing import Callable

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from numpy.random import Generator
from arch.bootstrap import CircularBlockBootstrap, optimal_block_length
from sklearn.pipeline import Pipeline
from sklearn.covariance import (LedoitWolf, GraphicalLassoCV,
                                EmpiricalCovariance)
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.correlation_tools import corr_nearest

from . import stats

__all__ = [
    "estimate_correlation",
    "compute_robust_distance",
    "bootstrap_metric",
    "get_scorecard",
]


def bootstrap_metric(
    returns: np.ndarray,
    metric: str | Callable = "sharpe_ratio",
    n_bootstraps: int = 1000,
    n_jobs: int = 2,
    min_length: int = 5,
    rng: None | Generator = None,
    **kwargs: dict,
) -> np.ndarray:
    """
    Compute input metric using bootstrapping procedure

    Args:
        returns: a vector-like object of returns
        metric: input metric, as defined in stats module
        n_bootstraps: number of bootstrap samples
        n_jobs: number of parallel jobs in the computation
        min_length: minimum size of bootstrap sample
        rng: numpy random Generator
        kwargs: optional arguments

    Returns:
        the array contained the bootstrapped results

    """
    if rng is None:
        rng = np.random.default_rng()

    if isinstance(metric, str):
        module = inspect.getmembers(stats)
        f = [v for name, v in module if callable(v) and name == f"compute_{metric}"][0]
    elif callable(metric):
        f = metric
    else:
        raise TypeError("metric not defined")

    optimal_length = optimal_block_length(returns**2)["circular"]
    optimal_length = max(int(optimal_length.iloc[0]), min_length)

    cb = CircularBlockBootstrap(optimal_length, returns, seed=rng)
    result = Parallel(n_jobs=n_jobs)(
        delayed(lambda x: f(x, **kwargs))(*pos_data)
        for pos_data, kw_data in cb.bootstrap(n_bootstraps)
    )

    return np.array(result)


def get_scorecard(portfolio: pd.DataFrame, freq: str = "Y") -> pd.DataFrame:
    """
    Get score-cards report

    Args:
        portfolio: (pd.DataFrame) input data with at least 'returns' and 'ptf'
            columns
        freq: (str) resampling frequency

    Returns:
        metrics values of the portfolio performance

    """

    if isinstance(portfolio, pd.Series):
        portfolio = portfolio.to_frame(name="pnl")

    if "returns" not in portfolio.columns:
        portfolio["returns"] = portfolio["pnl"].diff()
    elif "pnl" not in portfolio.columns:
        portfolio["pnl"] = portfolio["returns"].cumsum()

    # TODO issue in pandas https://github.com/pandas-dev/pandas/issues/32803
    """scorecard = results['Total'].resample(freq).agg({
        'returns': [compute_final_pnl, compute_sharpe_ratio, compute_sortino_ratio, compute_var],
        'pnl': [compute_max_drawdown]
    })"""
    map_freq = [("Y", "year"), ("Q", "quarter"), ("M", "month")]
    idx = [itm[0] for itm in map_freq].index(freq)

    cal = portfolio.index
    vals = np.array([getattr(cal, itm[1]) for itm in map_freq[: idx + 1]]).T

    portfolio.loc[:, "freq"] = [
        "-".join(f"{map_freq[k][0]}{x}" for k, x in enumerate(itm)) for itm in vals
    ]

    scorecard = portfolio.groupby("freq").agg(
        sharpe_ratio=("returns", lambda x: stats.compute_sharpe_ratio(x.dropna().values)),
        sortino_ratio=(
            "returns",
            lambda x: stats.compute_sortino_ratio(x.dropna().values),
        ),
        max_drawdown=("returns", lambda x: stats.compute_max_drawdown(x.dropna().values)),
        var=("returns", lambda x: stats.compute_var(x.dropna().values)),
        cvar=("returns", lambda x: stats.compute_cvar(x.dropna().values)),
        final_pnl=("returns", lambda x: stats.compute_final_pnl(x.dropna().values)),
    )

    # scorecard.index = [str(pd.Timestamp(itm).date()) for itm in scorecard.index]

    scorecard.index.name = "Period"
    # reset columns name
    # scorecard = scorecard.droplevel(0, axis=1)

    scorecard.rename(
        columns={
            "sharpe_ratio": "Sharpe-Ratio",
            "sortino_ratio": "Sortino-Ratio",
            "max_drawdown": "MaxDD",
            "var": "VaR",
            "cvar": "CVaR",
            "final_pnl": "FinalP&L",
        },
        inplace=True,
    )

    returns = portfolio["returns"].dropna().values

    scorecard.loc["Total", "Sharpe-Ratio"] = stats.compute_sharpe_ratio(returns)
    scorecard.loc["Total", "Sortino-Ratio"] = stats.compute_sortino_ratio(returns)
    scorecard.loc["Total", "MaxDD"] = stats.compute_max_drawdown(returns)
    scorecard.loc["Total", "VaR"] = stats.compute_var(returns)
    scorecard.loc["Total", "CVaR"] = stats.compute_cvar(returns)
    scorecard.loc["Total", "FinalP&L"] = stats.compute_final_pnl(returns)

    return scorecard.T


def compute_returns(x: pd.Series) -> float:
    """
    Compute returns from input series

    Args:
        x: input series

    Returns:
        returns of the input series

    """
    return x.iloc[-1] - x.iloc[0]


def compute_pct_returns(x: pd.Series) -> float:
    """
    Compute returns from input series

    Args:
        x: input series

    Returns:
        percentage returns of the input series

    """
    p0 = x.iloc[0]
    if p0 != 0:
        return x.iloc[-1] / p0 - 1
    else:
        return np.nan


def _fit_covariance_pipeline(r, method: str):
    """Fit the covariance pipeline"""
    map_estimators = {
        "empyrical": EmpiricalCovariance,
        "glassocv": GraphicalLassoCV,
        "ledoit_wolf": LedoitWolf,
    }

    f = map_estimators[method]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = Pipeline([("scaler", StandardScaler()), ("estimator", f())])
        pipe.fit(r)
        return pipe[-1].covariance_


def estimate_correlation(
    returns: pd.DataFrame,
    method: str = "empyrical",
    rolling_window: int = 5,
    n_bootstraps: int = 100,
    n_jobs: int = 2,
    min_length: int = 5,
    rng: None | Generator = None,
) -> pd.DataFrame:
    """
    Estimate the correlation matrix using a bootstrap procedure

    Args:
        returns: a series of returns object
        method: estimation method, can be 'empyrical', 'glassocv' or 'ledoit_wolf'
        rolling_window: returns window size
        n_bootstraps: number of bootstrap samples
        n_jobs: number of parallel jobs
        rng: numpy random Generator
        min_length: minimum size of bootstrap sample

    Returns:
        estimated correlation matrix

    """

    returns = returns.cumsum().rolling(rolling_window).apply(compute_returns).dropna()

    ol = max(int(optimal_block_length(returns**2)["circular"].median()), min_length)

    if rng is None:
        rng = np.random.default_rng()

    bs = CircularBlockBootstrap(ol, returns, seed=rng)

    covariances = Parallel(n_jobs=n_jobs)(
        delayed(lambda x: _fit_covariance_pipeline(x, method=method))(*pos_data)
        for pos_data, kw_data in bs.bootstrap(n_bootstraps)
    )

    corr = np.array(covariances)
    corr = corr.mean(axis=0)
    corr = corr_nearest(corr)
    return pd.DataFrame(corr, index=returns.columns, columns=returns.columns)


def compute_robust_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a robust version of distance metric from correlation

    Args:
        corr: input correlation matrix

    Returns:
        robust distance

    """

    return np.sqrt(1.0 - np.clip(corr, a_min=-1, a_max=1))


if __name__ == "__main__":
    pass
