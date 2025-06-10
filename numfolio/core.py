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
        metric: input metric, either the name of a metric function
            (without 'compute_' prefix) defined in the 'stats' module
            or a callable that accepts returns as the first argument.
        n_bootstraps: number of bootstrap samples
        n_jobs: number of parallel jobs in the computation
        min_length: minimum size of bootstrap sample
        rng: numpy random Generator
        kwargs: additional arguments passed to the metric function

    Returns:
        the array contained the bootstrapped results

    Example:

        >>> import numpy as np
        >>> returns = np.random.default_rng().normal(loc=0, scale=0.01, size=100)
        >>> results = bootstrap_metric(returns, metric="sharpe_ratio", n_bootstraps=100)
        >>> print(results.shape)
        (100,)

        Using a custom metric function:
        >>> def mean_return(x):
        ...     return np.mean(x)
        >>> results = bootstrap_metric(returns, metric=mean_return, n_bootstraps=100)
        >>> np.mean(results)
        0.0005  # (example output)

    """

    if rng is None:
        rng = np.random.default_rng()

    if n_bootstraps <= 0:
        raise ValueError("n_bootstraps must be a positive integer")

    if isinstance(metric, str):
        module = inspect.getmembers(stats)
        matches = [v for name, v in module if callable(v) and name == f"compute_{metric}"]
        if not matches:
            raise ValueError(f"Metric '{metric}' not found in stats module.")
        f = matches[0]
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
    Generate a performance scorecard of portfolio metrics aggregated by period.

    Args:
        portfolio: DataFrame containing at least 'returns' or 'pnl' columns.
            If one is missing, it will be computed internally
        freq: Resampling frequency: 'Y' (year), 'Q' (quarter), or 'M' (month)

    Returns:
        DataFrame with metrics such as Sharpe Ratio, Sortino Ratio, Max Drawdown,
        VaR, CVaR, and Final P&L for each period plus a total summary.

    Example:

        >>> import pandas as pd
        >>> import numpy as np
        >>> dates = pd.date_range("2020-01-01", periods=100, freq="D")
        >>> pnl = np.cumsum(np.random.normal(0, 1, size=100))
        >>> df = pd.DataFrame({"pnl": pnl}, index=dates)
        >>> scorecard = get_scorecard(df, freq="M")
        >>> print(scorecard)
        Period         2020-M1   2020-M2  Total
        Sharpe-Ratio    0.10      0.12     0.11
        Sortino-Ratio   0.15      0.18     0.16
        MaxDD          -0.25     -0.30    -0.28
        VaR            -0.05     -0.04    -0.045
        CVaR           -0.07     -0.06    -0.065
        FinalP&L       12.34     14.56    26.90

    """

    if "pnl" not in portfolio.columns and "returns" not in portfolio.columns:
        raise ValueError("Portfolio must contain at least 'pnl' or 'returns' column.")

    portfolio = portfolio.copy()

    if "returns" not in portfolio:
        portfolio["returns"] = portfolio["pnl"].diff()
    elif "pnl" not in portfolio.columns:
        portfolio["pnl"] = portfolio["returns"].cumsum()

    map_freq = [
        ("Y", {"year": "Y"}),
        ("Q", {"year": "Y", "quarter": "Q"}),
        ("M", {"year": "Y", "month": ""}),
    ]

    idx = [itm[0] for itm in map_freq].index(freq)
    keys = map_freq[idx][1].keys()
    template = map_freq[idx][1].values()

    index = portfolio.index
    vals = np.array([getattr(index, k) for k in keys]).T.astype(str)

    portfolio.loc[:, "freq"] = [
        "-".join(f"{t}{x.zfill(2) if freq == 'M' else x}" for t, x in zip(template, itm))
        for itm in vals
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
    Compute the absolute return of a series

    Args:
        x: Input pandas Series representing prices or values over time

    Returns:
        The absolute return computed as last value minus first value

    Example:

        >>> import pandas as pd
        >>> s = pd.Series([100, 105, 110])
        >>> compute_returns(s)
        10.0

    """
    return x.iloc[-1] - x.iloc[0]


def compute_pct_returns(x: pd.Series) -> float:
    """
    Compute the percentage return of a series

    Args:
        x: Input pandas Series representing prices or values over time

    Returns:
        The percentage return computed as (last / first) - 1,
        or NaN if the first value is zero

    Example:

        >>> import pandas as pd
        >>> s = pd.Series([100, 110])
        >>> compute_pct_returns(s)
        0.10

        >>> s_zero = pd.Series([0, 110])
        >>> compute_pct_returns(s_zero)
        nan

    """
    p0 = x.iloc[0]
    if p0 != 0:
        return x.iloc[-1] / p0 - 1
    else:
        return np.nan


def _fit_covariance_pipeline(r: pd.DataFrame, method: str):
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
    Estimate a correlation matrix using a rolling window and bootstrap procedure.


    Args:
        returns: DataFrame of returns with assets as columns
        method: estimation method, can be 'empyrical', 'glassocv' or 'ledoit_wolf'
        rolling_window: Window size for rolling returns computation.
        n_bootstraps: Number of bootstrap samples.
        n_jobs: Number of parallel jobs.
        min_length: Minimum block length for bootstrap.
        rng: Random generator for reproducibility.

    Returns:
        DataFrame containing the estimated correlation matrix

    Example:

        >>> import pandas as pd
        >>> import numpy as np
        >>> returns = pd.DataFrame(np.random.normal(0, 0.01, (100, 3)),
        ...                        columns=["A", "B", "C"])
        >>> corr = estimate_correlation(returns, method="ledoit_wolf", n_bootstraps=10)
        >>> corr.shape
        (3, 3)
        >>> corr.columns.tolist()
        ['A', 'B', 'C']

    """

    returns = returns.cumsum().rolling(rolling_window).apply(compute_returns).dropna()

    optimal_length = max(
        int(optimal_block_length(returns**2)["circular"].median()), min_length
    )

    if n_bootstraps <= 0:
        raise ValueError("n_bootstraps must be a positive integer")

    if rng is None:
        rng = np.random.default_rng()

    bs = CircularBlockBootstrap(optimal_length, returns, seed=rng)

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

    Example:

        >>> import pandas as pd
        >>> import numpy as np
        >>> corr = pd.DataFrame([[1, 0.5], [0.5, 1]], columns=["A", "B"], index=["A", "B"])
        >>> dist = compute_robust_distance(corr)
        >>> dist.loc["A", "B"]
        0.7071067811865476

    """

    return np.sqrt(1.0 - np.clip(corr, a_min=-1, a_max=1))


if __name__ == "__main__":
    pass
