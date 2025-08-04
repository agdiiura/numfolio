"""
A module for computing various risk measures and portfolio
optimization objectives.

It includes functions to compute downside risk, maximum drawdown, and an objective function
for portfolio optimization using Optuna.
"""

import pandas as pd

import numfolio as nf


def make_portfolio_returns(data: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """
    Calculate portfolio returns based on the provided weights.

    Args:
        data: DataFrame containing asset returns.
        weights: Dictionary with asset names as keys and their respective weights as values.

    Returns:
        DataFrame with portfolio returns.

    """

    return data.ffill().diff().dropna() @ [weights[c] for c in data.columns]


class Objective(object):
    """
    Objective function for portfolio optimization.

    This class evaluates a portfolio based on various risk measures and constraints.
    """

    def __init__(
        self,
        score: callable,
        train: pd.DataFrame,
        penalty: float = -1.0,
        max_weight: int = 100,
        budget: float = 1000.0,
        min_diversification: int = 2,
        max_var: float = 0.05,
        max_drawdown: float = 0.2,
        min_r2: float = 0.5,
    ):
        """
        Initialize the Objective function.

        Args:
            score: Function to compute the score of the portfolio.
            train: DataFrame containing training data for returns.
            penalty: Penalty for invalid portfolios.
            max_weight: Maximum weight for any asset in the portfolio.
            budget: Total budget for the portfolio.
            min_diversification: Minimum number of assets in the portfolio.
            max_var: Maximum allowed variance of the portfolio returns.
            max_drawdown: Maximum allowed drawdown of the portfolio returns.
            min_r2: Minimum R-squared value for the stability of the time series.

        """

        self.score = score
        self.train = train
        self.columns = train.columns
        self.penalty = penalty
        self.max_weight = max_weight
        self.min_diversification = min_diversification
        self.max_var = max_var
        self.max_drawdown = max_drawdown
        self.min_r2 = min_r2
        self.budget = budget

        self.lp = train.iloc[-1]

    def __call__(self, trial):
        """
        Call method to evaluate the portfolio based on the trial parameters.

        Args:
            trial: Optuna trial object containing the parameters for the portfolio.

        """

        weights = dict()
        for col in self.columns:
            weights[col] = trial.suggest_int(col, 0, self.max_weight)

        returns = make_portfolio_returns(data=self.train, weights=weights)
        returns = returns.values

        b = sum([lp[k] * v for k, v in weights.items()])
        if b > self.budget:
            return self.penalty

        if len([v for v in weights.values() if v > 0]) < self.min_diversification:
            return self.penalty

        if nf.compute_var(returns) > self.max_var:
            return self.penalty

        if nf.compute_max_drawdown(returns) > self.max_drawdown:
            return self.penalty

        if nf.compute_stability_of_timeseries(returns) < self.min_r2:
            return self.penalty

        return self.score(returns=returns)
