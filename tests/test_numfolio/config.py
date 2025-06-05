"""
---------
config.py
---------

The configuration file for tests
"""

import unittest

from sys import platform
from string import ascii_letters
from typing import Union, Optional
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import xmlrunner

from scipy.stats import wishart
from numpy.random import default_rng
from statsmodels.stats.correlation_tools import corr_nearest

__all__ = ["xml_test_folder", "simulate_univariate_process"]

config_path = Path(__file__).absolute().parent


def simulate_univariate_process(
    size: int,
    baseline: float = 100.0,
    std: float = 1.0,
    min_value: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Create an array of simulated prices from Student-T returns with high ddf
    to be gaussian-like.

    Args:
        size: length of prices
        baseline: starting value of the price
        std: standard deviation scale
        min_value: minimum allowed price value
        rng: numpy Generator

    Returns:
        prices array

    """

    if rng is None:
        rng = default_rng()

    returns = rng.standard_t(8, size=size)
    return np.maximum(baseline + std * returns.cumsum(), min_value)


def _is_positive_definite(m: np.ndarray) -> bool:
    """
    Test if a given matrix is positive-definite

    Args:
        m: input matrix

    Returns:
        a boolean indicator

    """
    if np.allclose(m, m.T):
        try:
            np.linalg.cholesky(m)
            return True
        except Exception:
            return False
    return False


def _symmetrize(corr: np.ndarray) -> np.ndarray:
    """Create a symmetric matrix from a given input"""
    corr = corr @ corr.T
    corr /= np.abs(corr).max()
    corr = corr_nearest(corr)
    np.fill_diagonal(corr, 1.0)
    assert _is_positive_definite(corr), "`corr` is not symmetric positive definite"
    return corr


def generate_correlation(
    n_col: int,
    rng: np.random.Generator | None = None,
    coefficients: dict[tuple[int, int], float] | None = None,
) -> np.ndarray:
    """
    Generate a correlation matrix
    In order to avoid small off-diagonal elements in the correlation matrix
    we use as a starting point a Wishart distribution, see this topic:
    https://stats.stackexchange.com/questions/2746

    Args:
        n_col: number of columns
        rng: numpy Generator
        coefficients: a dict in the form {(i, j): val, (i', j'): val', ...}
            used to fix some values

    Returns:
        correlation matrix

    """
    corr = wishart.rvs(df=n_col + 1, scale=np.eye(n_col), random_state=rng)
    if coefficients is not None:
        corr = _symmetrize(corr)
        for (i, j), val in coefficients.items():
            corr[i, j] = corr[j, i] = val
    return _symmetrize(corr=corr)


def simulate_multivariate_process(
    size: int,
    n_col: int,
    baseline: float | np.ndarray | str = 100.0,
    std: float = 1.0,
    min_price: float = 1.0,
    corr: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Create an array of simulated prices using a multivariate Student-t distribution
    for returns.

    Args:
        size: length of prices
        n_col: number of columns
        baseline: starting value of the price (if float)
            or a common trend for all the series generated
        std: standard deviation scale
        min_price: minimum allowed price value
        corr: an optional correlation matrix
        rng: numpy Generator

    Returns:
        multivariate process

    """

    if rng is None:
        rng = default_rng()

    if corr is None:
        corr = generate_correlation(n_col=n_col, rng=rng)
    else:
        assert corr.shape == (
            n_col,
            n_col,
        ), f"corr.shape = {corr.shape} =! ({n_col}, {n_col})"
        if not _is_positive_definite(corr):
            raise ValueError("`corr` is not positive definite")

    scale = rng.chisquare(df=8, size=size)
    returns = rng.multivariate_normal([0.0] * n_col, corr, size) / np.sqrt(scale)[:, None]

    if isinstance(baseline, (float, int)):
        baseline = np.ones(size) * baseline
    elif baseline == "trend":
        baseline = simulate_univariate_process(size=size, rng=rng)
    else:
        raise TypeError("baseline is float/int or `trend` for a stochastic trend")

    return np.maximum(baseline[:, None] + std * returns.cumsum(axis=1), min_price)


def simulate_scenario(
    size: int, n_col: int, rng: np.random.Generator | None = None, freq: str = "B"
) -> pd.DataFrame:
    """
    Simulate a random dataset

    Args:
        size: length of the dataset
        n_col: number of time-series
        rng: numpy Generator
        freq: frequency type of pd.DateTimeIndex

    Returns:
        simulated scenario

    """

    if rng is None:
        rng = default_rng()

    index = pd.date_range(start="2021-01-02", freq=freq, periods=size)
    std = rng.exponential()
    min_price = rng.exponential()

    prices = simulate_multivariate_process(
        size=size, n_col=n_col, std=std, min_price=min_price, baseline="trend", rng=rng
    )

    return pd.DataFrame(prices, columns=list(ascii_letters[:n_col]), index=index)


def get_xml_test_folder() -> str:
    """
    Serve a test-reports folder based on the current system environment

    Returns:
        xml_test_folder: default test report folder

    """
    path = config_path.parent / "test-reports"
    path.mkdir(exist_ok=True)

    return str(path)


xml_test_folder = get_xml_test_folder()


class TestConfig(unittest.TestCase):
    """The test class for configuration file"""

    def test_all(self):
        """Assert that test are allowed"""

        for name in __all__:
            self.assertIn(name, globals())

    def test_simulate_univariate_process(self):
        """Test the simulate_univariate_process function"""

        n_tests = 5
        rng = default_rng()

        for k in range(n_tests):
            size = rng.integers(low=10, high=500)
            std = rng.exponential()
            min_value = rng.standard_normal()
            baseline = rng.exponential(scale=100)

            prices = simulate_univariate_process(
                size, std=std, min_value=min_value, baseline=baseline
            )
            self.assertIsInstance(prices, np.ndarray)
            self.assertEqual(prices.shape, (size,))
            self.assertTrue((prices >= min_value).all())

    def test_generate_correlation(self):
        """Test the generate_correlation function"""

        n_tests = 5
        rng = default_rng()

        for _ in range(n_tests):
            n_col = rng.integers(low=2, high=20)

            corr = generate_correlation(n_col)
            self.assertEqual(corr.shape, (n_col, n_col))
            self.assertTrue((np.abs(corr) <= 1.0).all())
            self.assertTrue(np.allclose(corr, corr.T))
            self.assertTrue(all(corr.diagonal() == 1.0))
            self.assertTrue(all(np.linalg.eigvals(corr) > 0))

            try:
                np.linalg.cholesky(corr)
            except Exception as e:
                self.assertTrue(False, msg=f"{e.__class__.__name__}: {e}")

    def test_simulate_multivariate_process(self):
        """Test the simulate_multivariate_prrocess function"""

        n_tests = 5
        rng = default_rng()

        for k in range(n_tests):
            size = rng.integers(low=10, high=500)
            n_col = rng.integers(low=2, high=20)
            min_price = rng.standard_normal()
            std = rng.exponential()
            if k % 2 == 0:
                baseline = "trend"
            else:
                baseline = rng.exponential(scale=100)

            prices = simulate_multivariate_process(
                size, n_col, std=std, min_price=min_price, baseline=baseline
            )
            self.assertIsInstance(prices, np.ndarray)
            self.assertEqual(prices.shape, (size, n_col))
            self.assertTrue((prices >= min_price).all())


def build_suite():
    """Construct the TestSuite"""

    suite = unittest.TestSuite()

    suite.addTest(TestConfig("test_all"))
    suite.addTest(TestConfig("test_simulate_univariate_process"))
    suite.addTest(TestConfig("test_generate_correlation"))
    suite.addTest(TestConfig("test_simulate_multivariate_process"))

    return suite


if __name__ == "__main__":
    """The main script"""

    runner = xmlrunner.XMLTestRunner(output=xml_test_folder)
    runner.run(build_suite())
