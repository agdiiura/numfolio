"""
------------
test_core.py
------------

A test to check the core module

To run the code
$ python test_core.py
"""

import os
import unittest

import numpy as np
import pandas as pd
import xmlrunner

from test_numfolio.config import (xml_test_folder, simulate_scenario,
                                  generate_correlation,
                                  simulate_univariate_process)

from numfolio import stats
from numfolio.core import (get_scorecard, compute_returns, bootstrap_metric,
                           compute_pct_returns, estimate_correlation,
                           compute_robust_distance)

N_JOBS = int(os.environ.get("N_JOBS", 8))
N_BOOTSTRAPS = int(os.environ.get("N_BOOTSTRAPS", 100))
SEED = int(os.environ.get("SEED", 8))

rng = np.random.default_rng(SEED)


def mock_function(returns: np.ndarray) -> float:
    """Compute mean"""

    return np.nanmean(returns)


class TestComputeReturnsFunctions(unittest.TestCase):
    """A class for test compute_returns/compute_pct_returns function"""

    @classmethod
    def setUpClass(cls):
        """Configure the test"""

        size = 252

        cls.prices = pd.Series(simulate_univariate_process(size=size, rng=rng))

    def _assert(self, method):
        map_functions = {
            "compute_returns": compute_returns,
            "compute_pct_returns": compute_pct_returns,
        }

        f = map_functions[method]

        for _ in range(5):
            val = f(self.prices.sample(100))
            self.assertIsInstance(val, float)

    def test_compute_returns(self):
        """Test the compute_returns function"""

        self._assert("compute_returns")

    def test_compute_pct_returns(self):
        """Test the compute_pct_returns function"""

        self._assert("compute_pct_returns")


class TestEstimateCorrelation(unittest.TestCase):
    """A class for test estimate_correlation function"""

    @classmethod
    def setUpClass(cls):
        """Configure the test"""

        cls.size = 252
        cls.n_col = 20

        cls.returns = simulate_scenario(size=cls.size, n_col=cls.n_col).diff().dropna()

    def _test_base(self, method: str):
        corr = estimate_correlation(
            returns=self.returns, method=method, n_bootstraps=N_BOOTSTRAPS, n_jobs=N_JOBS
        )
        self.assertIsInstance(corr, pd.DataFrame)
        self.assertEqual(corr.shape, (self.n_col, self.n_col))

        self.assertIsNotNone(np.linalg.cholesky(corr))

    def test_empyrical(self):
        """Test the empyrical estimation"""
        self._test_base("empyrical")

    def test_glassocv(self):
        """Test the glassocv estimation"""
        self._test_base("glassocv")

    def test_ledoit_wolf(self):
        """Test the ledoit_wolf estimation"""
        self._test_base("ledoit_wolf")


class TestComputeRobustDistance(unittest.TestCase):
    """A class for compute_robust_distance function"""

    @classmethod
    def setUpClass(cls):
        """Configure the test"""

        n_col = 10
        corr = generate_correlation(n_col=n_col, rng=rng)

        cls.corr = pd.DataFrame(corr, index=range(n_col), columns=range(n_col))

    def test_call(self):
        """Test the call function"""
        d = compute_robust_distance(self.corr)

        self.assertTrue((d >= 0).all().all())


class TestBootstrapMetric(unittest.TestCase):
    """A class for get_scorecard function"""

    @classmethod
    def setUpClass(cls):
        """Configure the test"""

        size = 252

        p = pd.Series(simulate_univariate_process(size=size, rng=rng))
        cls.returns = p.diff().dropna().values

    def _common_test(self, metric: str):
        m = bootstrap_metric(
            returns=self.returns, metric=metric, n_bootstraps=N_BOOTSTRAPS, n_jobs=N_JOBS
        )

        msg = f"Error with metric {metric}"
        self.assertIsInstance(m, np.ndarray, msg=msg)

        mean = np.mean(m)
        self.assertTrue(pd.notnull(mean), msg=msg)

    def test_compute_sharpe_ratio(self):
        """Test the compute_sharpe_ratio function"""
        self._common_test("sharpe_ratio")

    def test_compute_sortino_ratio(self):
        """Test the compute_sortino_ratio function"""
        self._common_test("sortino_ratio")

    def test_compute_downside_risk(self):
        """Test the compute_downside_risk function"""
        self._common_test("downside_risk")

    def test_compute_max_drawdown(self):
        """Test the compute_max_drawdown function"""
        self._common_test("max_drawdown")

    def test_compute_var(self):
        """Test the compute_var function"""
        self._common_test("var")

    def test_compute_cvar(self):
        """Test the compute_cvar function"""
        self._common_test("cvar")

    def test_compute_evar(self):
        """Test the compute_evar function"""
        self._common_test("evar")

    def test_compute_tail_ratio(self):
        """Test the compute_tail_ratio function"""
        self._common_test("tail_ratio")

    def test_compute_omega_ratio(self):
        """Test the compute_omega_ratio function"""
        self._common_test("omega_ratio")

    def test_compute_calmar_ratio(self):
        """Test the compute_calmar_ratio function"""
        self._common_test("calmar_ratio")

    def test_compute_final_pnl(self):
        """Test the compute_final_pnl function"""
        self._common_test("final_pnl")

    def test_compute_final_pnl_percentage(self):
        """Test the compute_final_pnl_percentage function"""
        self._common_test("final_pnl_percentage")

    def test_compute_raroc(self):
        """Test the compute_raroc function"""
        self._common_test("raroc")

    def test_compute_stability_of_timeseries(self):
        """Test the compute_stability_of_timeseries function"""
        self._common_test("stability_of_timeseries")

    def test_callable(self):
        """Test the execution using a callable object"""

        with self.assertRaises(TypeError):
            _ = bootstrap_metric(
                returns=self.returns, metric=1, n_bootstraps=N_BOOTSTRAPS, n_jobs=N_JOBS
            )

        m = bootstrap_metric(
            returns=self.returns,
            metric=mock_function,
            n_bootstraps=N_BOOTSTRAPS,
            n_jobs=N_JOBS,
        )

        self.assertIsInstance(m, np.ndarray)


class TestGetScorecard(unittest.TestCase):
    """A class for get_scorecard function"""

    @classmethod
    def setUpClass(cls):
        """Configure the test"""

        size = 252 * 3

        p = pd.Series(simulate_univariate_process(size=size, rng=rng))
        cls.returns = p.diff().dropna()

    def test_call(self):
        """Test the call function"""

        returns = np.array(self.returns).flatten()
        index = pd.date_range(start="2019-01-02", freq="B", periods=returns.shape[0])

        data = pd.DataFrame(
            {"returns": returns, "ptf": np.arange(returns.size)}, index=index
        )

        for freq in ["Y", "M", "Q"]:
            n_samples = data.resample(freq).last().shape[0]

            obj = get_scorecard(data, freq=freq)
            self.assertIsInstance(obj, pd.DataFrame)
            self.assertEqual(obj.shape[1], n_samples + 1)
            self.assertGreater(obj.shape[0], 0)

            map_func = {
                "Sharpe-Ratio": stats.compute_sharpe_ratio,
                "Sortino-Ratio": stats.compute_sortino_ratio,
                "MaxDD": stats.compute_max_drawdown,
                "VaR": stats.compute_var,
                "FinalP&L": stats.compute_final_pnl,
            }

            for key, f in map_func.items():
                values = data.loc[:, "returns"].to_numpy()
                expected = f(values)
                self.assertAlmostEqual(
                    obj.at[key, "Total"],
                    expected,
                    msg=f"Error with function `{key}` and freq `{freq}`",
                )


def build_suite():
    """Construct the TestSuite"""

    suite = unittest.TestSuite()

    suite.addTest(TestComputeReturnsFunctions("test_compute_returns"))
    suite.addTest(TestComputeReturnsFunctions("test_compute_pct_returns"))

    for t in ("empyrical", "glassocv", "ledoit_wolf"):
        suite.addTest(TestEstimateCorrelation(f"test_{t}"))

    suite.addTest(TestComputeRobustDistance("test_call"))

    tests = (
        "sharpe_ratio",
        "sortino_ratio",
        "downside_risk",
        "max_drawdown",
        "var",
        "cvar",
        "evar",
        "tail_ratio",
        "omega_ratio",
        "calmar_ratio",
        "final_pnl",
        "final_pnl_percentage",
        "stability_of_timeseries",
        "raroc",
    )

    for t in tests:
        suite.addTest(TestBootstrapMetric(f"test_compute_{t}"))

    TestBootstrapMetric("test_callable")

    suite.addTest(TestGetScorecard("test_call"))

    return suite


if __name__ == "__main__":
    """The main script"""
    runner = xmlrunner.XMLTestRunner(output=xml_test_folder)
    runner.run(build_suite())
