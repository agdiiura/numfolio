"""
-------------
test_stats.py
-------------

A test to check the stats module

To run the code
$ python test_stats.py
"""

import os
import unittest

from inspect import signature

import numpy as np
import pandas as pd
import xmlrunner

from test_numfolio.config import xml_test_folder

from numfolio import stats

SEED = int(os.environ.get("SEED", 8))
rng = np.random.default_rng(SEED)


class TestCompileNumbaFunctions(unittest.TestCase):
    """Class for test the compilation"""

    def test_call(self):
        """Test the compile_numba_functions function"""
        obj = stats.compile_numba_functions()
        self.assertIsInstance(obj, dict)

        expected_keys = [
            f
            for f in dir(stats)
            if callable(getattr(stats, f)) and f.startswith("compute_")
        ]
        self.assertSetEqual(set(expected_keys), set(obj.keys()))


class TestComputeStats(unittest.TestCase):
    """Class for the test of the stats module"""

    @classmethod
    def setUpClass(cls):
        """Configure the test"""

        cls.size = 252
        cls.repeat = 5
        cls.n_nans = 3

        cls.kwargs_list = list()
        for r in range(cls.repeat):
            values = rng.standard_t(df=6, size=cls.size)
            for n in range(cls.n_nans):
                idx = rng.integers(0, cls.size)
                values[idx] = np.nan

            alpha = rng.uniform(0.01, 0.1)
            r = rng.uniform(0.01, 0.1)
            cls.kwargs_list.append({"returns": values, "alpha": alpha, "r": r})

    def _common_test(self, function):
        """Test common functionality"""

        f = getattr(stats, f"compute_{function}")
        self.assertTrue(callable(f))
        sig = signature(f)

        self.assertIn("returns", sig.parameters)

        for k, kwargs in enumerate(self.kwargs_list):
            kw = kwargs.copy()
            kw = {key: val for key, val in kw.items() if key in sig.parameters}

            r = f(**kw)
            msg = (
                f"Error with run `{k}`; result = {r}\nreturns: {kw['returns'][:10]}..."
                f"\nalpha: {kw.get('alpha', None)}\nr: {kw.get('r', None)}"
            )
            self.assertIsInstance(r, float, msg=msg)
            self.assertTrue(pd.notnull(r), msg=msg)

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

    def test_compute_average_drawdown(self):
        """Test the compute_average_drawdown function"""
        self._common_test("average_drawdown")

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

    def test_compute_sterling_ratio(self):
        """Test the compute_sterling_ratio function"""
        self._common_test("sterling_ratio")

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


def build_suite():
    """Construct the TestSuite"""

    suite = unittest.TestSuite()

    suite.addTest(TestCompileNumbaFunctions("test_call"))

    tests = (
        "sharpe_ratio",
        "sortino_ratio",
        "downside_risk",
        "max_drawdown",
        "average_drawdown",
        "var",
        "cvar",
        "evar",
        "tail_ratio",
        "omega_ratio",
        "calmar_ratio",
        "sterling_ratio",
        "final_pnl",
        "final_pnl_percentage",
        "stability_of_timeseries",
        "raroc",
    )

    for t in tests:
        suite.addTest(TestComputeStats(f"test_compute_{t}"))

    return suite


if __name__ == "__main__":
    """The main script"""
    runner = xmlrunner.XMLTestRunner(output=xml_test_folder)
    runner.run(build_suite())
