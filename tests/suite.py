"""
--------
suite.py
--------

A script to execute and manage tests defined in the 'test_numfolio' package.

Usage:
Run the script in a shell using the following commands:
- To run a single test file:
  $ python suite.py -t <path/to/test.py>
- To run all test files in a folder:
  $ python suite.py -t <path/to/folder/>

Dependencies:
- pandas: Required for timestamp handling and summary creation.
- xmlrunner: Required for generating XML test reports.
- colorama: Used for colored console output.
- test_numfolio.config: Import required configurations from the 'test_numfolio' package.

.. note::
    Ensure that the required dependencies are installed before running this script.

"""

import sys
import argparse
import unittest
import warnings
import importlib

from pathlib import Path

import pandas as pd
import xmlrunner

from colorama import Back, Style
from test_numfolio.config import xml_test_folder

warnings.filterwarnings("ignore")


class ErrorUnittest(unittest.TestCase):
    """A class used to raise error"""

    def __init__(self, test_name: str, module: str, exception: Exception):
        """Override the default constructor"""
        print("\n\n>>> ERROR!\n\n")
        super().__init__(test_name)
        self._module = module
        self._exception = exception

    def test_raise_error(self):
        """Raise an error"""
        self.assertTrue(
            False,
            msg=f"Error in loading `{self._module}`. "
            f"{self._exception.__class__.__name__}: {self._exception}",
        )


def build_error_suite(module: Path, exception: Exception) -> unittest.TestSuite:
    """Build a TestSuite object"""

    suite = unittest.TestSuite()
    suite.addTest(
        ErrorUnittest("test_raise_error", module=str(module), exception=exception)
    )

    return suite


def make_summary(test_results: dict):
    """Make a pretty summary for multiple test"""

    print("\n\n")
    for t, r in test_results.items():
        if len(r.errors) > 0 or len(r.failures) > 0:
            print(f"{Style.DIM + Back.LIGHTRED_EX}Error with `{t}`{Style.RESET_ALL}")

    if any(len(r.errors) > 0 or len(r.failures) > 0 for r in test_results.values()):
        sys.exit(1)


def run_test(file: Path):
    """
    Execute the test

    Args:
        file: test file path

    """
    runner = xmlrunner.XMLTestRunner(output=xml_test_folder)

    print(
        f"\n{Style.DIM + Back.LIGHTBLUE_EX}{pd.Timestamp.now()}{Style.RESET_ALL}\nReading `{file}` module"
    )
    if sys.platform in ["win32"]:
        target = str(file).replace("\\", ".").replace(".py", "")
    else:
        target = str(file).replace("/", ".").replace(".py", "")
    module = importlib.import_module(target)

    build_suite = getattr(module, "build_suite")
    try:
        suite = build_suite()
    except Exception as e:
        suite = build_error_suite(module=file, exception=e)

    r = runner.run(suite)

    if len(r.errors) > 0 or len(r.failures) > 0:
        print(f"\n{Style.DIM + Back.LIGHTRED_EX}Something went wrong!{Style.RESET_ALL}\n")
    else:
        print(f"\n{Style.DIM + Back.GREEN}All test passed{Style.RESET_ALL}\n")

    return r


if __name__ == "__main__":
    """The main script"""

    parser = argparse.ArgumentParser(description="Run the test")

    parser.add_argument(
        "--test",
        "-t",
        type=Path,
        default="test_numfolio/",
        help="Set the single test or a subpackage",
    )

    args = parser.parse_args()

    filename = Path(args.test)
    if filename.is_file():
        _ = run_test(file=filename)
    else:
        results = dict()
        for test in filename.rglob("test*.py"):
            print("\n\n")
            if test.is_file():
                res = run_test(file=test)
                results[test] = res

        make_summary(results)
