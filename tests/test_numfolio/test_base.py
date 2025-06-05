"""
------------
test_base.py
------------

This is a dummy test to check the import

To run the code
$ python test_base.py
"""

import unittest
import importlib

from test_numfolio.config import xml_test_folder


class TestImport(unittest.TestCase):
    """The base class for import test"""

    def test_import(self):
        """Test the import using dynamical import"""

        nf = importlib.import_module("numfolio")
        self.assertIsInstance(nf.__version__, str)


def build_suite():
    """Build the TestSuite"""
    suite = unittest.TestSuite()

    suite.addTest(TestImport("test_import"))

    return suite


if __name__ == "__main__":
    """The main script"""

    runner = unittest.TextTestRunner(failfast=True, output=xml_test_folder)
    runner.run(build_suite())
