# Tests

A package for test.

The simplest way to perform a test is
```bash
python <test_script.py>
```

To run all the test in a single shot just type:
```bash
python suite.py
```


### Coverage
To test the code coverage we use the package [`coverage`](https://coverage.readthedocs.io/en/coverage-5.1/index.html).
To run a test use
```bash
coverage run -m unittest <test_script.py>
```
For the whole suite use `coverage run -m unittest test_*`.
To perform the test and display the report use `bash coverage.sh`

To get the report
```bash
coverage report
```
