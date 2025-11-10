"""
------
run.py
------

Benchmarking script for the numfolio stats module.
"""

import timeit

from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import quirtylog

from tqdm import tqdm

from numfolio import stats

logger = quirtylog.create_logger(config_file="logger-config.yaml")

rng = np.random.default_rng()

results_path = Path(__file__).parent.absolute() / "results"
results_path.mkdir(parents=True, exist_ok=True)

N_TESTS = 250


def run_benchmarks(n_tests: int = N_TESTS):
    """
    Run benchmarks for the numfolio stats module.

    This function will execute various statistical computations and measure their execution time.
    """
    statistics = list()

    functions = [
        getattr(stats, f)
        for f in dir(stats)
        if callable(getattr(stats, f)) and f.startswith("compute_")
    ]

    # Different sizes of returns to test
    n_returns = [10, 100, 1000, 10000]
    numba_flag = [True, False]

    for func, n_ret, use_numba in tqdm(
        product(functions, n_returns, numba_flag),
        total=len(functions) * len(n_returns) * len(numba_flag),
    ):
        out = {"function": func.__name__, "use_numba": use_numba, "n_returns": n_ret}

        times = list()
        for _ in range(n_tests):
            # Generate random returns for testing
            returns = rng.standard_t(4, n_ret)
            times.append(benchmark_function(func, returns, use_numba=use_numba))

        out["min_time"] = np.min(times)
        out["max_time"] = np.max(times)
        out["mean_time"] = np.mean(times)
        out["std_time"] = np.std(times)
        out["median_time"] = np.median(times)
        out["n_nans"] = np.sum(np.isnan(times))
        out["n_tests"] = n_tests
        out["datetime"] = pd.Timestamp.now()

        statistics.append(out)

    # Convert to DataFrame for better visualization
    df = pd.DataFrame(statistics)
    filename = (
        results_path
        / f"benchmark_results-{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    )

    df.to_csv(filename, index=False)

    logger.info(f"Benchmark results saved to {filename}")
    logger.info(df)


def benchmark_function(func, returns: np.ndarray, use_numba: bool = True) -> float:
    """
    Benchmark a function by measuring its execution time.

    Args:
        func: The function to benchmark.
        returns: The input data for the function.
        use_numba: Whether to use the numba-compiled version of the function.

    Returns:
        The execution time in seconds.

    """
    try:
        start_time = timeit.default_timer()
        if use_numba:
            func(returns)
        else:
            func.py_func(returns)
        end_time = timeit.default_timer()
        return end_time - start_time
    except Exception as e:
        return np.nan


if __name__ == "__main__":
    logger.info("Starting benchmarks for numfolio stats module...")
    run_benchmarks()

    logger.info(f"Benchmarks completed. Results saved to {results_path}.")
