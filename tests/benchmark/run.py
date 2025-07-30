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

N_TESTS = 100


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
    n_returns = [10, 100, 1000, 10000]  # Different sizes of returns to test

    for func, n_ret in tqdm(product(functions, n_returns)):
        out = {"function": func.__name__}

        times = list()
        for _ in range(n_tests):
            # Generate random returns for testing
            returns = rng.standard_t(4, n_ret)
            times.append(benchmark_function(func, returns))

        out["min_time"] = np.min(times)
        out["max_time"] = np.max(times)
        out["mean_time"] = np.mean(times)
        out["std_time"] = np.std(times)
        out["median_time"] = np.median(times)
        out["n_nans"] = np.sum(np.isnan(times))
        out["n_tests"] = n_tests
        out["n_returns"] = n_ret
        out["datetime"] = pd.Timestamp.now()

        statistics.append(out)

    # Convert to DataFrame for better visualization
    df = pd.DataFrame(statistics)
    filename = (
        results_path
        / f"benchmark_results-{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    )

    df.to_csv(filename, index=False)

    print(df)


def benchmark_function(func, *args, **kwargs) -> float:
    """
    Benchmark a function by measuring its execution time.

    Args:
        func: The function to benchmark.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        The execution time in seconds.

    """
    try:
        start_time = timeit.default_timer()
        func(*args, **kwargs)
        end_time = timeit.default_timer()
        return end_time - start_time
    except Exception as e:
        return np.nan


if __name__ == "__main__":
    logger.info("Starting benchmarks for numfolio stats module...")
    run_benchmarks()

    logger.info(f"Benchmarks completed. Results saved to {results_path}.")
