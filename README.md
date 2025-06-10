# numfolio ⚡
Portfolio performance accelerated by Numba

A lightweight, flexible Python package for analyzing portfolio returns,
risk metrics, and correlations using modern statistical and machine learning methods.

---

## 🚀 Features

✅ Bootstrapped metric estimation (e.g., Sharpe Ratio, Sortino Ratio)

✅ Automatic Scorecard Generation with time-based aggregation (Yearly, Quarterly, Monthly)

✅ Covariance and Correlation estimation with robust shrinkage methods

✅ Parallel computation for scalability

✅ Clean, consistent API inspired by scikit-learn & pandas


---

## 🔧 Installation

To install the package the simplest procedure is:
```bash
pip install numfolio
```
Now you can test the installation... In a python shell:

```python
import numfolio as nf

nf.__version__
```

Optional dependencies are `docs` for documentation and
`build` for development. To install optional
dependencies `pip install numfolio[docs,build]`.

## 📚 Example Usage

### 1. Compute Scorecard from PnL or Returns:

```python
import pandas as pd
from numfolio import get_scorecard

# Sample PnL data
dates = pd.date_range("2025-01-01", periods=60, freq="D")
pnl = pd.Series(range(100, 160), index=dates)

df = pd.DataFrame({"pnl": pnl})

scorecard = get_scorecard(df, freq="M")
print(scorecard)
```

### 2. Estimate Bootstrapped Sharpe Ratio:

```python
import numpy as np
from numfolio import bootstrap_metric

# Generate fake returns
returns = np.random.default_rng().normal(0, 1, 100)

bootstrapped = bootstrap_metric(returns, metric="sharpe_ratio", n_bootstraps=500)
print("Bootstrapped Sharpe Ratios:", bootstrapped[:5])
```

### 3. Estimate Correlation Matrix:

```python
import pandas as pd
import numpy as np
from numfolio import estimate_correlation

dates = pd.date_range("2025-01-01", periods=100, freq="D")
returns = pd.DataFrame(np.random.default_rng().normal(0, 1, (100, 3)), columns=["A", "B", "C"], index=dates)

correlation = estimate_correlation(returns, method="ledoit_wolf", n_bootstraps=200)
print(correlation)
```
