# numfolio ⚡

A python package for portfolio evaluation build on top of Numba

## Minimal examples

Compute statistics from input returns:

```python
import numpy as np
from numfolio import compute_sharpe_ratio

returns = np.array([
    0.10002028, -0.52892259, -1.07383022, -0.19794636, -2.17779965,
    -0.28190857,  0.74667829,  0.2446915 ,  0.13643797,  0.4230784
])
sr = compute_sharpe_ratio(returns)
## -5.162601446765606
```

It is also possible to compute bootstrapping statistics
```python
import numpy as np
from numfolio import compute_cvar, bootstrap_metric

returns = np.random.default_rng(42).standard_t(df=4, size=252)
m = bootstrap_metric(returns=returns, metric=compute_cvar)
cvar = m.mean()
## 3.60224242103749
```

You can estimate a correlation (or covariance) matrix from multivariate return data using `estimate_correlation`.
The function accepts a pandas DataFrame of returns (columns = assets) and supports different estimation methods
(e.g. "empyrical", "glassocv", "ledoit_wolf") and optional bootstrapping.

```python
import numpy as np
import pandas as pd
from numfolio import estimate_correlation

rng = np.random.default_rng(42)
n_obs, n_assets = 250, 5

# simulate correlated returns
A = rng.standard_normal((n_assets, n_assets))
cov_true = A @ A.T
returns = rng.multivariate_normal(mean=np.zeros(n_assets), cov=cov_true, size=n_obs)
df = pd.DataFrame(returns, columns=[f"asset_{i}" for i in range(n_assets)])

# estimate correlation matrix (method can be 'empyrical', 'glassocv', 'ledoit_wolf')
corr_est = estimate_correlation(df, method="ledoit_wolf", n_bootstraps=100, n_jobs=-1)

print(corr_est)
# Expected: a pandas DataFrame with shape (n_assets, n_assets) showing the estimated correlations
```


## Installation

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
