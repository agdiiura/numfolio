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
