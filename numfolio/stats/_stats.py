import numba
import numpy as np

annualized_factor = np.sqrt(252.0)


@numba.njit("float64[:](float64[:])", cache=True)
def _compute_pnl(returns: np.ndarray) -> np.ndarray:
    """
    Compute cumulative PNL from input returns.

    Args:
        returns: array-like of returns

    Returns:
        cumulative PNL array

    Examples:
        >>> _compute_pnl(np.array([0.01, -0.02, 0.03]))
        array([0.01, -0.01, 0.02])

        >>> _compute_pnl(np.array([0.1, 0.1, 0.1]))
        array([0.1, 0.2, 0.3])

    """
    pnl = returns[np.isfinite(returns)].cumsum()
    return pnl[np.isfinite(pnl)]


@numba.njit("float64[:](float64[:])", cache=True)
def _compute_loss(returns: np.ndarray) -> np.ndarray:
    """
    Compute losses (negatives) from input returns.

    Args:
        returns: array-like of returns

    Returns:
        losses array (negatives of returns)

    Examples:
        >>> _compute_loss(np.array([0.01, -0.02, 0.03]))
        array([-0.01,  0.02, -0.03])

        >>> _compute_loss(np.array([-0.05, 0.1, 0.0]))
        array([0.05, -0.1 , -0. ])

    """
    return -returns[np.isfinite(returns)]


if __name__ == "__main__":
    pass
