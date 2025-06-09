"""
--------
numfolio
--------

Portfolio performance package accelerated by Numba
"""

from pathlib import Path
from importlib.metadata import version

from .stats import *
from .core import *


def _read_version() -> str:
    """Read version from metadata or pyproject.toml"""

    try:
        return version("numfolio")

    except Exception:  # pragma: no cover
        # For development
        file = Path(__file__).absolute().parents[1] / "pyproject.toml"

        if file.exists():
            with open(file, "r") as f:
                lines = f.read().splitlines()
                for line in lines:
                    if line.startswith("version"):
                        return line.split("=")[-1].strip()

        return "0.x"


__version__ = _read_version()
__all__ = [itm for itm in dir() if not itm.startswith("_")]
