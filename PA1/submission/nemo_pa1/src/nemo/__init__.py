from ._version import __version__

__all__ = [
    "__version__",
]

from . import solvers

__all__ += [
    "solvers",
]
