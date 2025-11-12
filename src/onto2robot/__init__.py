"""onto2robot package.

Public API is intentionally small at start; expand as modules mature.
"""

from .core import load_ontology  # re-export simple example for tests

__all__ = ["load_ontology"]
