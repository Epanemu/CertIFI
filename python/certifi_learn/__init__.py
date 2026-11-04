"""CertIFI: Certifiably Intersectionally Fair & Interpretable models.

* :func:`find_subgroup` -- detection of the most biased subgroup
  (exact Rust search or Mixed-Integer Optimization).
* :class:`FairLinearClassifier`, :class:`FairDNFClassifier` -- interpretable
  classifiers trained under intersectional fairness constraints.
* :mod:`certifi_learn.measures` -- evaluation of the bias measures for a given subgroup.
"""

from importlib.metadata import PackageNotFoundError, version as _version

from . import exact, measures
from .classifiers import FairDNFClassifier, FairLinearClassifier
from .detection import DetectedSubgroup, find_subgroup

try:
    __version__ = _version("certifi-learn")
except PackageNotFoundError:  # not installed (e.g. running from a source checkout)
    __version__ = "unknown"

__all__ = [
    "find_subgroup",
    "DetectedSubgroup",
    "FairLinearClassifier",
    "FairDNFClassifier",
    "measures",
    "exact",
]
