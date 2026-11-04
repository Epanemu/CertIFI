"""CertIFI: Certifiably Intersectionally Fair & Interpretable models.

* :func:`find_subgroup` -- detection of the most biased subgroup
  (exact Rust search or Mixed-Integer Optimization).
* :class:`FairLinearClassifier`, :class:`FairDNFClassifier` -- interpretable
  classifiers trained under intersectional fairness constraints.
* :mod:`certifi_learn.measures` -- evaluation of the bias measures for a given subgroup.
"""

from . import exact, measures
from .classifiers import FairDNFClassifier, FairLinearClassifier
from .detection import DetectedSubgroup, find_subgroup

__version__ = "0.2.0"

__all__ = [
    "find_subgroup",
    "DetectedSubgroup",
    "FairLinearClassifier",
    "FairDNFClassifier",
    "measures",
    "exact",
]
