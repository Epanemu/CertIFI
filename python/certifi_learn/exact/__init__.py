"""Exact branch-and-bound search for the conjunction subgroup with maximum
violation of MSD, SPSF or FPSF (implemented in Rust, no MIO solver needed).

See ``docs/subgroup_pruning_proof.tex`` for the algorithm and its proof of
correctness.
"""

from ._api import fpsf, msd, spsf
from ._common import Subgroup

__all__ = ["msd", "spsf", "fpsf", "Subgroup"]
