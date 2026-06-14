"""PROBE — Probabilistic Occupancy BEV Encoding for 3D Place Recognition.

Public API:
    PROBENode      — per-scan probabilistic BEV descriptor
    compute_score  — similarity distance between two PROBENode descriptors
"""

from .probe import PROBENode, compute_score

__all__ = ["PROBENode", "compute_score"]
__version__ = "1.0.0"
