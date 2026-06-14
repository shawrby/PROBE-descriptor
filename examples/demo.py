#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal PROBE usage example.

Generates two synthetic point clouds related by a small translation,
extracts PROBE descriptors, and computes their similarity distance.
"""

import numpy as np

from probe import PROBENode, compute_score


def main():
    # 1. Load an N x 3 (or N x 4) point cloud frame.
    #    Here we use dummy data representing two sequential frames.
    pc_map = np.random.rand(100_000, 3) * 50.0
    pc_query = pc_map + np.array([0.5, 0.2, 0.0])  # slight translation

    # 2. Extract PROBE descriptors.
    #    sigma_t sets the translation uncertainty threshold (default: 2.0 m).
    node_m = PROBENode(pc_map, sigma_t=2.0)
    node_q = PROBENode(pc_query, sigma_t=2.0)

    # 3. Compute similarity distance (0 = exact match, 1 = max distance).
    distance = compute_score(node_m, node_q)
    print(f"PROBE Distance: {distance:.4f}")


if __name__ == "__main__":
    main()
