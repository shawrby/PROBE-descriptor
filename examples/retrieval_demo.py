#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PROBE place-recognition retrieval example.

Reproduces the retrieval protocol used in the paper:
  1. Build a database of PROBE descriptors from map scans.
  2. KD-tree pre-filter: top-K nearest ring-mean retrieval keys (Sec. III-C).
  3. Re-rank the K candidates with the full PROBE score and take the
     minimum distance as the match (Sec. III-D).

The query is a rotated + translated + noisy observation of one database
place; the pipeline should retrieve that place as the top match.
"""

import numpy as np
from scipy.spatial import cKDTree

from probe import PROBENode, compute_score

SIGMA_T = 2.0
N_PLACES = 80
K_PREFILTER = 50   # top-K KD-tree candidates, as in the paper
GT_INDEX = 42      # the database place the query actually revisits


def make_place(seed, n_clusters=14, pts_per_cluster=500):
    """Synthetic structured scan: vertical point columns at random 2D anchors
    plus a flat ground plane. Each seed yields a distinct 'place'."""
    rng = np.random.default_rng(seed)
    anchors = rng.uniform(-45.0, 45.0, size=(n_clusters, 2))
    pts = []
    for cx, cy in anchors:
        x = cx + rng.normal(0.0, 0.8, pts_per_cluster)
        y = cy + rng.normal(0.0, 0.8, pts_per_cluster)
        z = rng.uniform(0.0, 8.0, pts_per_cluster)
        pts.append(np.stack([x, y, z], axis=1))
    # ground plane
    g = rng.uniform(-45.0, 45.0, size=(4000, 2))
    pts.append(np.stack([g[:, 0], g[:, 1], np.zeros(len(g))], axis=1))
    return np.vstack(pts).astype(np.float32)


def transform(points, yaw_deg, translation, noise=0.1, seed=0):
    """Apply yaw rotation + translation + sensor noise to a scan."""
    rng = np.random.default_rng(seed)
    c, s = np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg))
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    xy = points[:, :2] @ R.T + np.asarray(translation, dtype=np.float32)
    z = points[:, 2:3]
    out = np.hstack([xy, z]) + rng.normal(0.0, noise, points.shape)
    return out.astype(np.float32)


def main():
    # 1. Build the database of PROBE descriptors.
    db = [PROBENode(make_place(i), sigma_t=SIGMA_T) for i in range(N_PLACES)]
    keys = np.stack([node.retrieval_key for node in db])

    # 2. KD-tree pre-filter over the ring-mean retrieval keys (Sec. III-C).
    tree = cKDTree(keys)

    # Query = place GT_INDEX revisited under a different heading and pose.
    query_scan = transform(make_place(GT_INDEX), yaw_deg=35.0,
                            translation=(3.0, -2.0), noise=0.1, seed=999)
    query = PROBENode(query_scan, sigma_t=SIGMA_T)

    k = min(K_PREFILTER, N_PLACES)
    _, cand_idx = tree.query(query.retrieval_key, k=k)
    print(f"KD-tree top-{k} candidates (first 10): "
          f"{[int(c) for c in cand_idx[:10]]} ...")

    # 3. Re-rank candidates with the full PROBE score, take the minimum.
    best_dist, best_i = 1.0, -1
    for c in cand_idx:
        d = compute_score(db[int(c)], query)
        if d < best_dist:
            best_dist, best_i = d, int(c)

    print(f"\nBest match: place {best_i}  (PROBE distance {best_dist:.4f})")
    print(f"Ground-truth place: {GT_INDEX} | retrieved: {best_i} | "
          f"{'CORRECT' if best_i == GT_INDEX else 'WRONG'}")


if __name__ == "__main__":
    main()
