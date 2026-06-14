#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROBE — Probabilistic Occupancy BEV Encoding for 3D Place Recognition

Related Publication:
  J. Lee, B. Lee, and G. Yoo,
  "PROBE: Probabilistic Occupancy BEV Encoding with Analytical
   Translation Robustness for 3D Place Recognition,"
  IEEE Robotics and Automation Letters (RA-L), 2026.
  DOI: 10.1109/LRA.2026.3703245

Author:  Jinseop Lee (jinseop.llee@gmail.com)

Pipeline
========
1. BEV Polar Grid Construction (max-height encoding, vectorized) (Sec. III-A)
2. Jacobian-derived adaptive Gaussian blur → Bernoulli(μ, σ) per cell (Sec. III-B)
   - Angular blur: σ_θ = σ_t / (r · Δθ)  (distance-adaptive) (Eq. 5)
   - Radial blur:  σ_r = σ_t / Δr         (uniform) (Eq. 6)
3. Rotation-invariant retrieval key (KD-tree pre-filter) (Sec. III-C)
4. Height CC → rotation alignment δ* (Sec. III-D.1), cosine similarity C (Sec. III-D.3)
5. Bernoulli-KL Jaccard at aligned δ* → similarity J_KL (Sec. III-D.2)
6. S_PROBE = J_KL · C (Eq. 16)
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

# ================================================================
#  Hyper-parameters (Table I in Paper)
# ================================================================
N_RINGS        = 40    # radial bins (R)
N_SECTORS      = 60    # azimuthal bins (S)
MAX_RANGE      = 80.0  # [m] maximum radial range (R_max)
SIGMA_T        = 2.0   # [m] translation uncertainty (Gaussian std, σ_t)
EPS_BERNOULLI  = 1e-6  # clamp for Bernoulli log stability
EPS_NORM       = 1e-9  # numerical guard for division
VOXEL_SIZE     = 0.5   # [m] voxel downsample (consistent with SC++)


# ================================================================
#  Grid Construction
# ================================================================

def _make_grid(points: np.ndarray, n_sectors: int = N_SECTORS):
    """Build (R, S) BEV polar grid from (already downsampled) point cloud.
    (Sec. III-A, Eq. 1)

    Returns:
        grid:     (R, S) float32  — max-height per cell (G)
        occ_mask: (R, S) float32  — binary occupancy (O)
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r_xy = np.sqrt(x ** 2 + y ** 2)

    valid = r_xy <= MAX_RANGE
    x, y, z, r_xy = x[valid], y[valid], z[valid], r_xy[valid]

    max_z    = np.full((N_RINGS, n_sectors), -np.inf, dtype=np.float32)
    occ_mask = np.zeros((N_RINGS, n_sectors), dtype=np.float32)

    ring = np.clip((r_xy / MAX_RANGE * N_RINGS).astype(int), 0, N_RINGS - 1)
    theta = np.arctan2(y, x)
    sector = np.clip(
        ((theta + np.pi) / (2 * np.pi) * n_sectors).astype(int),
        0, n_sectors - 1,
    )

    np.maximum.at(max_z, (ring, sector), z)
    occ_mask[ring, sector] = 1.0

    grid = np.where(occ_mask > 0, max_z, 0.0).astype(np.float32)
    return grid, occ_mask


# ================================================================
#  Polar Adaptive Blur
# ================================================================

def _polar_adaptive_blur(occ_mask: np.ndarray, sigma_t: float = SIGMA_T, n_sectors: int = N_SECTORS):
    """Jacobian-derived adaptive blur → occupancy probability + Bernoulli σ.
    (Sec. III-B, Eq. 3-8)

    Cartesian uncertainty (dx, dy) ~ N(0, σ_t²I) induces:
        Δr ~ N(0, σ_t²)        — constant radial uncertainty
        Δθ ~ N(0, σ_t²/r₀²)   — distance-dependent angular uncertainty

    Returns:
        mu:      (R, S) float32  ∈ [0, 1] — expected occupancy probability (μ)
        sigma:   (R, S) float32  σ = √(μ(1-μ)) — Bernoulli uncertainty (σ)
    """
    ring_width = MAX_RANGE / N_RINGS
    sec_width  = 2.0 * np.pi / n_sectors
    mu         = occ_mask.astype(np.float64).copy()

    if sigma_t <= 0:
        mu = mu.astype(np.float32)
        return mu, np.zeros_like(mu)

    ring_density = occ_mask.mean(axis=1)  # (R,) -> ρ(r)

    # Angular blur (Eq. 5): σ_θ(r) = σ_eff / (r · Δθ), density-scaled (Eq. 7)
    for r in range(N_RINGS):
        r_center = (r + 0.5) * ring_width
        rho = max(ring_density[r], 0.01)
        sigma_eff = sigma_t * np.sqrt(rho)
        sigma_theta = max((sigma_eff / r_center) / sec_width, 0.1)
        mu[r, :] = gaussian_filter1d(mu[r, :], sigma=sigma_theta, mode='wrap')

    # Radial blur (Eq. 6): σ_r = σ_t / Δr (uniform across sectors)
    sigma_r = sigma_t / ring_width
    for s in range(n_sectors):
        mu[:, s] = gaussian_filter1d(mu[:, s], sigma=sigma_r, mode='constant')

    mu = np.clip(mu, 0.0, 1.0).astype(np.float32)
    # Bernoulli standard deviation (Eq. 8)
    sigma = np.sqrt(mu * (1.0 - mu)).astype(np.float32)
    return mu, sigma


# ================================================================
#  PROBENode
# ================================================================

class PROBENode:
    """Per-scan probabilistic descriptor.

    Attributes:
        grid          (R, S)         max-height BEV polar grid (G)
        grid_fft      (R, S//2+1)    rFFT of grid (azimuth axis)
        occ_mask      (R, S)         binary occupancy (O)
        mu            (R, S)         expected occupancy probability (μ)
        sigma         (R, S)         Bernoulli uncertainty σ = √(μ(1-μ))
        retrieval_key (2R,)          ring-mean key for KD-tree pre-filter (k)
        n_sectors     int            azimuthal bins used for this node (S)
    """
    __slots__ = [
        'grid', 'grid_fft', 'norm_h',
        'occ_mask', 'mu', 'sigma',
        'retrieval_key', 'n_sectors',
    ]

    def __init__(self, points: np.ndarray, sigma_t: float = None, n_sectors: int = None, fov_deg: float = 360.0):
        if sigma_t is None:
            sigma_t = SIGMA_T
        if n_sectors is None:
            n_sectors = N_SECTORS
        self.n_sectors = n_sectors

        # Voxel downsample
        if VOXEL_SIZE > 0 and len(points) > 0:
            keys = np.floor(points[:, :3] / VOXEL_SIZE).astype(np.int32)
            _, idx = np.unique(keys, axis=0, return_index=True)
            points = points[idx]

        # ── FOV-aware pipeline ────────────────────────────────
        # If fov_deg < 360, filter points to observed FOV BEFORE grid
        # construction so that Gaussian blur operates only on observed
        # data.  Unobserved sectors are then set to maximum Bernoulli
        # uncertainty (μ=0.5, σ=0.5) AFTER blurring.
        _fov_limited = (fov_deg is not None and fov_deg < 360.0)
        if _fov_limited and len(points) > 0:
            _theta = np.arctan2(points[:, 1], points[:, 0])
            _half = np.deg2rad(fov_deg / 2.0)
            points = points[np.abs(_theta) <= _half]

        # Grid + blur (now only from observed-FOV points)
        self.grid, self.occ_mask = _make_grid(points, n_sectors=n_sectors)
        self.mu, self.sigma = _polar_adaptive_blur(self.occ_mask, sigma_t, n_sectors=n_sectors)

        if _fov_limited:
            fov_rad = np.deg2rad(fov_deg)
            idx_min = int((-fov_rad / 2.0 + np.pi) / (2 * np.pi) * n_sectors)
            idx_max = int(( fov_rad / 2.0 + np.pi) / (2 * np.pi) * n_sectors)
            # Mask unobserved sectors with maximum uncertainty (0.5)
            self.mu[:, :idx_min] = 0.5
            self.sigma[:, :idx_min] = 0.5
            self.grid[:, :idx_min] = 0.0
            self.mu[:, idx_max:] = 0.5
            self.sigma[:, idx_max:] = 0.5
            self.grid[:, idx_max:] = 0.0

        # Height FFT + norm (rotation alignment)
        self.grid_fft = np.fft.rfft(self.grid, axis=1)
        self.norm_h = float(np.sum(self.grid ** 2))

        # Retrieval key: ring-wise mean height + occ mean (rotation-invariant, 80D)
        # k = [G_bar || μ_bar] (Eq. 9)
        z_ring = self.grid.mean(axis=1)  # G_bar
        mu_ring = self.mu.mean(axis=1)   # μ_bar
        self.retrieval_key = np.concatenate([z_ring, mu_ring]).astype(np.float32)


# ================================================================
#  Rotation Alignment
# ================================================================

def _rotation_alignment(m_node: PROBENode, q_node: PROBENode):
    """Height cross-correlation for rotation alignment.
    (Sec. III-D.1, Eq. 10-11)

    Returns:
        None | (delta_star: int, CC[delta_star]: float)
    """
    S = m_node.n_sectors

    cc_unnorm = np.fft.irfft(
        (m_node.grid_fft * np.conj(q_node.grid_fft)).sum(axis=0), n=S)
    d_h = np.sqrt(m_node.norm_h * q_node.norm_h)
    if d_h < EPS_NORM:
        return None

    cc = cc_unnorm / d_h
    delta_star = int(np.argmax(cc))
    c_sim = float(np.clip(cc[delta_star], 0.0, 1.0)) # C(δ*)

    return delta_star, c_sim


# ================================================================
#  Bernoulli-KL Jaccard
# ================================================================

def _bernoulli_kl_jaccard(m_node: PROBENode, q_node: PROBENode, shift: int):
    """Symmetric KL divergence on shrinkage-regularized Bernoulli occupancy.
    (Sec. III-D.2, Eq. 12-14)

    Each cell's Bernoulli parameter is shrunk toward 0.5 by its σ,
    so uncertain cells contribute negligible KL divergence naturally.

    Returns:
        J_KL ∈ [0, 1]:  1 = identical,  0 = completely different.
    """
    S = m_node.n_sectors
    idx = (np.arange(S) - shift) % S

    mu_m = m_node.mu.astype(np.float64)
    sigma_m = m_node.sigma.astype(np.float64)
    mu_q = q_node.mu[:, idx].astype(np.float64)
    sigma_q = q_node.sigma[:, idx].astype(np.float64)

    # Soft union mask U (includes cells where blurred prob has support)
    union_mask = (mu_m + mu_q) > 1e-3
    if not np.any(union_mask):
        return 0.0

    eps = EPS_BERNOULLI
    # Shrinkage (Eq. 12): p_m, p_q
    p_m = np.clip(mu_m[union_mask] * (1 - sigma_m[union_mask])
                  + 0.5 * sigma_m[union_mask], eps, 1 - eps)
    p_q = np.clip(mu_q[union_mask] * (1 - sigma_q[union_mask])
                  + 0.5 * sigma_q[union_mask], eps, 1 - eps)

    # Symmetric KL per cell (Eq. 13)
    kl_mq = p_m * np.log(p_m / p_q) + (1 - p_m) * np.log((1 - p_m) / (1 - p_q))
    kl_qm = p_q * np.log(p_q / p_m) + (1 - p_q) * np.log((1 - p_q) / (1 - p_m))
    d_kl = 0.5 * (kl_mq + kl_qm)

    # J_KL (Eq. 14)
    n = len(p_m)
    return float(np.clip(np.exp(-np.sum(d_kl) / n), 0.0, 1.0))


# ================================================================
#  Scoring
# ================================================================

def compute_score(m_node: PROBENode, q_node: PROBENode):
    """PROBE distance: d = 1 − S_PROBE. (Eq. 16)

    S_PROBE = J_KL · C
        J_KL = Bernoulli-KL Jaccard (occupancy agreement, Sec. III-D.2)
        C    = Height cosine similarity (structural shape, Sec. III-D.3)

    Returns:
        float ∈ [0, 1]:  0 = identical,  1 = completely different.
    """
    result = _rotation_alignment(m_node, q_node)
    if result is None:
        return 1.0
    delta_star, c_sim = result

    j_kl = _bernoulli_kl_jaccard(m_node, q_node, delta_star)
    s_probe = j_kl * c_sim

    return float(np.clip(1.0 - s_probe, 0.0, 1.0))
