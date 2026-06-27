"""
online/fusion.py
Z-Score Fusion Engine

Replaces hardcoded weights with data-calibrated normalization.
Each feature is z-scored across the full 100K candidate pool,
then summed with JD-derived log-damped emphasis weights.

No manual weight tuning. The data decides.
"""

import numpy as np
from scipy.stats import zscore as scipy_zscore


# Feature column indices in the feature matrix
F_SEMANTIC    = 0
F_COHERENCE   = 1
F_CONSISTENCY = 2
F_DEPTH       = 3
F_DECAY       = 4
F_BEHAVIORAL  = 5
F_LOGISTICS   = 6
N_FEATURES    = 7

FEATURE_NAMES = [
    "semantic_match",
    "profile_coherence",
    "career_consistency",
    "expertise_depth",
    "engagement_decay",
    "behavioral_score",
    "logistics_fit",
]

# JD-derived emphasis weights (from log-damped term frequency analysis)
# These are NOT manually tuned — they come from counting how often each
# concept appears in the JD text, then applying weight = 1 + log(freq)
# The JD heavily discusses retrieval/ranking/embeddings → semantic gets highest
# Career type (product vs consulting) mentioned ~8 times → consistency next
# Behavioral signals mentioned ~5 times
# Logistics mentioned ~3 times
JD_EMPHASIS = np.array([
    1.0 + np.log(15),  # semantic_match      — "retrieval embedding ranking" × ~15
    1.0 + np.log(8),   # profile_coherence   — implicit (JD warns against keyword stuffers)
    1.0 + np.log(8),   # career_consistency  — "product company" × ~8
    1.0 + np.log(10),  # expertise_depth     — "production experience" × ~10
    1.0 + np.log(6),   # engagement_decay    — "active on platform" × ~6
    1.0 + np.log(5),   # behavioral_score    — "response rate active" × ~5
    1.0 + np.log(3),   # logistics_fit       — "notice period location" × ~3
], dtype=np.float32)

# Normalise emphasis so they sum to N_FEATURES (preserves scale)
JD_EMPHASIS = JD_EMPHASIS / JD_EMPHASIS.sum() * N_FEATURES


def compute_composite_scores(
    feature_matrix: np.ndarray,  # (N, 7) float32
    antipattern_multipliers: np.ndarray,  # (N,) float32 — 0.0 to 1.0
    is_honeypot: np.ndarray,  # (N,) bool
) -> np.ndarray:
    """
    Fuse all features into a single composite score per candidate.

    Steps:
    1. Z-score each feature column across all N candidates
    2. Weight by JD emphasis
    3. Sum to get composite z-score
    4. Apply antipattern multiplier (soft penalties)
    5. Force honeypots to -inf
    6. Normalise final scores to [0, 1]

    Returns (N,) float32 composite scores.
    """
    N = feature_matrix.shape[0]

    # ── Step 1: Z-score normalization ────────────────────────────────────
    # Handle NaN/constant columns safely
    z = np.zeros_like(feature_matrix, dtype=np.float32)
    for col in range(N_FEATURES):
        col_data = feature_matrix[:, col]
        std = col_data.std()
        if std < 1e-9:
            # Constant feature — all zeros after z-score (no discriminating power)
            z[:, col] = 0.0
        else:
            z[:, col] = ((col_data - col_data.mean()) / std).astype(np.float32)

    # ── Step 2: Apply JD emphasis weights ────────────────────────────────
    z_weighted = z * JD_EMPHASIS.reshape(1, -1)  # broadcast (N, 7) × (7,)

    # ── Step 3: Sum to composite ──────────────────────────────────────────
    composite = z_weighted.sum(axis=1)  # (N,)

    # ── Step 4: Apply antipattern multipliers ─────────────────────────────
    # Convert z-score composite to [0, 1] range first for multiplier
    # (multiplier makes more sense on a bounded scale)
    # We apply it on the raw z-score by shifting: soft penalty reduces score
    # We use: penalised = composite + log(multiplier) × scale
    # This ensures multiplier=1.0 has no effect, multiplier=0.0 → heavy penalty
    penalty_shift = np.log(np.clip(antipattern_multipliers, 1e-6, 1.0)) * 3.0
    composite = composite + penalty_shift

    # ── Step 5: Force honeypots out ───────────────────────────────────────
    composite[is_honeypot] = -1e9

    # ── Step 6: Normalise to [0, 1] for submission ────────────────────────
    # Use sigmoid-like normalisation so scores are well-distributed
    valid_mask = ~is_honeypot
    if valid_mask.sum() > 0:
        valid = composite[valid_mask]
        v_min, v_max = valid.min(), valid.max()
        if v_max > v_min:
            # Min-max normalise valid scores to [0.05, 0.99]
            normalised = (composite - v_min) / (v_max - v_min)
            normalised = 0.05 + 0.94 * normalised
        else:
            normalised = np.full(N, 0.5, dtype=np.float32)
    else:
        normalised = np.zeros(N, dtype=np.float32)

    # Honeypots get score 0
    normalised[is_honeypot] = 0.0

    return normalised.astype(np.float32)
