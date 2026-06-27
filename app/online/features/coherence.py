"""
online/features/coherence.py
Feature 2 — Profile Coherence Score (NOVEL SIGNAL)

Detects whether a candidate's skills are semantically consistent
with their current job title.

A Marketing Analyst with TensorFlow/Kubernetes/Blockchain = low coherence (~0.15)
An ML Engineer with PyTorch/Transformers/FAISS            = high coherence (~0.85)

This catches the most common honeypot pattern: copied AI keywords
on an otherwise irrelevant profile.
"""

import numpy as np


def compute_profile_coherence(
    candidate_idx: int,
    skill_embs_flat: np.ndarray,
    skill_offsets: np.ndarray,
    title_embs: np.ndarray,
) -> float:
    """
    Returns coherence score in [0, 1].

    Method:
      mean cosine_similarity(title_embedding, each_skill_embedding)

    Since all embeddings are L2-normalised, cosine_sim = dot product.
    """
    title_emb = title_embs[candidate_idx]  # (384,)

    start, end = int(skill_offsets[candidate_idx, 0]), int(skill_offsets[candidate_idx, 1])
    if end <= start:
        return 0.5  # no skills — neutral

    skill_embs = skill_embs_flat[start:end]  # (k, 384)

    # cosine similarities (dot products, embeddings are normalised)
    sims = skill_embs @ title_emb  # (k,)

    # Clip negatives to 0 (anti-correlation isn't more incoherent than zero)
    sims = np.clip(sims, 0.0, 1.0)

    # Mean coherence
    score = float(sims.mean())
    return float(np.clip(score, 0.0, 1.0))
