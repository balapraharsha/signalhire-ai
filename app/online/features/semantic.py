"""
online/features/semantic.py
Feature 1 — Semantic Match Score

Measures how well a candidate's skills and career summary
align with the JD's intent — not just keyword overlap.
"""

import numpy as np


PROFICIENCY_MAP = {
    "beginner":     0.25,
    "intermediate": 0.55,
    "advanced":     0.80,
    "expert":       1.00,
}


def compute_semantic_match(
    candidate_idx: int,
    skill_embs_flat: np.ndarray,
    skill_offsets: np.ndarray,
    summary_embs: np.ndarray,
    jd_concept_embs: np.ndarray,
    candidate: dict,
) -> float:
    """
    Returns semantic match score in [0, 1].

    Strategy:
      - Embed candidate skills and summary (precomputed)
      - Compute cosine sim vs. JD concept embedding centroid
      - Weight skills by proficiency × duration
      - Blend skill match with summary match
    """
    # ── JD centroid ───────────────────────────────────────────────────────
    # Mean of all JD concept embeddings (already L2-normalised individually)
    jd_centroid = jd_concept_embs.mean(axis=0)
    jd_centroid = jd_centroid / (np.linalg.norm(jd_centroid) + 1e-9)

    # ── Skill match ───────────────────────────────────────────────────────
    start, end = int(skill_offsets[candidate_idx, 0]), int(skill_offsets[candidate_idx, 1])
    if end > start:
        cand_skill_embs = skill_embs_flat[start:end]  # (k, 384)

        # Cosine similarity (embeddings are already normalised)
        sims = cand_skill_embs @ jd_centroid  # (k,)

        # Proficiency weights
        skills = candidate.get("skills", [])
        prof_weights = np.array([
            PROFICIENCY_MAP.get(s.get("proficiency", "intermediate"), 0.55)
            for s in skills
        ], dtype=np.float32)
        if len(prof_weights) != len(sims):
            prof_weights = np.ones(len(sims), dtype=np.float32)

        # Duration weights (capped at 60 months)
        dur_weights = np.array([
            min(s.get("duration_months", 12), 60) / 60.0
            for s in skills
        ], dtype=np.float32)
        if len(dur_weights) != len(sims):
            dur_weights = np.ones(len(sims), dtype=np.float32)

        # Combined weight
        combined = prof_weights * (0.5 + 0.5 * dur_weights)

        # Weighted mean of top-5 skill similarities
        weighted_sims = sims * combined
        top_k = min(5, len(weighted_sims))
        top_indices = np.argpartition(weighted_sims, -top_k)[-top_k:]
        skill_score = float(weighted_sims[top_indices].mean())
        skill_score = np.clip(skill_score, 0.0, 1.0)
    else:
        skill_score = 0.0

    # ── Summary match ─────────────────────────────────────────────────────
    # Cosine sim between career summary embedding and JD centroid
    summary_emb = summary_embs[candidate_idx]
    summary_score = float(np.clip(summary_emb @ jd_centroid, 0.0, 1.0))

    # ── Blend: skills carry more weight than summary ──────────────────────
    final = 0.60 * skill_score + 0.40 * summary_score
    return float(np.clip(final, 0.0, 1.0))
