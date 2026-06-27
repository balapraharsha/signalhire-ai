"""
online/features/consistency.py
Feature 3 — Career Consistency Score

Measures whether a candidate's career trajectory is logical
and progressive — rewarding specialisation, penalising random pivots.

Data Analyst → ML Engineer → Senior ML Engineer = high consistency
Marketing Exec → Blockchain Dev → AI Architect in 3 years = low consistency
"""

import numpy as np


def compute_career_consistency(
    candidate_idx: int,
    htitle_embs_flat: np.ndarray,
    htitle_offsets: np.ndarray,
    candidate: dict,
) -> float:
    """
    Returns career consistency score in [0, 1].

    Method:
      - Get embeddings for all career titles (chronological order)
      - Compute cosine similarity between consecutive titles
      - Average the sequential similarities
      - Bonus for product company experience
    """
    start = int(htitle_offsets[candidate_idx, 0])
    end   = int(htitle_offsets[candidate_idx, 1])

    if end - start < 2:
        # Single job or no history → neutral (can't assess trajectory)
        base = 0.60
    else:
        embs = htitle_embs_flat[start:end]  # (t, 384) — chronological

        # Pairwise cosine similarity between consecutive titles
        # dot product since embeddings are L2-normalised
        a = embs[:-1]  # (t-1, 384)
        b = embs[1:]   # (t-1, 384)
        sims = np.einsum("ij,ij->i", a, b)  # element-wise dot, (t-1,)
        sims = np.clip(sims, 0.0, 1.0)

        # Weight more recent transitions more heavily
        n = len(sims)
        weights = np.linspace(0.6, 1.0, n)  # older transitions count less
        base = float(np.average(sims, weights=weights))

    # ── Product company bonus ─────────────────────────────────────────────
    # JD strongly prefers product-company backgrounds
    from online.jd_config import CONSULTING_COMPANIES, PRODUCT_COMPANY_SIGNALS

    history = candidate.get("career_history", [])
    has_product = False
    all_consulting = True

    for h in history:
        company = h.get("company", "")
        industry = h.get("industry", "")

        is_consulting = any(c.lower() in company.lower() for c in CONSULTING_COMPANIES)
        is_product = (
            not is_consulting
            and any(p.lower() in company.lower() for p in PRODUCT_COMPANY_SIGNALS)
        ) or industry in ("Internet", "Software", "SaaS", "E-commerce", "FinTech",
                          "EdTech", "HealthTech", "Gaming", "AI/ML")

        if is_product:
            has_product = True
        if not is_consulting:
            all_consulting = False

    # Penalise pure consulting backgrounds (explicit JD disqualifier)
    if all_consulting and len(history) > 0:
        base *= 0.50

    # Reward product company experience
    elif has_product:
        base = min(base * 1.15, 1.0)

    return float(np.clip(base, 0.0, 1.0))
