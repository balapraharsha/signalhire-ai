"""
online/features/depth.py
Feature 4 — Expertise Depth Score

Goes beyond skill presence to measure HOW DEEPLY a candidate
knows their key skills, using platform-verified data.

Depth = proficiency × months_of_experience × endorsement_boost
"""

import numpy as np
from online.jd_config import HARD_REQUIRED_SKILLS


PROFICIENCY_MAP = {
    "beginner":     0.25,
    "intermediate": 0.55,
    "advanced":     0.80,
    "expert":       1.00,
}

# JD hard-required skill families for direct matching
JD_SKILL_FAMILIES = {
    "embeddings":    ["embedding", "sentence-transformer", "bge", "e5", "ada", "openai embedding"],
    "vector_db":     ["faiss", "pinecone", "qdrant", "weaviate", "milvus", "opensearch",
                      "elasticsearch", "chroma", "vector search", "hybrid search"],
    "ranking":       ["ranking", "retrieval", "search", "ndcg", "mrr", "ltr",
                      "learning to rank", "recommendation", "recommender"],
    "llm":           ["llm", "gpt", "claude", "gemini", "transformer", "bert", "rag",
                      "fine-tuning", "fine tuning", "lora", "peft", "qlora"],
    "python_ml":     ["python", "pytorch", "tensorflow", "scikit-learn", "sklearn",
                      "numpy", "pandas", "hugging face", "huggingface"],
    "nlp":           ["nlp", "natural language", "text classification", "named entity",
                      "sentiment", "tokenization", "spacy", "nltk"],
    "infra":         ["spark", "kafka", "airflow", "mlflow", "kubeflow", "docker",
                      "kubernetes", "aws", "gcp", "azure"],
}


def skill_matches_jd(skill_name: str) -> bool:
    """True if a skill name matches any JD-required skill family."""
    name_lower = skill_name.lower()
    for keywords in JD_SKILL_FAMILIES.values():
        if any(kw in name_lower for kw in keywords):
            return True
    return False


def compute_expertise_depth(candidate: dict) -> float:
    """
    Returns expertise depth score in [0, 1].

    Specifically measures depth in JD-relevant skills only.
    A candidate with 60 months of Python + expert proficiency + 50 endorsements
    scores much higher than one who lists Python with 2 months and beginner level.
    """
    skills = candidate.get("skills", [])
    if not skills:
        return 0.0

    sig = candidate.get("redrob_signals", {})
    assessment_scores = sig.get("skill_assessment_scores", {})

    relevant_scores = []

    for skill in skills:
        name = skill.get("name", "")
        if not skill_matches_jd(name):
            continue

        prof  = PROFICIENCY_MAP.get(skill.get("proficiency", "intermediate"), 0.55)
        months = min(skill.get("duration_months", 6), 72)  # cap at 72 months (6 yrs)
        dur_score = months / 72.0

        # Endorsement boost (soft signal, capped)
        endorsements = min(skill.get("endorsements", 0), 50)
        endorse_boost = 1.0 + 0.1 * (endorsements / 50.0)  # max 1.1×

        # Platform assessment score (strongest verification signal)
        assess_key = None
        for k in assessment_scores:
            if k.lower() in name.lower() or name.lower() in k.lower():
                assess_key = k
                break

        if assess_key:
            # Assessment verified — blend with profile data
            assess_norm = assessment_scores[assess_key] / 100.0
            depth = (0.35 * prof + 0.30 * dur_score + 0.35 * assess_norm) * endorse_boost
        else:
            # No assessment — rely on profile claims
            depth = (0.50 * prof + 0.50 * dur_score) * endorse_boost

        relevant_scores.append(min(depth, 1.0))

    if not relevant_scores:
        return 0.0

    # Mean of top-5 relevant skill depths
    relevant_scores.sort(reverse=True)
    top = relevant_scores[:5]
    return float(np.clip(np.mean(top), 0.0, 1.0))
