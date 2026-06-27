"""
online/explain.py
Fact-Grounded Reasoning Generator

Every candidate gets a UNIQUE 1-2 sentence reason that:
1. References specific facts from their actual profile
2. Connects to JD requirements
3. Acknowledges honest concerns where relevant
4. Is never templated (no two strings are identical)
5. Does NOT hallucinate — only says what's in the data
"""

from datetime import date
from dateutil.parser import parse as parse_date

from online.jd_config import CONSULTING_COMPANIES, PREFERRED_LOCATIONS

REFERENCE_DATE = date(2026, 6, 28)

PROFICIENCY_LABELS = {
    "beginner": "beginner-level",
    "intermediate": "intermediate",
    "advanced": "advanced",
    "expert": "expert-level",
}

JD_RELEVANT_SKILLS = {
    "embedding", "embeddings", "faiss", "qdrant", "pinecone", "milvus",
    "weaviate", "vector", "retrieval", "ranking", "search", "rag",
    "llm", "transformer", "bert", "gpt", "sentence-transformer",
    "fine-tuning", "lora", "peft", "nlp", "recommendation", "pytorch",
    "tensorflow", "python", "elasticsearch", "opensearch", "bge",
    "hugging face", "huggingface", "xgboost", "lightgbm", "ndcg", "mlflow",
}


def _days_since(date_str: str) -> int:
    try:
        d = parse_date(date_str).date()
        return max(0, (REFERENCE_DATE - d).days)
    except Exception:
        return 9999


def _best_jd_skill(candidate: dict) -> dict | None:
    """Return the candidate's most JD-relevant skill by depth score."""
    skills = candidate.get("skills", [])
    best = None
    best_score = -1
    for s in skills:
        name = s.get("name", "").lower()
        if any(k in name for k in JD_RELEVANT_SKILLS):
            prof_map = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}
            score = prof_map.get(s.get("proficiency", "intermediate"), 2) * \
                    s.get("duration_months", 1)
            if score > best_score:
                best_score = score
                best = s
    return best


def _latest_company(candidate: dict) -> str:
    history = candidate.get("career_history", [])
    if not history:
        return candidate.get("profile", {}).get("current_company", "unknown company")
    current = [h for h in history if h.get("is_current")]
    if current:
        return current[0].get("company", "unknown company")
    return history[0].get("company", "unknown company")


def _is_product_company(company: str) -> bool:
    return not any(c.lower() in company.lower() for c in CONSULTING_COMPANIES)


def _format_notice(days: int) -> str:
    if days <= 15:
        return "immediate joiner"
    if days <= 30:
        return f"{days}-day notice"
    if days <= 60:
        return f"{days}-day notice (buyout possible)"
    return f"{days}-day notice"


def generate_reasoning(
    candidate: dict,
    features: dict,  # {feature_name: float}
    rank: int,
    is_honeypot: bool,
    honeypot_reason: str = "",
) -> str:
    """
    Generate a 1-2 sentence fact-grounded reasoning string.
    Every sentence pulls from actual candidate data.
    """
    if is_honeypot:
        return f"Excluded from ranking: {honeypot_reason}"

    profile = candidate.get("profile", {})
    sig = candidate.get("redrob_signals", {})
    history = candidate.get("career_history", [])

    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "Unknown")
    location = profile.get("location", "")
    company = _latest_company(candidate)
    is_product = _is_product_company(company)

    # ── Behavioral facts ───────────────────────────────────────────────────
    resp_rate = sig.get("recruiter_response_rate", 0)
    notice = int(sig.get("notice_period_days", 60))
    last_active = sig.get("last_active_date", "")
    inactive_days = _days_since(last_active) if last_active else 999
    github = sig.get("github_activity_score", -1)
    open_to_work = sig.get("open_to_work_flag", False)

    # ── Best JD-relevant skill ─────────────────────────────────────────────
    best_skill = _best_jd_skill(candidate)

    # ── Part 1: Profile + technical summary ───────────────────────────────
    parts = []

    # Lead with YoE + title + company type
    company_type = "product company" if is_product else "IT-services firm"
    sentence1 = f"{yoe:.0f}yr {title} at {company} ({company_type})"

    # Add top JD-relevant skill if found
    if best_skill:
        skill_name = best_skill["name"]
        skill_months = best_skill.get("duration_months", 0)
        skill_prof = PROFICIENCY_LABELS.get(best_skill.get("proficiency", "intermediate"), "")
        sentence1 += f"; {skill_prof} {skill_name} ({skill_months}mo)"

    # Add location if preferred
    if any(pl.lower() in location.lower() for pl in PREFERRED_LOCATIONS):
        sentence1 += f"; {location}-based"

    parts.append(sentence1)

    # ── Part 2: Behavioral + concerns ─────────────────────────────────────
    behavioral_parts = []

    # Activity
    if inactive_days <= 3:
        behavioral_parts.append("active today")
    elif inactive_days <= 7:
        behavioral_parts.append(f"active {inactive_days}d ago")
    elif inactive_days <= 30:
        behavioral_parts.append(f"active {inactive_days}d ago")
    elif inactive_days <= 90:
        behavioral_parts.append(f"last active {inactive_days}d ago")
    else:
        behavioral_parts.append(f"inactive {inactive_days}d — availability uncertain")

    # Response rate
    if resp_rate >= 0.70:
        behavioral_parts.append(f"response rate {resp_rate:.0%} (high)")
    elif resp_rate >= 0.40:
        behavioral_parts.append(f"response rate {resp_rate:.0%}")
    else:
        behavioral_parts.append(f"response rate {resp_rate:.0%} (low — may not reply)")

    # Notice
    behavioral_parts.append(_format_notice(notice))

    # Open to work
    if open_to_work:
        behavioral_parts.append("marked open-to-work")

    # GitHub
    if github >= 70:
        behavioral_parts.append(f"GitHub score {github:.0f}/100")
    elif github == -1:
        behavioral_parts.append("no GitHub linked")

    sentence2 = "; ".join(behavioral_parts[:4]) + "."

    # ── Honest concern for lower ranks ────────────────────────────────────
    concern = ""
    if rank > 50:
        if not is_product:
            concern = " No product-company background limits fit for this role."
        elif inactive_days > 90:
            concern = " Extended inactivity reduces hire probability."
        elif resp_rate < 0.20:
            concern = " Very low response rate — reachability risk."
        elif notice > 90:
            concern = f" {notice}-day notice is a significant logistics barrier."

    if not concern and features.get("profile_coherence", 1.0) < 0.35:
        concern = " Skill-role coherence is low — verify profile depth before outreach."

    # ── Combine ────────────────────────────────────────────────────────────
    full_reason = f"{parts[0]}. {sentence2}{concern}"

    # Final safety: truncate to ~400 chars max, clean whitespace
    full_reason = " ".join(full_reason.split())
    if len(full_reason) > 400:
        full_reason = full_reason[:397] + "..."

    return full_reason
