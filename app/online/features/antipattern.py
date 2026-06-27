"""
online/features/antipattern.py
Feature 7 — Anti-Pattern Detection + Honeypot Disqualification

Returns a multiplier in [0.0, 1.0]:
  1.0 = no issues detected
  0.0 = hard disqualified (honeypot or clear JD violation)

The JD explicitly lists disqualifiers. We enforce them.
"""

import re
from datetime import date
from dateutil.parser import parse as parse_date

from online.jd_config import (
    CONSULTING_COMPANIES,
    DISQUALIFIER_TITLES,
    WRONG_DOMAIN_SKILLS,
    RESEARCH_ONLY_SIGNALS,
)

REFERENCE_DATE = date(2026, 6, 28)

PROFICIENCY_MAP = {
    "beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4
}


# ── HONEYPOT DETECTION ───────────────────────────────────────────────────────

def _check_timeline_impossibility(candidate: dict) -> tuple[bool, str]:
    """
    Detect: tenure claimed at company exceeds company's plausible founding.
    We use a heuristic: if total claimed months at a single company exceeds
    the candidate's stated years of experience × 12 by > 12 months, flag it.

    Also detect overlapping date ranges in career history.
    """
    history = candidate.get("career_history", [])
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)

    for h in history:
        dur = h.get("duration_months", 0)
        # If a single job tenure exceeds total YoE + 6 months — impossible
        if dur > (yoe * 12) + 6 and yoe > 0:
            return True, f"Impossible tenure: {dur}mo at {h.get('company','?')} but only {yoe}yr total experience"

    # Check for overlapping date ranges
    dated = []
    for h in history:
        try:
            s = parse_date(h["start_date"]).date() if h.get("start_date") else None
            e = parse_date(h["end_date"]).date() if h.get("end_date") else REFERENCE_DATE
            if s:
                dated.append((s, e))
        except Exception:
            pass

    dated.sort()
    for i in range(len(dated) - 1):
        _, e1 = dated[i]
        s2, _ = dated[i + 1]
        overlap = (e1 - s2).days
        if overlap > 180:  # more than 6 months overlap = suspicious
            return True, f"Overlapping career dates: {overlap} day overlap detected"

    return False, ""


def _check_skill_inflation(candidate: dict) -> tuple[bool, str]:
    """
    Detect: expert proficiency on many skills with near-zero use time.
    Pattern: someone who claims to be expert in 8+ skills but used none of them.
    """
    skills = candidate.get("skills", [])
    expert_count = 0
    zero_use_expert = 0

    for s in skills:
        prof = s.get("proficiency", "")
        months = s.get("duration_months", 0)
        if prof == "expert":
            expert_count += 1
            if months <= 3:
                zero_use_expert += 1

    # Expert in 8+ skills with no usage on 6+ of them = suspicious
    if expert_count >= 8 and zero_use_expert >= 6:
        return True, f"Skill inflation: {expert_count} expert skills, {zero_use_expert} with ≤3 months use"

    return False, ""


def _check_coherence_keyword_stuffing(
    candidate: dict,
    coherence_score: float,
    semantic_score: float,
) -> tuple[bool, str]:
    """
    Detect: high keyword match but incoherent profile.
    This is the sneaky honeypot: AI keywords on a non-AI profile.
    """
    if coherence_score < 0.15 and semantic_score > 0.70:
        title = candidate.get("profile", {}).get("current_title", "")
        return True, f"Keyword stuffing: coherence={coherence_score:.2f} but semantic={semantic_score:.2f} on '{title}'"
    return False, ""


# ── SOFT DISQUALIFIERS (reduce score, don't eliminate) ───────────────────────

def _check_title_mismatch(candidate: dict) -> float:
    """
    Returns a multiplier penalty for clearly wrong-domain titles.
    0.2 = hard disqualifier title, 0.6 = soft mismatch, 1.0 = no issue.
    """
    title = candidate.get("profile", {}).get("current_title", "").lower()

    # Hard disqualifier titles from JD
    hard_disqualifiers = [
        "marketing manager", "marketing executive", "marketing analyst",
        "hr manager", "hr executive", "human resources manager",
        "accountant", "financial analyst", "finance manager",
        "content writer", "content creator",
        "graphic designer", "civil engineer", "mechanical engineer",
        "customer support", "customer success",
    ]
    for d in hard_disqualifiers:
        if d in title:
            return 0.20  # Very heavy penalty but not full elimination
            # (they might have AI skills we still want to partially credit)

    # Soft mismatch — adjacent but not ideal
    soft_mismatch = [
        "business analyst", "project manager", "operations manager",
        "sales executive", "product manager",  # PM is borderline
    ]
    for s in soft_mismatch:
        if s in title:
            return 0.55

    return 1.0


def _check_wrong_domain_skills(candidate: dict) -> float:
    """
    Returns multiplier: penalise if ALL listed skills are wrong-domain.
    """
    skills = candidate.get("skills", [])
    if not skills:
        return 0.5

    wrong_count = 0
    for s in skills:
        name = s.get("name", "").lower()
        if any(w.lower() in name for w in WRONG_DOMAIN_SKILLS):
            wrong_count += 1

    # If > 80% of skills are wrong domain, heavy penalty
    ratio = wrong_count / len(skills)
    if ratio > 0.8:
        return 0.25
    if ratio > 0.5:
        return 0.60
    return 1.0


def _check_research_only(candidate: dict) -> float:
    """
    JD: "pure research environments without production deployment — will not move forward."
    """
    summary = candidate.get("profile", {}).get("summary", "").lower()
    history = candidate.get("career_history", [])

    research_signals = sum(
        1 for sig in RESEARCH_ONLY_SIGNALS
        if sig.lower() in summary
    )

    # Check if all history is research/academic
    has_production = False
    for h in history:
        desc = h.get("description", "").lower()
        title = h.get("title", "").lower()
        if any(w in desc + title for w in ["production", "deployed", "shipped", "users", "customers"]):
            has_production = True
            break

    if research_signals >= 2 and not has_production:
        return 0.30

    return 1.0


def _check_experience_range(candidate: dict) -> float:
    """
    JD: 5-9 years preferred. < 3 years is very unlikely. > 15 is title-chaser risk.
    """
    from online.jd_config import IDEAL_YOE_MIN, IDEAL_YOE_MAX, ABSOLUTE_MIN_YOE
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)

    if yoe < ABSOLUTE_MIN_YOE:
        return 0.50
    if yoe < IDEAL_YOE_MIN:
        return 0.80  # slight penalty, 3-5 years is borderline
    if yoe <= IDEAL_YOE_MAX:
        return 1.0   # sweet spot
    if yoe <= 12:
        return 0.90  # slightly over but still fine
    return 0.75      # 12+ years — possible title-chaser risk


# ── PUBLIC INTERFACE ─────────────────────────────────────────────────────────

def compute_antipattern_penalty(
    candidate: dict,
    coherence_score: float = 0.5,
    semantic_score: float = 0.5,
) -> tuple[float, bool, str]:
    """
    Returns:
      (multiplier, is_honeypot, reason)

    multiplier: 0.0 = disqualified, 1.0 = clean
    is_honeypot: True if candidate should be forced to tier 0
    reason: human-readable explanation of any penalty
    """
    reasons = []

    # ── HARD HONEYPOT CHECKS (force tier 0) ──────────────────────────────
    timeline_bad, tl_reason = _check_timeline_impossibility(candidate)
    if timeline_bad:
        return 0.0, True, f"HONEYPOT: {tl_reason}"

    inflation_bad, inf_reason = _check_skill_inflation(candidate)
    if inflation_bad:
        return 0.0, True, f"HONEYPOT: {inf_reason}"

    stuffing_bad, st_reason = _check_coherence_keyword_stuffing(
        candidate, coherence_score, semantic_score
    )
    if stuffing_bad:
        return 0.0, True, f"HONEYPOT: {st_reason}"

    # ── SOFT PENALTIES (multiply together) ───────────────────────────────
    multiplier = 1.0
    multiplier *= _check_title_mismatch(candidate)
    multiplier *= _check_wrong_domain_skills(candidate)
    multiplier *= _check_research_only(candidate)
    multiplier *= _check_experience_range(candidate)

    if multiplier < 1.0:
        reasons.append(f"soft_penalty={multiplier:.2f}")

    reason_str = "; ".join(reasons) if reasons else ""
    return float(multiplier), False, reason_str
