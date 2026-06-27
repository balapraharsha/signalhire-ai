"""
online/features/behavioral.py
Feature 5 — Engagement Decay Rate (NOVEL SIGNAL)
Feature 6 — Behavioral Availability Score

Engagement decay: response_rate / log1p(platform_tenure_months)
A 0.60 response rate from someone who joined 2 months ago = excellent.
Same rate after 18 months = stagnant/going cold.

Behavioral: composite of all 23 Redrob platform signals.
"""

import numpy as np
from datetime import date
from dateutil.parser import parse as parse_date


REFERENCE_DATE = date(2026, 6, 28)  # submission deadline — "today" for ranking purposes


def days_since(date_str: str) -> int:
    """Days between date_str and reference date. Returns 9999 on parse failure."""
    try:
        d = parse_date(date_str).date()
        return max(0, (REFERENCE_DATE - d).days)
    except Exception:
        return 9999


def compute_engagement_decay(signals: dict) -> float:
    """
    Novel signal: response rate normalised by how long they've been on the platform.

    A new user with 0.60 response rate is genuinely engaged.
    A 2-year user with 0.60 response rate hasn't improved — going stale.

    Returns score in [0, 1].
    """
    resp_rate = float(signals.get("recruiter_response_rate", 0.3))

    # Platform tenure in months
    signup_str = signals.get("signup_date", "")
    if signup_str:
        tenure_days = days_since(signup_str)
        tenure_months = max(tenure_days / 30.0, 0.5)  # at least 0.5 months
    else:
        tenure_months = 12.0  # default assumption

    # log1p dampens long-tenure effect — prevents runway penalisation
    decay_score = resp_rate / np.log1p(tenure_months)

    # Normalise to [0, 1]: theoretical max is ~1.0 / log1p(0.5) ≈ 1.44
    # We clip and scale
    normalised = decay_score / 1.44
    return float(np.clip(normalised, 0.0, 1.0))


def compute_behavioral_score(signals: dict) -> float:
    """
    Composite behavioral availability score from all 23 Redrob signals.
    Returns score in [0, 1].

    Component weights are chosen to prioritise:
      - Recent activity (are they actually looking?)
      - Response rate (will they respond to us?)
      - Open to work (explicit intent)
      - GitHub (passive skill verification)
      - Interview completion (serious candidate)
    """
    # ── Recency ──────────────────────────────────────────────────────────
    last_active = signals.get("last_active_date", "")
    if last_active:
        inactive_days = days_since(last_active)
        # Exponential decay: full score at 0 days, ~0 at 180 days
        recency = float(np.exp(-inactive_days / 60.0))
    else:
        recency = 0.3  # unknown — penalise mildly

    # ── Open to work ──────────────────────────────────────────────────────
    open_to_work = 1.0 if signals.get("open_to_work_flag", False) else 0.4

    # ── Response quality ─────────────────────────────────────────────────
    resp_rate = float(signals.get("recruiter_response_rate", 0.3))

    # Fast responders get a bonus
    avg_response_h = float(signals.get("avg_response_time_hours", 48))
    # < 4h = fast, > 96h = slow
    response_speed = float(np.clip(1.0 - avg_response_h / 96.0, 0.0, 1.0))

    # ── GitHub activity ───────────────────────────────────────────────────
    github = float(signals.get("github_activity_score", -1))
    if github == -1:
        github_score = 0.3  # no GitHub linked — mild penalty
    else:
        github_score = github / 100.0

    # ── Platform assessment scores ────────────────────────────────────────
    assessment_scores = signals.get("skill_assessment_scores", {})
    if assessment_scores:
        assess_avg = float(np.mean(list(assessment_scores.values()))) / 100.0
    else:
        assess_avg = 0.3

    # ── Interview completion rate ─────────────────────────────────────────
    ivw_rate = float(signals.get("interview_completion_rate", 0.5))

    # ── Offer acceptance ─────────────────────────────────────────────────
    offer_acc = float(signals.get("offer_acceptance_rate", -1))
    if offer_acc == -1:
        offer_score = 0.5  # no history — neutral
    else:
        offer_score = offer_acc

    # ── Verification ─────────────────────────────────────────────────────
    verified = (
        (1.0 if signals.get("verified_email", False) else 0.0) +
        (1.0 if signals.get("verified_phone", False) else 0.0) +
        (0.5 if signals.get("linkedin_connected", False) else 0.0)
    ) / 2.5

    # ── Recruiter interest signals ────────────────────────────────────────
    saved_30d = min(float(signals.get("saved_by_recruiters_30d", 0)), 20) / 20.0
    views_30d = min(float(signals.get("profile_views_received_30d", 0)), 100) / 100.0

    # ── Applications activity ─────────────────────────────────────────────
    apps_30d = min(float(signals.get("applications_submitted_30d", 0)), 10) / 10.0

    # ── Composite (weighted) ──────────────────────────────────────────────
    score = (
        0.22 * recency         +   # Are they active NOW?
        0.18 * resp_rate       +   # Will they respond?
        0.10 * open_to_work    +   # Explicit intent
        0.10 * github_score    +   # Passive skill signal
        0.10 * assess_avg      +   # Platform-verified skills
        0.08 * ivw_rate        +   # Serious about interviews
        0.07 * response_speed  +   # How fast they respond
        0.06 * offer_score     +   # Doesn't ghost at offer stage
        0.04 * verified        +   # Identity trust
        0.03 * saved_30d       +   # Recruiter demand signal
        0.01 * views_30d       +   # Visibility
        0.01 * apps_30d            # Active job searching
    )
    return float(np.clip(score, 0.0, 1.0))
