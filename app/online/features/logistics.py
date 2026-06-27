"""
online/features/logistics.py
Feature 8 — Logistics Fit Score

Notice period, location, salary band, work mode, relocation.
Lower weight than technical signals but still matters per the JD.
"""

import numpy as np
from online.jd_config import (
    PREFERRED_LOCATIONS,
    IDEAL_NOTICE_DAYS,
    MAX_NOTICE_DAYS,
    SALARY_BAND_MIN_LPA,
    SALARY_BAND_MAX_LPA,
)


def compute_logistics_fit(candidate: dict) -> float:
    """
    Returns logistics fit score in [0, 1].
    """
    sig = candidate.get("redrob_signals", {})
    profile = candidate.get("profile", {})

    # ── Notice period ─────────────────────────────────────────────────────
    notice = float(sig.get("notice_period_days", 60))
    if notice <= IDEAL_NOTICE_DAYS:
        notice_score = 1.0
    elif notice <= 60:
        # JD: "can buy out up to 30 days; 30+ still in scope but bar higher"
        notice_score = 0.75
    elif notice <= MAX_NOTICE_DAYS:
        notice_score = 0.55
    else:
        notice_score = 0.30  # > 90 days is a real problem for Series A

    # ── Location ──────────────────────────────────────────────────────────
    location = profile.get("location", "")
    country  = profile.get("country", "India")
    relocate = bool(sig.get("willing_to_relocate", False))

    location_str = f"{location} {country}".lower()
    in_preferred = any(pl.lower() in location_str for pl in PREFERRED_LOCATIONS)

    if in_preferred:
        location_score = 1.0
    elif relocate:
        location_score = 0.75  # willing to move
    elif country.lower() in ("india", "in"):
        location_score = 0.50  # India but wrong city
    else:
        location_score = 0.20  # Outside India — case-by-case per JD

    # ── Salary band overlap ───────────────────────────────────────────────
    salary_range = sig.get("expected_salary_range_inr_lpa", {})
    cand_min = float(salary_range.get("min", 0))
    cand_max = float(salary_range.get("max", 999))

    # Check overlap with JD salary band
    overlap = min(cand_max, SALARY_BAND_MAX_LPA) - max(cand_min, SALARY_BAND_MIN_LPA)
    band_width = SALARY_BAND_MAX_LPA - SALARY_BAND_MIN_LPA

    if overlap <= 0:
        salary_score = 0.20  # no overlap — potential misalignment
    else:
        salary_score = min(overlap / band_width, 1.0)

    # ── Work mode preference ───────────────────────────────────────────────
    # JD: "Hybrid — flexible cadence"
    work_mode = sig.get("preferred_work_mode", "hybrid")
    if work_mode in ("hybrid", "flexible", "onsite"):
        work_mode_score = 1.0
    elif work_mode == "remote":
        work_mode_score = 0.60  # JD has offices, remote-only is tension

    # ── Composite ─────────────────────────────────────────────────────────
    score = (
        0.40 * notice_score    +   # JD emphasises notice period explicitly
        0.35 * location_score  +   # Location mentioned prominently
        0.15 * salary_score    +   # Salary needs overlap
        0.10 * work_mode_score     # Work mode minor factor
    )
    return float(np.clip(score, 0.0, 1.0))
