"""
tests/validate_sample.py
Test the full pipeline on the 50 sample candidates.

Usage:
    python tests/validate_sample.py

This runs the complete online pipeline on sample_candidates.json
and prints a detailed analysis of rankings, feature scores,
honeypot detection, and reasoning quality.
"""

import json
import os
import sys
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SAMPLE_PATH = os.path.join(ROOT, "data/sample_candidates.json")

from online.features.depth       import compute_expertise_depth
from online.features.behavioral  import compute_engagement_decay, compute_behavioral_score
from online.features.antipattern import compute_antipattern_penalty
from online.features.logistics   import compute_logistics_fit
from online.fusion               import compute_composite_scores, N_FEATURES
from online.explain              import generate_reasoning


def load_sample_candidates():
    with open(SAMPLE_PATH) as f:
        return json.load(f)


def run_non_embedding_features(candidates):
    """
    Run all features that don't need embeddings.
    Returns feature matrix (N, 5) for depth, decay, behavioral, logistics, anti-pattern.
    """
    N = len(candidates)
    results = []

    for i, c in enumerate(candidates):
        sig = c.get("redrob_signals", {})
        dep = compute_expertise_depth(c)
        dec = compute_engagement_decay(sig)
        beh = compute_behavioral_score(sig)
        log = compute_logistics_fit(c)
        # Use neutral values for embedding-based features (semantic=0.5, coherence=0.5, consistency=0.6)
        mult, hp, hp_reason = compute_antipattern_penalty(c, coherence_score=0.5, semantic_score=0.5)

        results.append({
            "candidate_id": c["candidate_id"],
            "title": c["profile"]["current_title"],
            "yoe": c["profile"]["years_of_experience"],
            "company": c.get("career_history", [{}])[0].get("company", "?"),
            "location": c["profile"]["location"],
            "expertise_depth": dep,
            "engagement_decay": dec,
            "behavioral_score": beh,
            "logistics_fit": log,
            "antipattern_multiplier": mult,
            "is_honeypot": hp,
            "honeypot_reason": hp_reason,
            "resp_rate": sig.get("recruiter_response_rate", 0),
            "notice": sig.get("notice_period_days", 60),
            "last_active": sig.get("last_active_date", ""),
            "open_to_work": sig.get("open_to_work_flag", False),
        })
    return results


def print_feature_table(results, sort_by="expertise_depth"):
    """Print a readable feature table sorted by a column."""
    results_sorted = sorted(results, key=lambda x: x[sort_by], reverse=True)

    print(f"\n{'─'*120}")
    print(f"{'#':>3} {'Candidate':12} {'Title':30} {'YoE':>5} {'Depth':>6} {'Decay':>6} {'Behav':>6} {'Logist':>7} {'Mult':>5} {'HP':>4}")
    print(f"{'─'*120}")
    for i, r in enumerate(results_sorted[:25], 1):
        hp_mark = "❌" if r["is_honeypot"] else "  "
        print(
            f"{i:>3} {r['candidate_id']:12} {r['title'][:30]:30} "
            f"{r['yoe']:>5.1f} {r['expertise_depth']:>6.3f} {r['engagement_decay']:>6.3f} "
            f"{r['behavioral_score']:>6.3f} {r['logistics_fit']:>7.3f} "
            f"{r['antipattern_multiplier']:>5.2f} {hp_mark:>4}"
        )


def print_honeypots(results):
    honeypots = [r for r in results if r["is_honeypot"]]
    print(f"\n{'─'*80}")
    print(f"HONEYPOTS DETECTED: {len(honeypots)}")
    print(f"{'─'*80}")
    for h in honeypots:
        print(f"  {h['candidate_id']} | {h['title']} | {h['honeypot_reason']}")
    if not honeypots:
        print("  None detected (expected in sample — honeypots are subtle)")


def print_reasoning_samples(candidates, results, top_n=5):
    """Print reasoning for the top-N candidates by expertise_depth."""
    sorted_results = sorted(results, key=lambda x: x["expertise_depth"], reverse=True)
    top_ids = [r["candidate_id"] for r in sorted_results[:top_n]]
    cand_map = {c["candidate_id"]: c for c in candidates}

    print(f"\n{'─'*80}")
    print(f"SAMPLE REASONING STRINGS (top {top_n} by expertise_depth)")
    print(f"{'─'*80}")
    for rank, cid in enumerate(top_ids, 1):
        c = cand_map[cid]
        r = next(x for x in results if x["candidate_id"] == cid)
        feat_dict = {
            "profile_coherence": 0.5,
            "semantic_match": 0.5,
        }
        reasoning = generate_reasoning(
            c, feat_dict, rank=rank,
            is_honeypot=r["is_honeypot"],
            honeypot_reason=r["honeypot_reason"]
        )
        print(f"\n  Rank {rank}: {cid} ({r['title']}, {r['yoe']:.1f}yr)")
        print(f"  → {reasoning}")


def check_reasoning_quality(candidates, results):
    """Run all Stage 4 reasoning quality checks."""
    print(f"\n{'─'*80}")
    print("STAGE 4 REASONING QUALITY CHECKS")
    print(f"{'─'*80}")

    cand_map = {c["candidate_id"]: c for c in candidates}
    reasonings = []
    for rank, r in enumerate(results[:20], 1):
        c = cand_map[r["candidate_id"]]
        reasoning = generate_reasoning(
            c, {}, rank=rank,
            is_honeypot=r["is_honeypot"],
            honeypot_reason=r["honeypot_reason"]
        )
        reasonings.append(reasoning)

    checks = {
        "Non-empty": all(len(r) > 20 for r in reasonings),
        "Non-identical": len(set(reasonings)) == len(reasonings),
        "No 'template'": not any("template" in r.lower() for r in reasonings),
        "Contains facts": all(
            any(char.isdigit() for char in r) for r in reasonings
        ),
        "Reasonable length": all(20 < len(r) < 500 for r in reasonings),
    }
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")


def main():
    print("\n" + "="*60)
    print("  SignalHire AI — Sample Validation Suite")
    print("="*60)

    t0 = time.time()

    # Load samples
    print("\nLoading sample candidates...")
    candidates = load_sample_candidates()
    print(f"  {len(candidates)} candidates loaded")

    # Run non-embedding features
    print("\nComputing non-embedding features...")
    results = run_non_embedding_features(candidates)
    print(f"  Done in {time.time()-t0:.2f}s")

    # Feature distribution summary
    print(f"\n{'─'*60}")
    print("FEATURE DISTRIBUTION SUMMARY")
    print(f"{'─'*60}")
    for feat in ["expertise_depth", "engagement_decay", "behavioral_score", "logistics_fit"]:
        vals = [r[feat] for r in results]
        arr = np.array(vals)
        print(f"  {feat:22} min={arr.min():.3f}  mean={arr.mean():.3f}  max={arr.max():.3f}  std={arr.std():.3f}")

    # Anti-pattern stats
    soft_penalised = sum(1 for r in results if 0 < r["antipattern_multiplier"] < 1.0)
    honeypot_count = sum(1 for r in results if r["is_honeypot"])
    print(f"\n  Soft penalties applied: {soft_penalised}/{len(results)}")
    print(f"  Honeypots detected:     {honeypot_count}/{len(results)}")

    # Table
    print_feature_table(results, sort_by="expertise_depth")
    print_honeypots(results)
    print_reasoning_samples(candidates, results, top_n=5)
    check_reasoning_quality(candidates, results)

    print(f"\n{'='*60}")
    print(f"  Validation complete in {time.time()-t0:.2f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
