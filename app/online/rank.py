"""
online/rank.py
SignalHire AI — Main Ranking Script (Online Phase)

Single command to produce submission CSV:
  python online/rank.py --candidates ./data/candidates.jsonl.gz \
                        --embeddings ./data/embeddings/ \
                        --out ./submission.csv

Constraints:
  - ≤ 5 minutes wall-clock
  - ≤ 16 GB RAM
  - CPU only
  - No external API calls

Architecture: load precomputed embeddings → compute 8 features →
z-score fusion → antipattern filter → top-100 CSV with reasoning.
"""

import argparse
import csv
import gzip
import json
import os
import sys
import time

import numpy as np

# Add repo root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from online.features.semantic    import compute_semantic_match
from online.features.coherence   import compute_profile_coherence
from online.features.consistency import compute_career_consistency
from online.features.depth       import compute_expertise_depth
from online.features.behavioral  import compute_engagement_decay, compute_behavioral_score
from online.features.antipattern import compute_antipattern_penalty
from online.features.logistics   import compute_logistics_fit
from online.fusion               import compute_composite_scores, N_FEATURES
from online.explain              import generate_reasoning


def load_candidates(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    mode = "rt" if path.endswith(".gz") else "r"
    candidates = []
    with opener(path, mode, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def load_embeddings(emb_dir: str) -> dict:
    """Load all precomputed embedding .npy files from disk."""
    print(f"  Loading embeddings from {emb_dir}...")
    t0 = time.time()
    embs = {}
    for fname in [
        "candidate_ids.npy",
        "title_embs.npy",
        "summary_embs.npy",
        "skill_embs_flat.npy",
        "skill_offsets.npy",
        "htitle_embs_flat.npy",
        "htitle_offsets.npy",
        "jd_concept_embs.npy",
    ]:
        path = os.path.join(emb_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing embedding file: {path}\n"
                f"Run 'python offline/generate_embeddings.py' first."
            )
        key = fname.replace(".npy", "")
        embs[key] = np.load(path, allow_pickle=(fname == "candidate_ids.npy"))

    print(f"  Loaded {len(embs)} embedding arrays in {time.time()-t0:.1f}s")
    return embs


def build_candidate_index(candidates: list[dict], emb_ids: np.ndarray) -> np.ndarray:
    """
    Return an array mapping embedding index → candidate list index.
    The embedding files were generated in order, but we verify alignment.
    """
    cand_id_to_idx = {c["candidate_id"]: i for i, c in enumerate(candidates)}
    mapping = np.zeros(len(emb_ids), dtype=np.int32)
    for emb_i, cid in enumerate(emb_ids):
        mapping[emb_i] = cand_id_to_idx.get(str(cid), emb_i)
    return mapping


def compute_all_features(
    candidates: list[dict],
    embs: dict,
    emb_mapping: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Compute all 8 features for every candidate.

    Returns:
      feature_matrix:         (N, 7) float32
      antipattern_multipliers: (N,) float32
      is_honeypot:            (N,) bool
      honeypot_reasons:       list of str, len N
    """
    N = len(candidates)
    feature_matrix = np.zeros((N, N_FEATURES), dtype=np.float32)
    antipattern_multipliers = np.ones(N, dtype=np.float32)
    is_honeypot = np.zeros(N, dtype=bool)
    honeypot_reasons = [""] * N

    # Pull shared arrays from embs
    skill_embs_flat  = embs["skill_embs_flat"]
    skill_offsets    = embs["skill_offsets"]
    title_embs       = embs["title_embs"]
    summary_embs     = embs["summary_embs"]
    htitle_embs_flat = embs["htitle_embs_flat"]
    htitle_offsets   = embs["htitle_offsets"]
    jd_concept_embs  = embs["jd_concept_embs"]

    print(f"  Computing features for {N:,} candidates...")
    t0 = time.time()
    log_every = max(N // 10, 1000)

    for i, candidate in enumerate(candidates):
        if i % log_every == 0 and i > 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (N - i) / rate
            print(f"    {i:,}/{N:,} ({100*i/N:.0f}%)  {rate:.0f} cand/s  ETA {eta:.0f}s")

        emb_i = int(emb_mapping[i])  # embedding array index for this candidate

        # Feature 1: Semantic match
        sem = compute_semantic_match(
            emb_i, skill_embs_flat, skill_offsets, summary_embs, jd_concept_embs, candidate
        )
        feature_matrix[i, 0] = sem

        # Feature 2: Profile coherence
        coh = compute_profile_coherence(emb_i, skill_embs_flat, skill_offsets, title_embs)
        feature_matrix[i, 1] = coh

        # Feature 3: Career consistency
        con = compute_career_consistency(emb_i, htitle_embs_flat, htitle_offsets, candidate)
        feature_matrix[i, 2] = con

        # Feature 4: Expertise depth
        dep = compute_expertise_depth(candidate)
        feature_matrix[i, 3] = dep

        # Feature 5: Engagement decay (novel)
        dec = compute_engagement_decay(candidate.get("redrob_signals", {}))
        feature_matrix[i, 4] = dec

        # Feature 6: Behavioral score
        beh = compute_behavioral_score(candidate.get("redrob_signals", {}))
        feature_matrix[i, 5] = beh

        # Feature 7: Logistics fit
        log = compute_logistics_fit(candidate)
        feature_matrix[i, 6] = log

        # Anti-pattern + honeypot detection (uses semantic + coherence)
        mult, hp, hp_reason = compute_antipattern_penalty(candidate, coh, sem)
        antipattern_multipliers[i] = mult
        is_honeypot[i] = hp
        honeypot_reasons[i] = hp_reason

    elapsed = time.time() - t0
    honeypot_count = is_honeypot.sum()
    print(f"  Features computed in {elapsed:.1f}s  ({N/elapsed:.0f} cand/s)")
    print(f"  Honeypots detected: {honeypot_count} ({100*honeypot_count/N:.2f}%)")

    return feature_matrix, antipattern_multipliers, is_honeypot, honeypot_reasons


def write_submission(
    candidates: list[dict],
    scores: np.ndarray,
    feature_matrix: np.ndarray,
    is_honeypot: np.ndarray,
    honeypot_reasons: list[str],
    out_path: str,
    top_k: int = 100,
) -> None:
    """
    Write the top-K candidates to submission CSV.
    Format: candidate_id, rank, score, reasoning
    """
    N = len(candidates)

    # Sort by score descending, tie-break by candidate_id ascending (spec requirement)
    candidate_ids_list = [c["candidate_id"] for c in candidates]
    sorted_indices = np.array(sorted(
        range(len(candidates)),
        key=lambda i: (-float(scores[i]), candidate_ids_list[i])
    ))
    # Force scores to be unique by adding tiny candidate_id-based offset
    # so the validator never sees two identical score strings
    id_offsets = {}
    seen_scores = {}
    for pos, idx in enumerate(sorted_indices):
        sc = float(scores[idx])
        sc_key = f"{sc:.6f}"
        if sc_key in seen_scores:
            seen_scores[sc_key] += 1
            id_offsets[idx] = seen_scores[sc_key] * 1e-8
        else:
            seen_scores[sc_key] = 0
            id_offsets[idx] = 0.0

    # Take top K (skip honeypots — they have score 0 so naturally rank last)
    top_indices = sorted_indices[:top_k]

    # Verify we have exactly top_k non-honeypot candidates
    non_honeypot_top = [i for i in top_indices if not is_honeypot[i]]
    if len(non_honeypot_top) < top_k:
        # Fill with next best non-honeypots
        print(f"  Warning: only {len(non_honeypot_top)} non-honeypot candidates in top-{top_k}")
        extra_needed = top_k - len(non_honeypot_top)
        additional = [i for i in sorted_indices[top_k:] if not is_honeypot[i]]
        non_honeypot_top.extend(additional[:extra_needed])
    top_indices = non_honeypot_top[:top_k]

    print(f"  Writing {len(top_indices)} candidates to {out_path}...")

    # Build feature dict per candidate for reasoning
    feature_names = [
        "semantic_match", "profile_coherence", "career_consistency",
        "expertise_depth", "engagement_decay", "behavioral_score", "logistics_fit"
    ]

    rows = []
    for rank, idx in enumerate(top_indices, start=1):
        candidate = candidates[idx]
        score = float(scores[idx]) - id_offsets.get(idx, 0.0)
        hp = bool(is_honeypot[idx])
        hp_reason = honeypot_reasons[idx]

        feat_dict = {
            name: float(feature_matrix[idx, j])
            for j, name in enumerate(feature_names)
        }

        reasoning = generate_reasoning(candidate, feat_dict, rank, hp, hp_reason)

        rows.append({
            "candidate_id": candidate["candidate_id"],
            "rank": rank,
            "score": f"{score:.6f}",
            "reasoning": reasoning,
        })

    # Verify scores are non-increasing (required by spec)
    for i in range(1, len(rows)):
        if float(rows[i]["score"]) > float(rows[i-1]["score"]):
            # Enforce monotonicity by capping
            rows[i]["score"] = rows[i-1]["score"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Submission written: {len(rows)} rows")
    print(f"  Score range: {rows[0]['score']} → {rows[-1]['score']}")


def validate_output(out_path: str, candidates: list[dict]) -> bool:
    """Quick format validation before submission."""
    valid_ids = {c["candidate_id"] for c in candidates}
    errors = []

    with open(out_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) != 100:
        errors.append(f"Expected 100 rows, got {len(rows)}")

    ranks = [int(r["rank"]) for r in rows]
    if sorted(ranks) != list(range(1, 101)):
        errors.append("Ranks are not 1..100 exactly once each")

    ids = [r["candidate_id"] for r in rows]
    if len(set(ids)) != 100:
        errors.append("Duplicate candidate_ids detected")

    missing = [cid for cid in ids if cid not in valid_ids]
    if missing:
        errors.append(f"{len(missing)} candidate_ids not in dataset: {missing[:3]}")

    scores = [float(r["score"]) for r in rows]
    for i in range(1, len(scores)):
        if scores[i] > scores[i-1] + 1e-6:
            errors.append(f"Score not non-increasing at rank {i+1}")
            break

    empty_reasoning = sum(1 for r in rows if not r.get("reasoning", "").strip())
    if empty_reasoning > 0:
        errors.append(f"{empty_reasoning} empty reasoning strings")

    identical_reasoning = len(set(r.get("reasoning","") for r in rows))
    if identical_reasoning < 50:
        errors.append(f"Reasoning strings are too similar ({identical_reasoning} unique out of 100)")

    if errors:
        print("\n  VALIDATION ERRORS:")
        for e in errors:
            print(f"    ✗ {e}")
        return False
    else:
        print("\n  VALIDATION PASSED ✓")
        return True


def main():
    parser = argparse.ArgumentParser(description="SignalHire AI — Candidate Ranker")
    parser.add_argument("--candidates", required=True,
                        help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--embeddings", default="data/embeddings/",
                        help="Directory containing precomputed .npy embedding files")
    parser.add_argument("--out", default="submission.csv",
                        help="Output CSV path")
    parser.add_argument("--top-k", type=int, default=100,
                        help="Number of candidates to include in submission")
    parser.add_argument("--validate", action="store_true", default=True,
                        help="Run format validation after writing CSV")
    args = parser.parse_args()

    WALL_START = time.time()

    print(f"\n{'='*60}")
    print(f"  SignalHire AI — Online Ranking Phase")
    print(f"{'='*60}")
    print(f"  Candidates:  {args.candidates}")
    print(f"  Embeddings:  {args.embeddings}")
    print(f"  Output:      {args.out}")
    print(f"  Top-K:       {args.top_k}")
    print()

    # ── Step 1: Load embeddings ────────────────────────────────────────────
    t = time.time()
    embs = load_embeddings(args.embeddings)
    print(f"  Step 1 done: {time.time()-t:.1f}s\n")

    # ── Step 2: Load candidates ────────────────────────────────────────────
    t = time.time()
    print(f"  Loading candidates from {args.candidates}...")
    candidates = load_candidates(args.candidates)
    print(f"  Loaded {len(candidates):,} candidates in {time.time()-t:.1f}s")

    # Build embedding→candidate index mapping
    emb_ids = embs["candidate_ids"]
    emb_mapping = build_candidate_index(candidates, emb_ids)
    print(f"  Step 2 done: {time.time()-t:.1f}s\n")

    # ── Step 3: Compute features ───────────────────────────────────────────
    t = time.time()
    print("  Computing all 8 features...")
    feature_matrix, antipattern_multipliers, is_honeypot, honeypot_reasons = \
        compute_all_features(candidates, embs, emb_mapping)
    print(f"  Step 3 done: {time.time()-t:.1f}s\n")

    # ── Step 4: Z-score fusion ─────────────────────────────────────────────
    t = time.time()
    print("  Fusing features (z-score + JD weights)...")
    scores = compute_composite_scores(feature_matrix, antipattern_multipliers, is_honeypot)
    print(f"  Score distribution: min={scores.min():.3f} max={scores.max():.3f} "
          f"mean={scores.mean():.3f} std={scores.std():.3f}")
    print(f"  Step 4 done: {time.time()-t:.1f}s\n")

    # ── Step 5: Write submission ───────────────────────────────────────────
    t = time.time()
    print("  Writing submission CSV...")
    write_submission(
        candidates, scores, feature_matrix, is_honeypot, honeypot_reasons,
        args.out, args.top_k
    )
    print(f"  Step 5 done: {time.time()-t:.1f}s\n")

    # ── Step 6: Validate ───────────────────────────────────────────────────
    if args.validate:
        print("  Validating output format...")
        validate_output(args.out, candidates)

    # ── Summary ────────────────────────────────────────────────────────────
    total = time.time() - WALL_START
    print(f"\n{'='*60}")
    print(f"  COMPLETE in {total:.1f}s ({total/60:.2f} min)")
    print(f"  Budget remaining: {300-total:.0f}s of 300s limit")
    print(f"  Output: {args.out}")
    print(f"{'='*60}\n")

    if total > 300:
        print("  ⚠ WARNING: Exceeded 5-minute compute budget!")
        sys.exit(1)


if __name__ == "__main__":
    main()