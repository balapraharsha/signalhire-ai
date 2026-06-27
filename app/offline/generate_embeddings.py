"""
offline/generate_embeddings.py
Phase 1 — Offline embedding generation for all 100K candidates.

Run ONCE before ranking. No time limit.
Generates precomputed .npy embedding arrays used by the online ranker.

Usage:
    python offline/generate_embeddings.py \
        --candidates ../data/candidates.jsonl.gz \
        --out ../data/embeddings/
"""

import argparse
import gzip
import json
import os
import sys
import time

import numpy as np
from tqdm import tqdm

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from online.jd_config import JD_CONCEPT_SENTENCES


def load_candidates(path: str):
    """Load candidates from .jsonl or .jsonl.gz"""
    opener = gzip.open if path.endswith(".gz") else open
    mode = "rt" if path.endswith(".gz") else "r"
    candidates = []
    with opener(path, mode, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def build_text_corpus(candidates):
    """
    For each candidate, extract the text strings we need to embed:
      1. skill_texts:       list of skill names (one per skill)
      2. title_texts:       current job title
      3. history_titles:    all career titles (for consistency scoring)
      4. summary_texts:     profile summary + all career descriptions concatenated
    
    Returns parallel arrays (index = candidate index).
    """
    skill_texts = []       # list of lists
    title_texts = []       # list of strings
    history_titles = []    # list of lists
    summary_texts = []     # list of strings

    for c in tqdm(candidates, desc="Building text corpus", unit="cand"):
        # Skills
        skills = c.get("skills", [])
        skill_names = [s["name"] for s in skills] if skills else ["unknown"]
        skill_texts.append(skill_names)

        # Current title
        title = c.get("profile", {}).get("current_title", "Unknown")
        title_texts.append(title)

        # Career history titles
        history = c.get("career_history", [])
        htitles = [h["title"] for h in history] if history else [title]
        history_titles.append(htitles)

        # Summary: profile summary + career descriptions
        parts = []
        summary = c.get("profile", {}).get("summary", "")
        if summary:
            parts.append(summary)
        for h in history:
            desc = h.get("description", "")
            if desc:
                parts.append(desc[:500])  # cap at 500 chars per role
        summary_texts.append(" ".join(parts)[:1500] if parts else title)

    return skill_texts, title_texts, history_titles, summary_texts


def embed_all(model, texts_flat, batch_size=512, desc="Embedding"):
    """Embed a flat list of strings. Returns (N, 384) float32 array."""
    all_embs = []
    for i in tqdm(range(0, len(texts_flat), batch_size), desc=desc):
        batch = texts_flat[i : i + batch_size]
        embs = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        all_embs.append(embs.astype(np.float32))
    return np.vstack(all_embs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or .jsonl.gz")
    parser.add_argument("--out", default="data/embeddings/", help="Output directory for .npy files")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5",
                        help="SentenceTransformer model to use")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"SignalHire AI — Offline Embedding Generator")
    print(f"{'='*60}")
    print(f"  Candidates:   {args.candidates}")
    print(f"  Model:        {args.model}")
    print(f"  Output dir:   {args.out}")
    print(f"  Batch size:   {args.batch_size}")
    print()

    # ── Load model ─────────────────────────────────────────────────────────
    print("Loading SentenceTransformer model...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)
    print(f"  Model loaded. Embedding dim: {model.get_sentence_embedding_dimension()}")

    # ── Load candidates ────────────────────────────────────────────────────
    t0 = time.time()
    print(f"\nLoading candidates from {args.candidates}...")
    candidates = load_candidates(args.candidates)
    print(f"  Loaded {len(candidates):,} candidates in {time.time()-t0:.1f}s")

    # ── Build text corpus ──────────────────────────────────────────────────
    print("\nBuilding text corpus...")
    skill_texts, title_texts, history_titles, summary_texts = build_text_corpus(candidates)

    # Save candidate IDs in order (critical for alignment)
    candidate_ids = [c["candidate_id"] for c in candidates]
    np.save(os.path.join(args.out, "candidate_ids.npy"),
            np.array(candidate_ids, dtype=object))
    print(f"  Saved candidate_ids.npy ({len(candidate_ids):,} IDs)")

    # ── Embed: current titles ──────────────────────────────────────────────
    print("\n[1/5] Embedding current titles...")
    t1 = time.time()
    title_embs = embed_all(model, title_texts, args.batch_size, "Titles")
    np.save(os.path.join(args.out, "title_embs.npy"), title_embs)
    print(f"  Saved title_embs.npy  shape={title_embs.shape}  {time.time()-t1:.1f}s")

    # ── Embed: summaries ───────────────────────────────────────────────────
    print("\n[2/5] Embedding summaries + career descriptions...")
    t2 = time.time()
    summary_embs = embed_all(model, summary_texts, args.batch_size, "Summaries")
    np.save(os.path.join(args.out, "summary_embs.npy"), summary_embs)
    print(f"  Saved summary_embs.npy  shape={summary_embs.shape}  {time.time()-t2:.1f}s")

    # ── Embed: skills (flattened) ──────────────────────────────────────────
    # We embed all skills flattened, then store offsets to reconstruct per-candidate
    print("\n[3/5] Embedding skills (flattened across all candidates)...")
    t3 = time.time()
    flat_skills = []
    skill_offsets = []  # (start, end) index into flat_skills for each candidate
    for names in skill_texts:
        start = len(flat_skills)
        flat_skills.extend(names)
        skill_offsets.append((start, len(flat_skills)))

    skill_embs_flat = embed_all(model, flat_skills, args.batch_size, "Skills")
    np.save(os.path.join(args.out, "skill_embs_flat.npy"), skill_embs_flat)
    np.save(os.path.join(args.out, "skill_offsets.npy"),
            np.array(skill_offsets, dtype=np.int32))
    print(f"  Saved skill_embs_flat.npy  shape={skill_embs_flat.shape}  {time.time()-t3:.1f}s")
    print(f"  Saved skill_offsets.npy")

    # ── Embed: career history titles (flattened) ───────────────────────────
    print("\n[4/5] Embedding career history titles...")
    t4 = time.time()
    flat_htitles = []
    htitle_offsets = []
    for htitles in history_titles:
        start = len(flat_htitles)
        flat_htitles.extend(htitles)
        htitle_offsets.append((start, len(flat_htitles)))

    htitle_embs_flat = embed_all(model, flat_htitles, args.batch_size, "History titles")
    np.save(os.path.join(args.out, "htitle_embs_flat.npy"), htitle_embs_flat)
    np.save(os.path.join(args.out, "htitle_offsets.npy"),
            np.array(htitle_offsets, dtype=np.int32))
    print(f"  Saved htitle_embs_flat.npy  shape={htitle_embs_flat.shape}  {time.time()-t4:.1f}s")

    # ── Embed: JD concept sentences ────────────────────────────────────────
    print("\n[5/5] Embedding JD concept sentences...")
    t5 = time.time()
    jd_embs = embed_all(model, JD_CONCEPT_SENTENCES, len(JD_CONCEPT_SENTENCES), "JD")
    np.save(os.path.join(args.out, "jd_concept_embs.npy"), jd_embs)
    print(f"  Saved jd_concept_embs.npy  shape={jd_embs.shape}  {time.time()-t5:.1f}s")

    # ── Done ───────────────────────────────────────────────────────────────
    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  All embeddings generated in {total/60:.1f} minutes")
    print(f"  Output directory: {args.out}")
    print(f"{'='*60}\n")

    # Estimate disk usage
    total_bytes = sum(
        os.path.getsize(os.path.join(args.out, f))
        for f in os.listdir(args.out)
        if f.endswith(".npy")
    )
    print(f"  Total disk usage: {total_bytes / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
