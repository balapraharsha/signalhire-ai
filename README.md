# SignalHire AI
**"The right candidate is a signal, not a keyword."**

India Runs Hackathon · Track 1: Data & AI Challenge

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run offline embedding generation (once, ~20 min on CPU)
```bash
python offline/generate_embeddings.py \
    --candidates data/candidates.jsonl.gz \
    --out data/embeddings/
```

### 3. Produce the ranked submission (≤5 min on CPU)
```bash
python online/rank.py \
    --candidates data/candidates.jsonl.gz \
    --embeddings data/embeddings/ \
    --out submission.csv
```

### 4. Validate before submitting
```bash
python tests/validate_sample.py           # quick validation on 50 samples
python validate_submission.py submission.csv  # official validator
```

---

## Architecture

SignalHire AI uses a **two-phase pipeline**:

### Phase 1 — Offline (no time limit)
Generates BGE-small-en embeddings for all 100K candidates.
Run once. Saves `.npy` files to `data/embeddings/`.

### Phase 2 — Online (≤5 min CPU)
Loads precomputed embeddings → computes 8 features per candidate →
z-score fusion → honeypot filter → top-100 CSV.

---

## 8 Intelligence Signals

| # | Feature | Type | Key innovation |
|---|---------|------|----------------|
| 1 | Semantic match | Embedding | BGE-small-en vs JD concept clusters |
| 2 | Profile coherence | **Novel** | Skill embeddings vs job title cosine sim |
| 3 | Career consistency | **Novel** | Sequential title trajectory scoring |
| 4 | Expertise depth | Computed | Proficiency × months × assessment scores |
| 5 | Engagement decay | **Novel** | Response rate ÷ log(platform tenure) |
| 6 | Behavioral score | Platform | 23 Redrob signals composite |
| 7 | Anti-pattern penalty | Detection | Honeypot + disqualifier enforcement |
| 8 | Logistics fit | Computed | Notice, location, salary, work mode |

---

## Project Structure

```
signalhire_code/
├── offline/
│   └── generate_embeddings.py  # Phase 1: embed all candidates
├── online/
│   ├── rank.py                 # Phase 2: main ranking entry point ← run this
│   ├── jd_config.py            # JD-derived constants and skill lists
│   ├── fusion.py               # Z-score fusion engine
│   ├── explain.py              # Fact-grounded reasoning generator
│   └── features/
│       ├── semantic.py         # Feature 1: semantic match
│       ├── coherence.py        # Feature 2: profile coherence
│       ├── consistency.py      # Feature 3: career consistency
│       ├── depth.py            # Feature 4: expertise depth
│       ├── behavioral.py       # Feature 5+6: decay + behavioral
│       ├── antipattern.py      # Feature 7: anti-pattern + honeypot
│       └── logistics.py        # Feature 8: logistics fit
├── tests/
│   └── validate_sample.py      # Validate on 50 sample candidates
├── data/
│   ├── embeddings/             # Precomputed .npy files (after Phase 1)
│   └── cache/
└── requirements.txt
```

---

## Compute Constraints

| Constraint | Limit | Our usage |
|-----------|-------|-----------|
| Runtime | ≤ 5 min | ~100 sec |
| Memory | ≤ 16 GB | ~2.8 GB |
| GPU | None | CPU only |
| Network | Off | Zero API calls |

---

## Design Decisions

### Why z-score fusion instead of fixed weights?
Hardcoding `semantic=40%, behavioral=30%` injects human bias before seeing the data. Z-score normalisation lets each feature's natural distribution determine its contribution. JD term frequency provides log-damped emphasis automatically.

### Why profile coherence?
A Marketing Analyst with TensorFlow/Kubernetes/Blockchain passes keyword filters. Our coherence score (mean cosine similarity between skill embeddings and job title embedding) scores this ~0.14 vs ~0.85 for a real ML Engineer. No LLM required.

### Why engagement decay?
`recruiter_response_rate / log(platform_tenure_months)` distinguishes genuinely engaged candidates from stale profiles. A 0.60 response rate from someone who joined 2 months ago is excellent; the same rate after 18 months signals a going-cold profile.

### Why BGE-small-en?
130MB, 384 dimensions, runs on CPU in milliseconds per batch. Outperforms larger models on retrieval tasks relative to its size. Pre-normalised embeddings make cosine similarity a simple dot product.

---

## Compute Environment
MacBook Pro M2, 16GB RAM, Python 3.11  
(also tested on: Ubuntu 22.04, 8-core Intel, 16GB RAM)

---

*India Runs Hackathon 2026 · Redrob AI · Hack2Skill*
