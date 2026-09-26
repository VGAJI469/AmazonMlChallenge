# Business Entity Resolution — ML Challenge 2026

## Overview

This pipeline resolves business entities across three data sources (Source 1 = deduplicated reference; Source 2, Source 3 = noisy duplicates). For every Source 1 entity, it finds matching records in Source 2 and Source 3 using only `business_name`, `business_address`, and `country`.

## Directory Structure

```
code/business_entity_resolution/
├── src/
│   ├── normalize.py   — text normalization (legal suffixes, address abbreviations)
│   ├── blocking.py    — TF-IDF char-ngram ANN + rule-based candidate generation
│   ├── features.py    — pairwise similarity features (rapidfuzz, Jaccard, Dice)
│   ├── train.py       — trains LightGBM classifier, sweeps threshold for F0.5
│   ├── predict.py     — runs trained model on test set, writes output files
│   └── evaluate.py    — competition-exact macro F0.5 formula
├── README.md
└── requirements.txt
```

## Environment Setup

```bash
pip install -r requirements.txt
```

Python 3.9+ required.

## End-to-End Reproduction

Run all commands from the `student_resource/` directory.

### Step 1 — Train

```bash
python code/business_entity_resolution/src/train.py \
    --data-dir dataset/train \
    --model-out model.joblib \
    --train-sample 50000 \
    --val-sample 10000
```

This prints the validation F0.5 and saves `model.joblib`.

**Options:**
- `--train-sample N` — number of S1 entities to use for feature extraction (default 50,000)
- `--val-sample N`   — number of S1 entities to use for threshold sweep (default 10,000)
- `--top-k K`        — TF-IDF ANN neighbors per source (default 10)
- `--min-cosine F`   — minimum cosine similarity for TF-IDF candidates (default 0.20)

### Step 2 — Predict

```bash
python code/business_entity_resolution/src/predict.py \
    --data-dir dataset/test \
    --model model.joblib \
    --out-dir output
```

This writes:
- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`

### Step 3 — Validate

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Should print `PASS`.

## Pipeline Architecture

### Normalization (`normalize.py`)
- Lowercase; legal suffix canonicalization (LLC, Ltd, Pvt Ltd, SARL, etc.)
- Address abbreviation standardization (Street→st, Avenue→ave, etc.)
- Unicode preserved (Devanagari, French accents) — char-ngram TF-IDF handles these natively.
- NaN → empty string (never raises).

### Blocking (`blocking.py`)
Two-pass union per source (S2 and S3 treated separately):

1. **TF-IDF char (2,3)-ngram ANN**: Fit TF-IDF on all candidate names; retrieve top-K neighbors by cosine similarity ≥ MIN_COSINE. Batched for memory efficiency.
2. **Rule-based prefix bucket**: Exact match on `(country, first-3-chars-of-normalized-name)` — captures cases where TF-IDF misses due to rare tokens.

### Features (`features.py`)
20 pairwise features:
- RapidFuzz: WRatio, partial_ratio, token_sort_ratio, token_set_ratio, Jaro-Winkler (name)
- RapidFuzz: WRatio, partial_ratio, token_sort_ratio, token_set_ratio (address)
- Token Jaccard and Dice (name + address)
- Country match, address-empty flags, name prefix match (3/5 chars), length ratios

### Model (`train.py`)
- LightGBM with class-weight balancing (falls back to sklearn GradientBoosting)
- Threshold sweep over [0.10, 0.90] to maximize macro F0.5 on a 20% held-out S1 split

## Constraints

- No external API calls or internet lookups.
- No cloud services.
- Model is LightGBM (Apache 2.0, trained from scratch — no parameter limit applies).
- Country treated as open string label — France in test is handled without any hardcoding.
