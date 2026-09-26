"""
features.py — Pairwise similarity features for entity resolution.

For each (S1, candidate) pair, compute:

Name features (RapidFuzz):
  - name_ratio           : WRatio (best of multiple alignments)
  - name_partial_ratio   : longest common substring ratio
  - name_token_sort_ratio: sorted token comparison
  - name_token_set_ratio : set-based token comparison
  - name_jaro_winkler    : Jaro-Winkler distance

Address features (RapidFuzz):
  - addr_ratio
  - addr_partial_ratio
  - addr_token_sort_ratio
  - addr_token_set_ratio
  Address similarities are NaN when either side is missing so a blank
  address is not scored as 0 (i.e. "different address").

Token overlap features:
  - name_jaccard         : Jaccard on word tokens
  - addr_jaccard         : Jaccard on word tokens
  - name_dice            : Dice coefficient on word tokens

Boolean features:
  - country_match        : 1 if country strings are equal
  - s1_addr_empty        : 1 if S1 address is empty
  - cand_addr_empty      : 1 if candidate address is empty
  - both_addr_empty      : 1 if both addresses are empty
  - address_missing      : 1 if either side's address is empty

Prefix features:
  - name_prefix3_match   : 1 if first 3 chars of normalized name match
  - name_prefix5_match   : 1 if first 5 chars of normalized name match

Length ratio features:
  - name_len_ratio       : min/max of name lengths (0 if both empty)
  - addr_len_ratio       : min/max of address lengths (NaN if address missing)

All similarities are in [0, 100] from RapidFuzz (or normalized to [0, 1]
for Jaccard/Dice). Features are returned as a numpy array in a fixed order
(FEATURE_NAMES defines the order).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:
    from rapidfuzz import fuzz as rfuzz
    from rapidfuzz.distance import JaroWinkler
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

from normalize import normalize_name, normalize_address, safe_str


FEATURE_NAMES: List[str] = [
    # Name (RapidFuzz ratios, scale 0–100)
    'name_ratio',
    'name_partial_ratio',
    'name_token_sort_ratio',
    'name_token_set_ratio',
    'name_jaro_winkler',
    # Address (RapidFuzz ratios, scale 0–100; NaN if address_missing)
    'addr_ratio',
    'addr_partial_ratio',
    'addr_token_sort_ratio',
    'addr_token_set_ratio',
    # Token overlap (scale 0–1)
    'name_jaccard',
    'addr_jaccard',
    'name_dice',
    # Boolean (0/1)
    'country_match',
    's1_addr_empty',
    'cand_addr_empty',
    'both_addr_empty',
    'address_missing',
    # Prefix match (0/1)
    'name_prefix3_match',
    'name_prefix5_match',
    # Length ratio (0–1)
    'name_len_ratio',
    'addr_len_ratio',
]

N_FEATURES = len(FEATURE_NAMES)

_ADDR_SIM_SLICE = slice(5, 9)  # RapidFuzz address features


def _jaccard(a: str, b: str) -> float:
    """Jaccard similarity on whitespace-split token sets."""
    ta, tb = set(a.split()), set(b.split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def _dice(a: str, b: str) -> float:
    """Dice coefficient on word-token sets."""
    ta, tb = set(a.split()), set(b.split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return 2 * inter / (len(ta) + len(tb))


def _len_ratio(a: str, b: str) -> float:
    """min/max length ratio (0 if both empty)."""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 0.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def compute_features(
    s1_name: str,
    s1_addr: str,
    s1_country: str,
    cand_name: str,
    cand_addr: str,
    cand_country: str,
) -> np.ndarray:
    """
    Compute feature vector for a single (S1, candidate) pair.
    Inputs are raw (un-normalized) strings; normalization is applied here.
    """
    s1_addr_raw = safe_str(s1_addr)
    cand_addr_raw = safe_str(cand_addr)

    n1 = normalize_name(safe_str(s1_name))
    n2 = normalize_name(safe_str(cand_name))
    a1 = normalize_address(s1_addr_raw)
    a2 = normalize_address(cand_addr_raw)
    c1 = safe_str(s1_country).strip().lower()
    c2 = safe_str(cand_country).strip().lower()

    s1_empty = (s1_addr_raw == '') or (a1 == '')
    cand_empty = (cand_addr_raw == '') or (a2 == '')
    address_missing = s1_empty or cand_empty

    feats = np.zeros(N_FEATURES, dtype=np.float32)
    idx = 0

    # --- Name RapidFuzz ---
    if _HAS_RAPIDFUZZ:
        feats[idx]   = rfuzz.WRatio(n1, n2) if n1 or n2 else 0.0;        idx += 1
        feats[idx]   = rfuzz.partial_ratio(n1, n2) if n1 or n2 else 0.0; idx += 1
        feats[idx]   = rfuzz.token_sort_ratio(n1, n2);                    idx += 1
        feats[idx]   = rfuzz.token_set_ratio(n1, n2);                     idx += 1
        feats[idx]   = JaroWinkler.similarity(n1, n2) * 100;              idx += 1
    else:
        import difflib
        sm = difflib.SequenceMatcher(None, n1, n2)
        r  = sm.ratio() * 100
        feats[idx:idx+5] = [r, r, r, r, r];                               idx += 5

    # --- Address RapidFuzz ---
    if address_missing:
        feats[_ADDR_SIM_SLICE] = np.nan
        idx += 4
    elif _HAS_RAPIDFUZZ:
        feats[idx]   = rfuzz.WRatio(a1, a2) if a1 or a2 else 0.0;        idx += 1
        feats[idx]   = rfuzz.partial_ratio(a1, a2) if a1 or a2 else 0.0; idx += 1
        feats[idx]   = rfuzz.token_sort_ratio(a1, a2);                    idx += 1
        feats[idx]   = rfuzz.token_set_ratio(a1, a2);                     idx += 1
    else:
        import difflib
        sm  = difflib.SequenceMatcher(None, a1, a2)
        r   = sm.ratio() * 100
        feats[idx:idx+4] = [r, r, r, r];                                   idx += 4

    # --- Token overlap ---
    feats[idx]   = _jaccard(n1, n2);  idx += 1
    if address_missing:
        feats[idx] = np.nan
        idx += 1
    else:
        feats[idx]   = _jaccard(a1, a2);  idx += 1
    feats[idx]   = _dice(n1, n2);     idx += 1

    # --- Boolean ---
    feats[idx]   = float(c1 == c2 and c1 != '');  idx += 1
    feats[idx]   = float(s1_empty);               idx += 1
    feats[idx]   = float(cand_empty);             idx += 1
    feats[idx]   = float(s1_empty and cand_empty); idx += 1
    feats[idx]   = float(address_missing);        idx += 1

    # --- Prefix match ---
    feats[idx]   = float(n1[:3] == n2[:3] and n1[:3] != '');  idx += 1
    feats[idx]   = float(n1[:5] == n2[:5] and n1[:5] != '');  idx += 1

    # --- Length ratio ---
    feats[idx]   = _len_ratio(n1, n2);  idx += 1
    if address_missing:
        feats[idx] = np.nan
        idx += 1
    else:
        feats[idx]   = _len_ratio(a1, a2);  idx += 1

    assert idx == N_FEATURES, f'Feature count mismatch: {idx} != {N_FEATURES}'
    return feats


def compute_features_batch(
    pairs: List[Tuple[str, str, str, str, str, str]],
) -> np.ndarray:
    """
    Compute feature matrix for a list of (s1_name, s1_addr, s1_country,
    cand_name, cand_addr, cand_country) tuples.

    Returns shape (n_pairs, N_FEATURES).
    """
    if not pairs:
        return np.zeros((0, N_FEATURES), dtype=np.float32)
    return np.stack([compute_features(*p) for p in pairs], axis=0)
