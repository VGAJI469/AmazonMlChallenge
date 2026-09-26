"""
blocking.py — Candidate generation: country-partitioned FAISS + rule keys.

Within each country:
  1. Char n-gram hashed TF-IDF vectors (L2-normalized) indexed with FAISS.
     - IndexFlatIP when the partition has < IVF_THRESHOLD rows (exact cosine).
     - IndexIVFFlat otherwise (train on a sample, nlist ~ sqrt(N), nprobe 16).
  2. Rule inverted index (name prefix, name tokens, address numbers).

Candidates from both passes are unioned and capped at max_cands.

Run a single-country timing sanity check:
    python src/blocking.py --data-dir dataset/train --split train --country India
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import os
import re
import time
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

from normalize import normalize_name, normalize_address, safe_str

logger = logging.getLogger(__name__)

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    faiss = None
    _HAS_FAISS = False

TOP_K = 10
MIN_COSINE = 0.20
MAX_CANDIDATES = 25
NAME_PREFIX_LEN = 3
MAX_BUCKET_SIZE = 60

IVF_THRESHOLD = 100_000
NPROBE = 16
TFIDF_DIM = 128
VECTORIZE_BATCH = 50_000
SEARCH_BATCH = 4_096

STOPWORDS = {
    'inc', 'llc', 'corp', 'co', 'ltd', 'limited', 'pvt', 'private',
    'services', 'service', 'enterprises', 'enterprise', 'solutions',
    'technologies', 'group', 'company', 'and', 'the', 'of', 'sa',
    'sarl', 'sas', 'eurl', 'sci', 'pllc', 'llp', 'lp', 'assoc',
}


def _extract_keys(name: str, addr: str, country: str, prefix_len: int = NAME_PREFIX_LEN) -> Set[Tuple[str, str, str]]:
    """Extract blocking keys for an entity."""
    keys: Set[Tuple[str, str, str]] = set()
    n_norm = normalize_name(name)
    a_norm = normalize_address(addr)

    if len(n_norm) >= prefix_len:
        keys.add((country, 'p3', n_norm[:prefix_len]))
    elif n_norm:
        keys.add((country, 'p3', n_norm))

    words = [w for w in re.findall(r'\w+', n_norm) if len(w) >= 3 and w not in STOPWORDS]
    for w in words[:2]:
        keys.add((country, 'w', w))

    nums = re.findall(r'\b\d{3,6}\b', a_norm)
    for num in nums[:2]:
        keys.add((country, 'num', num))

    return keys


def _blocking_text(name: str, addr: str) -> str:
    n = normalize_name(name)
    a = normalize_address(addr)
    return (n + ' ' + a).strip() if a else n


def _faiss_add_batched(index, hasher: HashingVectorizer, texts: List[str]) -> None:
    """Encode and add vectors in chunks so a full dense matrix is never materialized."""
    for start in range(0, len(texts), VECTORIZE_BATCH):
        vecs = _vectorize_texts(hasher, texts[start:start + VECTORIZE_BATCH])
        index.add(vecs)
        del vecs


class _CountryFaissPartition:
    """FAISS index for one country, with IVF vs exact chosen by size."""

    def __init__(
        self,
        ids: np.ndarray,
        texts: List[str],
        hasher: HashingVectorizer,
        top_k: int,
        min_cosine: float,
        rng: np.random.Generator,
    ):
        self.ids = ids
        self.top_k = top_k
        self.min_cosine = min_cosine
        self.hasher = hasher
        self.n = len(ids)
        self.index_kind = 'none'
        self.index = None

        if self.n == 0 or not _HAS_FAISS:
            return

        d = TFIDF_DIM
        t0 = time.time()

        if self.n < IVF_THRESHOLD:
            self.index = faiss.IndexFlatIP(d)
            self.index_kind = 'flat'
            _faiss_add_batched(self.index, hasher, texts)
            logger.info('FlatIP partition n=%d (%.1fs)', self.n, time.time() - t0)
        else:
            nlist = max(int(math.sqrt(self.n)), 4)
            nlist = min(nlist, max(self.n // 39, 4))
            nprobe = min(max(NPROBE, 10), 20, nlist)
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            n_train = min(self.n, max(nlist * 40, min(256_000, self.n)))
            if n_train < nlist:
                n_train = self.n
            if n_train < self.n:
                train_idx = rng.choice(self.n, size=n_train, replace=False)
                train_texts = [texts[i] for i in train_idx]
            else:
                train_texts = texts
            train_vecs = _vectorize_texts(hasher, train_texts)
            index.train(train_vecs)
            del train_vecs
            _faiss_add_batched(index, hasher, texts)
            index.nprobe = nprobe
            self.index = index
            self.index_kind = 'ivf'
            logger.info(
                'IVF partition n=%d nlist=%d nprobe=%d train=%d (%.1fs)',
                self.n, nlist, nprobe, n_train, time.time() - t0,
            )
        gc.collect()

    def search(self, query_texts: List[str]) -> List[List[Tuple[str, float]]]:
        """Return per-query list of (entity_id, cosine) passing min_cosine."""
        nq = len(query_texts)
        if self.index is None or nq == 0 or self.n == 0:
            return [[] for _ in range(nq)]

        k = min(self.top_k, self.n)
        out: List[List[Tuple[str, float]]] = [[] for _ in range(nq)]
        for start in range(0, nq, SEARCH_BATCH):
            end = min(start + SEARCH_BATCH, nq)
            q = _vectorize_texts(self.hasher, query_texts[start:end])
            scores, idxs = self.index.search(q, k)
            for row, (sc, ii) in enumerate(zip(scores, idxs)):
                hits = []
                for s, j in zip(sc, ii):
                    if j < 0:
                        continue
                    if s < self.min_cosine:
                        continue
                    hits.append((str(self.ids[j]), float(s)))
                out[start + row] = hits
        return out


def _vectorize_texts(hasher: HashingVectorizer, texts: List[str]) -> np.ndarray:
    parts = []
    for start in range(0, len(texts), VECTORIZE_BATCH):
        X = hasher.transform(texts[start:start + VECTORIZE_BATCH])
        parts.append(np.ascontiguousarray(X.astype(np.float32).toarray()))
    if len(parts) == 1:
        return parts[0]
    return np.ascontiguousarray(np.vstack(parts))


class CandidateIndex:
    """
    Indexed candidate pool (for S2 or S3) allowing fast batched queries.
    """

    def __init__(
        self,
        cand_source: Union[pd.DataFrame, Tuple[np.ndarray, List[str], List[str], np.ndarray]],
        top_k: int = TOP_K,
        min_cosine: float = MIN_COSINE,
        max_cands: int = MAX_CANDIDATES,
        prefix_len: int = NAME_PREFIX_LEN,
        countries_keep: Optional[Set[str]] = None,
        seed: int = 42,
    ):
        self.top_k = top_k
        self.min_cosine = min_cosine
        self.max_cands = max_cands
        self.prefix_len = prefix_len
        self.rng = np.random.default_rng(seed)

        if isinstance(cand_source, pd.DataFrame):
            self.cand_ids = cand_source['entity_id'].values
            raw_names = cand_source['business_name'].tolist()
            raw_addrs = cand_source['business_address'].tolist()
            self.countries = cand_source['country'].values
        else:
            self.cand_ids, raw_names, raw_addrs, self.countries = cand_source

        if countries_keep is not None:
            mask = np.array([safe_str(c) in countries_keep for c in self.countries])
            self.cand_ids = self.cand_ids[mask]
            raw_names = [n for n, m in zip(raw_names, mask) if m]
            raw_addrs = [a for a, m in zip(raw_addrs, mask) if m]
            self.countries = self.countries[mask]

        logger.info('Building CandidateIndex on %d candidates…', len(self.cand_ids))

        self.buckets: Dict[Tuple[str, str, str], List[str]] = {}
        country_ids: Dict[str, List] = {}
        country_texts: Dict[str, List[str]] = {}

        for eid, name, addr, country in zip(self.cand_ids, raw_names, raw_addrs, self.countries):
            c = safe_str(country)
            keys = _extract_keys(safe_str(name), safe_str(addr), c, prefix_len=prefix_len)
            for k in keys:
                b = self.buckets.setdefault(k, [])
                if len(b) < MAX_BUCKET_SIZE:
                    b.append(eid)
            country_ids.setdefault(c, []).append(eid)
            country_texts.setdefault(c, []).append(_blocking_text(safe_str(name), safe_str(addr)))

        self.hasher = HashingVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 3),
            n_features=TFIDF_DIM,
            alternate_sign=False,
            norm='l2',
            lowercase=False,
        )

        self.partitions: Dict[str, _CountryFaissPartition] = {}
        if not _HAS_FAISS:
            logger.warning('faiss not installed — using rule-key blocking only')
        else:
            for c, texts in country_texts.items():
                ids_arr = np.array(country_ids[c], dtype=object)
                logger.info('Indexing country=%r n=%d', c, len(ids_arr))
                self.partitions[c] = _CountryFaissPartition(
                    ids_arr, texts, self.hasher, self.top_k, self.min_cosine, self.rng,
                )

        del raw_names, raw_addrs, country_ids, country_texts
        self.cand_ids = None
        self.countries = None
        gc.collect()
        logger.info(
            'CandidateIndex ready: %d keys, %d FAISS partitions',
            len(self.buckets), len(self.partitions),
        )

    def query(self, s1_df: pd.DataFrame) -> Dict[str, Set[str]]:
        """
        Retrieve candidate IDs for each row in s1_df.
        Returns: {s1_entity_id: set_of_candidate_ids}
        """
        s1_ids = s1_df['entity_id'].tolist()
        s1_names = s1_df['business_name'].tolist()
        s1_addrs = s1_df['business_address'].tolist()
        s1_countries = s1_df['country'].tolist()

        by_country: Dict[str, List[int]] = {}
        texts = []
        for i, (name, addr, country) in enumerate(zip(s1_names, s1_addrs, s1_countries)):
            texts.append(_blocking_text(safe_str(name), safe_str(addr)))
            by_country.setdefault(safe_str(country), []).append(i)

        faiss_hits: List[List[Tuple[str, float]]] = [[] for _ in s1_ids]
        for c, rows in by_country.items():
            part = self.partitions.get(c)
            if part is None:
                continue
            qtexts = [texts[i] for i in rows]
            hits = part.search(qtexts)
            for i, h in zip(rows, hits):
                faiss_hits[i] = h

        results: Dict[str, Set[str]] = {}
        for s1_id, name, addr, country, ann in zip(
            s1_ids, s1_names, s1_addrs, s1_countries, faiss_hits,
        ):
            keys = _extract_keys(
                safe_str(name), safe_str(addr), safe_str(country), prefix_len=self.prefix_len,
            )
            cand_counts: Dict[str, int] = {}
            for k in keys:
                for cid in self.buckets.get(k, []):
                    cand_counts[str(cid)] = cand_counts.get(str(cid), 0) + 1

            scored: Dict[str, float] = {}
            for cid, sim in ann:
                scored[cid] = max(scored.get(cid, 0.0), sim)
            for cid, cnt in cand_counts.items():
                scored[cid] = max(scored.get(cid, 0.0), 0.05 * cnt)

            if not scored:
                results[s1_id] = set()
            elif len(scored) <= self.max_cands:
                results[s1_id] = set(scored.keys())
            else:
                ranked = sorted(scored.keys(), key=lambda c: scored[c], reverse=True)
                results[s1_id] = set(ranked[:self.max_cands])

        return results


def generate_candidates(
    s1_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    top_k: int = TOP_K,
    min_cosine: float = MIN_COSINE,
    max_cands: int = MAX_CANDIDATES,
) -> Dict[str, Set[str]]:
    """Convenience functional interface."""
    index = CandidateIndex(
        cand_df,
        top_k=top_k,
        min_cosine=min_cosine,
        max_cands=max_cands,
    )
    return index.query(s1_df)


def _filter_country(df: pd.DataFrame, country: str) -> pd.DataFrame:
    c = country.strip().lower()
    mask = df['country'].fillna('').str.strip().str.lower() == c
    return df.loc[mask].reset_index(drop=True)


def benchmark_country_partition(
    data_dir: str,
    split: str,
    country: str,
    top_k: int = TOP_K,
    min_cosine: float = MIN_COSINE,
    max_cands: int = MAX_CANDIDATES,
) -> dict:
    """
    Build S2+S3 indexes and query all S1 rows for one country.
    Reports wall-clock time and candidate-recall vs ground truth when available.
    """
    prefix = 'train' if split == 'train' else 'test'
    s1_path = os.path.join(data_dir, f'{prefix}_source1.tsv')
    s2_path = os.path.join(data_dir, f'{prefix}_source2.tsv')
    s3_path = os.path.join(data_dir, f'{prefix}_source3.tsv')

    logger.info('Loading sources (filter country=%r)…', country)
    t_load = time.time()
    s1 = _filter_country(pd.read_csv(s1_path, sep='\t', dtype=str, encoding='utf-8').fillna(''), country)
    s2 = _filter_country(pd.read_csv(s2_path, sep='\t', dtype=str, encoding='utf-8').fillna(''), country)
    s3 = _filter_country(pd.read_csv(s3_path, sep='\t', dtype=str, encoding='utf-8').fillna(''), country)
    load_s = time.time() - t_load
    logger.info(
        'Loaded %s: S1=%d S2=%d S3=%d in %.1fs',
        country, len(s1), len(s2), len(s3), load_s,
    )

    s2_ids = set(s2['entity_id'])
    s3_ids = set(s3['entity_id'])
    t_build = time.time()
    idx_s2 = CandidateIndex(s2, top_k=top_k, min_cosine=min_cosine, max_cands=max_cands)
    del s2
    gc.collect()
    idx_s3 = CandidateIndex(s3, top_k=top_k, min_cosine=min_cosine, max_cands=max_cands)
    del s3
    gc.collect()
    build_s = time.time() - t_build

    t_query = time.time()
    c2 = idx_s2.query(s1)
    c3 = idx_s3.query(s1)
    query_s = time.time() - t_query

    n_pairs = 0
    n_empty = 0
    for sid in s1['entity_id']:
        u = set(c2.get(sid, set())) | set(c3.get(sid, set()))
        n_pairs += len(u)
        if not u:
            n_empty += 1

    recall = None
    n_gold = 0
    n_hit = 0
    gt_path = os.path.join(data_dir, f'{prefix}_ground_truth.tsv')
    if os.path.isfile(gt_path):
        gt_df = pd.read_csv(gt_path, sep='\t', dtype=str, encoding='utf-8').fillna('')
        s1_set = set(s1['entity_id'])
        in_pool = s2_ids | s3_ids
        for sid, matched in zip(gt_df['source1_entity_id'].tolist(), gt_df['matched_entity_ids'].tolist()):
            if sid not in s1_set or not matched:
                continue
            gold = {g for g in matched.split(',') if g in in_pool}
            if not gold:
                continue
            n_gold += len(gold)
            found = set(c2.get(sid, set())) | set(c3.get(sid, set()))
            n_hit += len(gold & found)
        recall = (n_hit / n_gold) if n_gold else None

    total_block = build_s + query_s
    logger.info('============================================================')
    logger.info('COUNTRY PARTITION BENCHMARK: %s (%s)', country, split)
    logger.info('  load=%.1fs  index_build=%.1fs  query=%.1fs  blocking_wall=%.1fs',
                load_s, build_s, query_s, total_block)
    logger.info('  S1=%d  candidate_pairs=%d  empty_S1=%.2f%%  avg_cands=%.2f',
                len(s1), n_pairs, 100.0 * n_empty / max(len(s1), 1),
                n_pairs / max(len(s1), 1))
    if recall is not None:
        logger.info('  candidate recall (gold IDs in this country pool): %.4f (%d/%d)',
                    recall, n_hit, n_gold)
    logger.info('============================================================')

    return {
        'country': country,
        'load_s': load_s,
        'build_s': build_s,
        'query_s': query_s,
        'blocking_wall_s': total_block,
        'n_s1': len(s1),
        'n_s2': len(s2),
        'n_s3': len(s3),
        'n_pairs': n_pairs,
        'recall': recall,
        'n_gold': n_gold,
        'n_hit': n_hit,
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )
    parser = argparse.ArgumentParser(description='Benchmark blocking on one country partition.')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--split', choices=['train', 'test'], default='train')
    parser.add_argument('--country', default='India')
    parser.add_argument('--top-k', type=int, default=TOP_K)
    parser.add_argument('--min-cosine', type=float, default=MIN_COSINE)
    parser.add_argument('--max-cands', type=int, default=MAX_CANDIDATES)
    args = parser.parse_args()
    benchmark_country_partition(
        args.data_dir, args.split, args.country,
        top_k=args.top_k, min_cosine=args.min_cosine, max_cands=args.max_cands,
    )


if __name__ == '__main__':
    main()
