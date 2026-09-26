"""
predict.py — Run trained entity resolution model on test set.

Usage:
    python src/predict.py --data-dir dataset/test --model model.joblib --out-dir output

Outputs:
    output/matching_results.tsv
    output/candidate_pairs.tsv
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from blocking import CandidateIndex
from features import compute_features_batch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def load_candidate_pool(path: str, top_k: int, min_cosine: float) -> tuple[dict, CandidateIndex]:
    """Load a candidate pool TSV, build fast lookup and CandidateIndex, and free DataFrame memory."""
    t0 = time.time()
    logger.info('Loading candidate pool from %s…', path)
    df = pd.read_csv(path, sep='\t', dtype=str, encoding='utf-8').fillna('')
    ids = df['entity_id'].values
    names = df['business_name'].values
    addrs = df['business_address'].values
    countries = df['country'].values

    lookup = dict(zip(ids, zip(names, addrs, countries)))
    cand_index = CandidateIndex(
        (ids, names.tolist(), addrs.tolist(), countries),
        top_k=top_k,
        min_cosine=min_cosine,
    )
    del df, ids, names, addrs, countries
    gc.collect()
    logger.info('Loaded %s (%d records) in %.1fs', path, len(lookup), time.time() - t0)
    return lookup, cand_index


def main():
    parser = argparse.ArgumentParser(description='Run entity resolution on test set.')
    parser.add_argument('--data-dir', required=True, help='Path to test/ folder')
    parser.add_argument('--model', default='model.joblib', help='Path to model.joblib')
    parser.add_argument('--out-dir', default='output', help='Output directory')
    parser.add_argument('--chunk-size', type=int, default=20_000,
                        help='Number of S1 entities per scoring chunk')
    args = parser.parse_args()

    # 1. Load model
    logger.info('Loading model from %s…', args.model)
    payload = joblib.load(args.model)
    clf = payload['model']
    threshold = payload['threshold']
    top_k = payload.get('top_k', 10)
    min_cosine = payload.get('min_cosine', 0.20)
    logger.info('Loaded model with threshold=%.2f, top_k=%d, min_cosine=%.2f',
                threshold, top_k, min_cosine)

    # 2. Build S2 and S3 candidate indices
    s2_path = os.path.join(args.data_dir, 'test_source2.tsv')
    s3_path = os.path.join(args.data_dir, 'test_source3.tsv')
    s2_lookup, idx_s2 = load_candidate_pool(s2_path, top_k, min_cosine)
    s3_lookup, idx_s3 = load_candidate_pool(s3_path, top_k, min_cosine)

    # 3. Prepare output directory and files
    os.makedirs(args.out_dir, exist_ok=True)
    match_path = os.path.join(args.out_dir, 'matching_results.tsv')
    cand_path = os.path.join(args.out_dir, 'candidate_pairs.tsv')

    s1_path = os.path.join(args.data_dir, 'test_source1.tsv')
    logger.info('Reading %s…', s1_path)
    s1_full = pd.read_csv(s1_path, sep='\t', dtype=str, encoding='utf-8').fillna('')
    total_s1 = len(s1_full)
    n_chunks = (total_s1 + args.chunk_size - 1) // args.chunk_size
    logger.info('Processing %d test S1 entities in %d chunks of %d…',
                total_s1, n_chunks, args.chunk_size)

    total_matches_found = 0

    with open(match_path, 'w', encoding='utf-8') as f_match, \
         open(cand_path,  'w', encoding='utf-8') as f_cand:

        # Write TSV headers
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for chunk_idx in range(n_chunks):
            c_t0 = time.time()
            start = chunk_idx * args.chunk_size
            end = min(start + args.chunk_size, total_s1)
            chunk_df = s1_full.iloc[start:end].reset_index(drop=True)

            cands_s2 = idx_s2.query(chunk_df)
            cands_s3 = idx_s3.query(chunk_df)

            pairs_raw = []
            pair_meta = []  # (s1_id, cand_id)
            chunk_candidates = {}

            for _, row in chunk_df.iterrows():
                sid = row['entity_id']
                s_name = row['business_name']
                s_addr = row['business_address']
                s_country = row['country']

                cands = set(cands_s2.get(sid, set())) | set(cands_s3.get(sid, set()))
                chunk_candidates[sid] = cands

                for cid in cands:
                    cand_info = s2_lookup.get(cid) if cid.startswith('S2-') else s3_lookup.get(cid)
                    if cand_info is None:
                        continue
                    c_name, c_addr, c_country = cand_info
                    pairs_raw.append((s_name, s_addr, s_country, c_name, c_addr, c_country))
                    pair_meta.append((sid, cid))

            chunk_matches = {sid: set() for sid in chunk_df['entity_id']}

            if pairs_raw:
                X_chunk = compute_features_batch(pairs_raw)
                probs = clf.predict_proba(X_chunk)[:, 1]
                for (sid, cid), prob in zip(pair_meta, probs):
                    if prob >= threshold:
                        chunk_matches[sid].add(cid)

            # Write chunk results line-by-line
            for sid in chunk_df['entity_id']:
                m = chunk_matches.get(sid, set())
                c = chunk_candidates.get(sid, set())
                if m:
                    total_matches_found += 1
                m_str = ','.join(sorted(m)) if m else ''
                c_str = ','.join(sorted(c)) if c else ''
                f_match.write(f"{sid}\t{m_str}\n")
                f_cand.write(f"{sid}\t{c_str}\n")

            logger.info('Chunk %d/%d (%d entities) done in %.1fs (Total matches so far: %d)',
                        chunk_idx + 1, n_chunks, len(chunk_df), time.time() - c_t0, total_matches_found)

    logger.info('Finished test prediction: %s and %s created.', match_path, cand_path)
    logger.info('Total matched S1 entities: %d / %d (%.2f%%)',
                total_matches_found, total_s1, 100.0 * total_matches_found / total_s1)


if __name__ == '__main__':
    main()
