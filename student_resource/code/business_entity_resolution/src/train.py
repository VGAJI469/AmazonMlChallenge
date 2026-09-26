"""
train.py — End-to-end training pipeline for business entity resolution.

Usage:
    python src/train.py --data-dir dataset/train --model-out model.joblib

Pipeline:
1. Loads train_source1/2/3.tsv and train_ground_truth.tsv.
2. Subsamples Source 1 for classifier fitting (default 10%) and a held-out
   validation slice (default 2%) for an honest macro F0.5 estimate.
3. Builds CandidateIndex on the *full* Source 2 and Source 3 pools (recall
   depends on complete candidate generation, not a subsampled gallery).
4. Generates candidate pairs, computes pairwise RapidFuzz/token features.
5. Trains LightGBM classifier with class imbalance weighting.
6. Evaluates macro F0.5 on the held-out validation set across a threshold grid.
7. Saves trained model, optimal threshold, and metadata to model.joblib.
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
from features import compute_features_batch, FEATURE_NAMES, N_FEATURES
from evaluate import macro_f05

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def load_source_full(path: str) -> tuple[pd.DataFrame, dict]:
    """Load a complete candidate source (S2 or S3). Blocking recall needs the full pool."""
    t_start = time.time()
    df = pd.read_csv(path, sep='\t', dtype=str, encoding='utf-8').fillna('')
    lookup = dict(zip(
        df['entity_id'],
        zip(df['business_name'], df['business_address'], df['country'])
    ))
    logger.info('Loaded %s full pool (%d rows) in %.1fs', path, len(df), time.time() - t_start)
    return df, lookup


def main():
    parser = argparse.ArgumentParser(description='Train entity resolution model.')
    parser.add_argument('--data-dir', required=True, help='Path to train/ folder')
    parser.add_argument('--model-out', default='model.joblib', help='Output model path')
    parser.add_argument('--train-frac', type=float, default=0.10,
                        help='Fraction of S1 entities used to fit the classifier (10–20% recommended)')
    parser.add_argument('--val-frac', type=float, default=0.02,
                        help='Held-out fraction of S1 entities for F0.5 threshold sweep')
    parser.add_argument('--train-sample', type=int, default=None,
                        help='Optional absolute S1 count for training (overrides --train-frac)')
    parser.add_argument('--val-sample', type=int, default=None,
                        help='Optional absolute S1 count for validation (overrides --val-frac)')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Top-K neighbors from TF-IDF ANN')
    parser.add_argument('--min-cosine', type=float, default=0.20,
                        help='Min cosine similarity to keep TF-IDF candidate')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # 1. Load ground truth and establish train/val split
    logger.info('Loading ground truth from %s…', args.data_dir)
    gt_path = os.path.join(args.data_dir, 'train_ground_truth.tsv')
    gt_df = pd.read_csv(gt_path, sep='\t', dtype=str, encoding='utf-8').fillna('')

    all_s1_gt_ids = gt_df['source1_entity_id'].values
    matched_str_arr = gt_df['matched_entity_ids'].values

    rng = np.random.default_rng(args.seed)
    n_all = len(all_s1_gt_ids)
    perm = rng.permutation(n_all)

    n_train = args.train_sample if args.train_sample is not None else max(int(n_all * args.train_frac), 1)
    n_val = args.val_sample if args.val_sample is not None else max(int(n_all * args.val_frac), 1)
    n_train = min(n_train, n_all)
    n_val = min(n_val, n_all - n_train)
    train_s1_set = set(all_s1_gt_ids[perm[:n_train]])
    val_s1_set = set(all_s1_gt_ids[perm[n_train:n_train + n_val]])
    active_s1_set = train_s1_set | val_s1_set
    logger.info(
        'S1 subsample: %d train (%.1f%%) + %d val (%.1f%%) of %d entities',
        n_train, 100.0 * n_train / n_all, n_val, 100.0 * n_val / n_all, n_all,
    )

    gt: dict[str, set] = {}
    all_needed_s2: set[str] = set()
    all_needed_s3: set[str] = set()

    for sid, m_str in zip(all_s1_gt_ids, matched_str_arr):
        if sid in active_s1_set:
            if m_str:
                cids = set(m_str.split(','))
                gt[sid] = cids
                for c in cids:
                    if c.startswith('S2-'):
                        all_needed_s2.add(c)
                    else:
                        all_needed_s3.add(c)
            else:
                gt[sid] = set()

    del gt_df
    gc.collect()
    logger.info('Selected %d train S1 and %d val S1 entities. Ground truth true matches: S2=%d, S3=%d',
                len(train_s1_set), len(val_s1_set), len(all_needed_s2), len(all_needed_s3))

    # 2. Load S1 active rows
    s1_path = os.path.join(args.data_dir, 'train_source1.tsv')
    s1_full = pd.read_csv(s1_path, sep='\t', dtype=str, encoding='utf-8').fillna('')
    s1_active = s1_full[s1_full['entity_id'].isin(active_s1_set)].copy()
    s1_lookup = dict(zip(
        s1_active['entity_id'],
        zip(s1_active['business_name'], s1_active['business_address'], s1_active['country'])
    ))
    del s1_full
    gc.collect()

    # 3. Load full S2 and S3 pools (blocking recall needs the complete gallery)
    s2_full, s2_lookup = load_source_full(os.path.join(args.data_dir, 'train_source2.tsv'))
    s3_full, s3_lookup = load_source_full(os.path.join(args.data_dir, 'train_source3.tsv'))
    logger.info('Ground-truth S2/S3 IDs present in full pools: S2=%d/%d S3=%d/%d',
                sum(1 for x in all_needed_s2 if x in s2_lookup), len(all_needed_s2),
                sum(1 for x in all_needed_s3 if x in s3_lookup), len(all_needed_s3))

    # 4. Build Candidate Indices on complete S2 / S3
    logger.info('Building CandidateIndex for full S2…')
    idx_s2 = CandidateIndex(s2_full, top_k=args.top_k, min_cosine=args.min_cosine)
    del s2_full
    gc.collect()
    logger.info('Building CandidateIndex for full S3…')
    idx_s3 = CandidateIndex(s3_full, top_k=args.top_k, min_cosine=args.min_cosine)
    del s3_full
    gc.collect()

    # 5. Extract training pairs
    train_s1_df = s1_active[s1_active['entity_id'].isin(train_s1_set)].copy()
    logger.info('Generating candidate pairs for %d train S1 entities…', len(train_s1_df))
    t0 = time.time()
    cands_s2 = idx_s2.query(train_s1_df)
    cands_s3 = idx_s3.query(train_s1_df)

    pairs_raw = []
    labels = []
    train_s1_ids = list(train_s1_set)

    for sid in train_s1_ids:
        if sid not in s1_lookup:
            continue
        s1_name, s1_addr, s1_country = s1_lookup[sid]
        gold = gt.get(sid, set())
        all_cands = set(cands_s2.get(sid, set())) | set(cands_s3.get(sid, set()))
        all_cands.update(gold)

        for cid in all_cands:
            cand_info = s2_lookup.get(cid) if cid.startswith('S2-') else s3_lookup.get(cid)
            if cand_info is None:
                continue
            cand_name, cand_addr, cand_country = cand_info
            pairs_raw.append((s1_name, s1_addr, s1_country, cand_name, cand_addr, cand_country))
            labels.append(1 if cid in gold else 0)

    logger.info('Extracted %d training pairs (pos=%d, neg=%d) in %.1fs',
                len(pairs_raw), sum(labels), len(labels) - sum(labels), time.time() - t0)

    # 6. Compute features
    logger.info('Computing %d pairwise similarity features…', N_FEATURES)
    t0 = time.time()
    X_train = compute_features_batch(pairs_raw)
    y_train = np.array(labels, dtype=np.int8)
    logger.info('Features computed in %.1fs, matrix shape: %s', time.time() - t0, X_train.shape)

    # 7. Train LightGBM classifier
    import lightgbm as lgb
    logger.info('Training LightGBM classifier…')
    t0 = time.time()
    pos_weight = (y_train == 0).sum() / max(y_train.sum(), 1)
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        n_jobs=-1,
        random_state=args.seed,
        verbose=-1,
    )
    clf.fit(X_train, y_train)
    logger.info('LightGBM trained in %.1fs', time.time() - t0)

    # 8. Validation sweep on held-out split
    logger.info('Scoring validation split (%d entities)…', len(val_s1_set))
    t0 = time.time()
    val_s1_df = s1_active[s1_active['entity_id'].isin(val_s1_set)].copy()
    val_cands_s2 = idx_s2.query(val_s1_df)
    val_cands_s3 = idx_s3.query(val_s1_df)

    val_pair_map: dict[str, list] = {}
    val_s1_ids = list(val_s1_set)

    for sid in val_s1_ids:
        if sid not in s1_lookup:
            continue
        s1_name, s1_addr, s1_country = s1_lookup[sid]
        all_cands = set(val_cands_s2.get(sid, set())) | set(val_cands_s3.get(sid, set()))
        if not all_cands:
            val_pair_map[sid] = []
            continue

        v_pairs = []
        v_cids = []
        for cid in all_cands:
            cand_info = s2_lookup.get(cid) if cid.startswith('S2-') else s3_lookup.get(cid)
            if cand_info is None:
                continue
            cand_name, cand_addr, cand_country = cand_info
            v_pairs.append((s1_name, s1_addr, s1_country, cand_name, cand_addr, cand_country))
            v_cids.append(cid)

        if v_pairs:
            X_val = compute_features_batch(v_pairs)
            probs = clf.predict_proba(X_val)[:, 1]
            val_pair_map[sid] = list(zip(v_cids, probs))
        else:
            val_pair_map[sid] = []

    logger.info('Validation candidates scored in %.1fs', time.time() - t0)

    # Threshold sweep
    gt_val = {sid: gt.get(sid, set()) for sid in val_s1_ids}
    best_thr = 0.50
    best_f05 = 0.0

    thresholds = [0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    for thr in thresholds:
        preds = {}
        for sid in val_s1_ids:
            scored = val_pair_map.get(sid, [])
            preds[sid] = {cid for cid, prob in scored if prob >= thr}
        score = macro_f05(preds, gt_val)
        logger.info('Threshold %.2f -> Validation Macro F0.5 = %.4f', thr, score)
        if score > best_f05:
            best_f05 = score
            best_thr = thr

    logger.info('============================================================')
    logger.info('BEST VALIDATION MACRO F0.5 = %.4f at threshold = %.2f', best_f05, best_thr)
    logger.info('============================================================')

    # 9. Save model
    payload = {
        'model': clf,
        'threshold': best_thr,
        'best_f05': best_f05,
        'top_k': args.top_k,
        'min_cosine': args.min_cosine,
        'feature_names': FEATURE_NAMES,
    }
    joblib.dump(payload, args.model_out, compress=3)
    logger.info('Model successfully saved to %s', args.model_out)


if __name__ == '__main__':
    main()
