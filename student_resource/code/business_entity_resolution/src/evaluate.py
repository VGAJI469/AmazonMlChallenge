"""
evaluate.py — Competition-exact F0.5 formula.

F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

Computed per S1 entity, then macro-averaged over ALL S1 entities
(including singletons — predict empty → 1.0, predict anything → 0.0).
"""

from __future__ import annotations
from typing import Dict, Set


def f05_per_entity(predicted: Set[str], gold: Set[str]) -> float:
    """Compute F0.5 for a single S1 entity."""
    if not gold and not predicted:
        return 1.0          # correct singleton prediction
    if not gold:
        return 0.0          # false positive — predicted matches for a singleton
    if not predicted:
        # precision undefined; convention: recall=0, precision=1
        # F0.5 = (1.25 * 1.0 * 0.0) / (0.25 * 1.0 + 0.0) = 0
        return 0.0

    tp = len(predicted & gold)
    precision = tp / len(predicted)
    recall = tp / len(gold)

    if precision + recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def macro_f05(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> float:
    """Compute macro-average F0.5 over all S1 entities in ground_truth."""
    if not ground_truth:
        return 0.0

    total = 0.0
    for s1_id, gold in ground_truth.items():
        pred = predictions.get(s1_id, set())
        total += f05_per_entity(pred, gold)

    return total / len(ground_truth)


def load_ground_truth(path: str) -> Dict[str, Set[str]]:
    """Load ground truth TSV into {s1_id: set_of_matched_ids}."""
    import pandas as pd
    gt = pd.read_csv(path, sep='\t', dtype=str, encoding='utf-8')
    result = {}
    for _, row in gt.iterrows():
        s1 = row['source1_entity_id']
        matched = row.get('matched_entity_ids', '')
        if not isinstance(matched, str) or not matched.strip():
            result[s1] = set()
        else:
            result[s1] = set(matched.split(','))
    return result


def load_predictions(path: str) -> Dict[str, Set[str]]:
    """Load matching_results.tsv into {s1_id: set_of_matched_ids}."""
    import pandas as pd
    df = pd.read_csv(path, sep='\t', dtype=str, encoding='utf-8')
    result = {}
    for _, row in df.iterrows():
        s1 = row['source1_entity_id']
        matched = row.get('matched_entity_ids', '')
        if not isinstance(matched, str) or not matched.strip():
            result[s1] = set()
        else:
            result[s1] = set(matched.split(','))
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compute F0.5 on a validation split.')
    parser.add_argument('--predictions', required=True, help='matching_results.tsv')
    parser.add_argument('--ground-truth', required=True, help='ground_truth.tsv')
    args = parser.parse_args()

    gt = load_ground_truth(args.ground_truth)
    pred = load_predictions(args.predictions)
    score = macro_f05(pred, gt)
    print(f'Macro F0.5: {score:.6f}')
