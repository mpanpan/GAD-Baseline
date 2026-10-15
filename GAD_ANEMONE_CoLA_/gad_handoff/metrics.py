"""Strict node anomaly evaluation, calibration uses VALIDATION labels only."""
import math
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve, auc,
    precision_score, recall_score, f1_score, balanced_accuracy_score, matthews_corrcoef,
    accuracy_score, confusion_matrix)


def _ranking(y, scores, node_ids, k):
    if not 1 <= k <= len(y):
        raise ValueError(f'Invalid top K={k} for {len(y)}')
    # scores descending; node ID ascending breaks ties deterministically.
    order = np.lexsort((node_ids, -scores))
    tp = int(y[order[:k]].sum())
    return tp/k, tp/int(y.sum())


def calibrate_threshold(val_y, val_scores):
    """Select threshold maximizing positive F1 on validation only; tie -> highest threshold."""
    p, r, ts = precision_recall_curve(val_y, val_scores)
    f1s = 2*p[:-1]*r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-15)
    best = np.flatnonzero(np.isclose(f1s, f1s.max(), atol=1e-12, rtol=0))
    if not len(best):
        raise ValueError('Validation threshold calibration failed')
    return float(np.max(ts[best]))


def evaluate(y, scores, val_mask, test_mask, *, node_ids=None):
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=np.float64)
    val_mask = np.asarray(val_mask, dtype=bool)
    test_mask = np.asarray(test_mask, dtype=bool)
    n = len(y)
    if any(a.shape != (n,) for a in [scores, val_mask, test_mask]):
        raise ValueError('Score/label/mask full graph alignment mismatch')
    if not np.isfinite(scores).all():
        raise ValueError('Full node scores contain NaN/Inf')
    if not np.isin(y,[-1,0,1]).all():
        raise ValueError('Unexpected labels, requires a verified encoding')
    if np.any(val_mask & test_mask):
        raise ValueError('Validation/test overlap')
    va = val_mask & (y>=0)
    te = test_mask & (y>=0)
    vy, vs = y[va], scores[va]
    ty, ss = y[te], scores[te]
    if set(np.unique(vy)) != {0,1} or set(np.unique(ty)) != {0,1}:
        raise ValueError('Both validation and test must contain positive and negative labels')
    if node_ids is None:
        node_ids = np.arange(n, dtype=np.int64)
    node_ids = np.asarray(node_ids)
    if node_ids.shape != (n,) or len(np.unique(node_ids)) != n:
        raise ValueError('Node IDs must be unique and aligned to graph order')
    ids = node_ids[te]
    out = {}
    out['n_validation'] = int(va.sum())
    out['n_test'] = int(te.sum())
    out['positives_validation'] = int(vy.sum())
    out['positives_test'] = int(ty.sum())
    out['unknown_labels_all_nodes'] = int((y==-1).sum())
    out['AUROC'] = float(roc_auc_score(ty, ss))
    out['AUPRC_AP'] = float(average_precision_score(ty, ss))
    pp, rr, _ = precision_recall_curve(ty, ss)
    out['PR_AUC_trapezoid'] = float(auc(rr, pp))
    true_k = int(ty.sum())
    out['Recall@K_oracle_K'] = true_k
    out['Precision@K_oracle'], out['Recall@K_oracle'] = _ranking(ty, ss, ids, true_k)
    for fraction, label in [(0.01,'1pct'),(0.05,'5pct')]:
        k = min(len(ty), max(1, math.ceil(len(ty) * fraction)))
        out[f'K_{label}'] = k
        out[f'Precision@{label}'], out[f'Recall@{label}'] = _ranking(ty, ss, ids, k)
    threshold = calibrate_threshold(vy, vs)
    out['validation_f1_threshold'] = threshold
    pred = (ss >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(ty, pred, labels=[0,1]).ravel()
    tpr = tp/(tp+fn)
    tnr = tn/(tn+fp)
    out.update({
        'Precision': float(precision_score(ty, pred, zero_division=0)),
        'Recall': float(recall_score(ty, pred, zero_division=0)),
        'F1': float(f1_score(ty, pred, zero_division=0)),
        'Macro-F1': float(f1_score(ty, pred, average='macro', zero_division=0)),
        'G-Mean': float(math.sqrt(tpr * tnr)),
        'Specificity': float(tnr),
        'Balanced_Accuracy': float(balanced_accuracy_score(ty, pred)),
        'MCC': float(matthews_corrcoef(ty, pred)),
        'Accuracy': float(accuracy_score(ty, pred)),
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
    })
    return out
