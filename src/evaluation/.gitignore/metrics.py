# src/visualization/metrics.py

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def auroc_from_errormap(errmap, mask):
    """
    errmap: (H,W) anomaly score (클수록 이상)
    mask: (H,W) bool 또는 {0,1}, 1 = 결함 영역
    """
    e = errmap.astype(np.float32).ravel()
    m = mask.astype(np.uint8).ravel()
    if m.max() == m.min():
        # GT에 결함이 하나도 없으면 AUROC 정의가 애매함
        return np.nan
    return roc_auc_score(m, e)

def roc_curve_from_errormap(errmap, mask):
    e = errmap.astype(np.float32).ravel()
    m = mask.astype(np.uint8).ravel()
    fpr, tpr, thr = roc_curve(m, e)
    auc = roc_auc_score(m, e)
    return fpr, tpr, thr, auc
