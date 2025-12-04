# src/visualization/eval_anomaly.py

import os
import glob
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from src.autoencoder.inference import reconstruct_normal
from .errormap import compute_errormap

def evaluate_ae_on_defects(model, defects_dir, return_curve=True):
    """
    model: 학습된 AE 모델
    defects_dir: A_*.npy (결함 normal), M_*.npy (결함 mask) 저장된 디렉토리

    return:
      mean_auroc, (fpr, tpr, auc)  # return_curve=False면 두 번째는 None
    """
    a_paths = sorted(glob.glob(os.path.join(defects_dir, "A_*.npy")))
    m_paths = sorted(glob.glob(os.path.join(defects_dir, "M_*.npy")))

    assert len(a_paths) == len(m_paths), "A_*.npy와 M_*.npy 개수가 다릅니다."
    if len(a_paths) == 0:
        print("[Eval] 결함 샘플이 없습니다.")
        return float("nan"), None

    per_image_aurocs = []
    all_scores = []
    all_labels = []

    for a_path, m_path in zip(a_paths, m_paths):
        x = np.load(a_path).astype(np.float32)  # (H,W,3)
        mask = np.load(m_path)
        if mask.max() > 1:
            mask = (mask > 0).astype(np.uint8)

        recon = reconstruct_normal(model, x)
        errmap = compute_errormap(x, recon, mode="l2")

        # per-image AUROC
        scores_flat = errmap.flatten().astype(np.float32)
        labels_flat = mask.flatten().astype(np.uint8)
        if labels_flat.max() != labels_flat.min():
            au = roc_auc_score(labels_flat, scores_flat)
            per_image_aurocs.append(au)

        if return_curve:
            all_scores.append(scores_flat)
            all_labels.append(labels_flat)

    mean_auroc = float(np.mean(per_image_aurocs)) if per_image_aurocs else float("nan")

    if not return_curve:
        return mean_auroc, None

    # 전체 픽셀 기준 ROC curve
    scores_all = np.concatenate(all_scores)
    labels_all = np.concatenate(all_labels)
    fpr, tpr, thr = roc_curve(labels_all, scores_all)
    auc = roc_auc_score(labels_all, scores_all)

    return mean_auroc, (fpr, tpr, auc)
