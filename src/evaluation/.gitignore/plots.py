# src/visualization/plots.py

import numpy as np
import matplotlib.pyplot as plt

from .errormap import normalize_errormap

def show_recon_and_errormap(x, recon, errmap, title=""):
    """
    x, recon: (H,W,3), [-1,1] or [0,1]
    errmap: (H,W)
    """
    # [-1,1]이면 [0,1]로
    def to_01(img):
        if img.min() < -0.1:
            img = (img + 1.0) / 2.0
        return np.clip(img, 0, 1)

    x_v = to_01(x)
    r_v = to_01(recon)
    e_v = normalize_errormap(errmap)

    plt.figure(figsize=(10, 3))
    plt.suptitle(title)

    plt.subplot(1, 3, 1)
    plt.title("Input")
    plt.imshow(x_v)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Reconstruction")
    plt.imshow(r_v)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Errormap")
    plt.imshow(e_v, cmap="jet")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_roc_curve(fpr, tpr, auc=None):
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUROC={auc:.4f}" if auc is not None else "ROC")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (pixel-wise anomaly score)")
    plt.legend()
    plt.grid(True)
    plt.show()
