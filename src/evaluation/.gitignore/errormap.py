# src/visualization/errormap.py

import numpy as np

def compute_errormap(x, recon, mode="l2"):
    """
    x, recon: (H,W,3) or (H,W) numpy, float32
    mode: "l1" or "l2"
    반환: (H,W) anomaly score map
    """
    x = x.astype(np.float32)
    recon = recon.astype(np.float32)

    if x.ndim == 3:
        diff = x - recon  # (H,W,3)
        if mode == "l1":
            err = np.mean(np.abs(diff), axis=-1)
        else:  # L2
            err = np.sqrt(np.sum(diff**2, axis=-1))
    else:
        diff = x - recon
        if mode == "l1":
            err = np.abs(diff)
        else:
            err = np.abs(diff)

    return err  # (H,W)

def normalize_errormap(errmap):
    """
    시각화를 위해 [0,1]로 normalize
    """
    e = errmap.astype(np.float32)
    mn, mx = e.min(), e.max()
    if mx - mn < 1e-8:
        return np.zeros_like(e)
    return (e - mn) / (mx - mn)
