# defect_generator.py

import numpy as np
import cv2

def normalize_normal(n):
    norm = np.linalg.norm(n, axis=2, keepdims=True) + 1e-8
    return n / norm

def add_scratch(normal, obj_mask, thickness=2, strength=0.3):
    H, W, _ = normal.shape

    ys, xs = np.where(obj_mask)
    idx = np.random.randint(0, len(ys))
    y1, x1 = ys[idx], xs[idx]

    idx = np.random.randint(0, len(ys))
    y2, x2 = ys[idx], xs[idx]

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.line(mask, (x1, y1), (x2, y2), color=255, thickness=thickness)
    mask[obj_mask == 0] = 0

    defect = normal.copy()
    idx = mask == 255
    if np.sum(idx) > 0:
        defect[idx, 2] -= strength
        defect[idx] += np.random.randn(np.sum(idx), 3).astype(np.float32) * 0.05

    defect = normalize_normal(defect)
    return defect, mask

def add_dent(normal, obj_mask, radius=15, strength=0.4):
    H, W, _ = normal.shape
    ys, xs = np.where(obj_mask)
    
    idx = np.random.randint(0, len(ys))
    cy, cx = ys[idx], xs[idx]

    Y, X = np.ogrid[:H, :W]
    dist = np.sqrt((Y - cy)**2 + (X - cx)**2)
    mask = np.logical_and(dist <= radius, obj_mask)

    weight = np.exp(-(dist**2) / (2*(radius/2)**2)) * mask

    defect = normal.copy()
    defect[..., 2] -= strength * weight
    defect[..., 0] += strength * (X - cx) / (radius+1) * weight
    defect[..., 1] += strength * (Y - cy) / (radius+1) * weight

    defect = normalize_normal(defect)
    return defect, (mask.astype(np.uint8) * 255)


def generate_defects(normal, obj_mask, num_scratch=2, num_dent=1):
    defect = normal.copy()
    total_mask = np.zeros(obj_mask.shape, dtype=np.uint8)

    for _ in range(num_scratch):
        defect, m = add_scratch(defect, obj_mask)
        total_mask = np.maximum(total_mask, m)

    for _ in range(num_dent):
        defect, m = add_dent(defect, obj_mask)
        total_mask = np.maximum(total_mask, m)

    return defect, total_mask
