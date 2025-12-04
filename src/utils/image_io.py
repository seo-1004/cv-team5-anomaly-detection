# image_io.py

import cv2
import numpy as np

def denorm_to_uint8(img):
    return ((img * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)

def save_image_from_normalized(img, path):
    """[-1,1] normal map을 [0,255] RGB png로 저장"""
    vis = denorm_to_uint8(img)
    cv2.imwrite(path, vis[..., ::-1])